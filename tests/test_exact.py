from pathlib import Path
import sys
import types
import unittest
from unittest import mock

import torch

sys.path.insert(0, str(Path(__file__).parents[1] / 'kvrecipe'))
import mix


class ExactMaskTests(unittest.TestCase):
    def test_evicted_closed_for_every_query_and_causal_among_new_tokens(self):
        rows, kv_heads, groups, ctx, new = 3, 2, 2, 50, 4
        blocked = torch.zeros(2, kv_heads, ctx, dtype=torch.bool)  # row 2 has no evicted entries
        blocked[0, 1, [3, 7, 49]] = True
        blocked[1, 0, 10] = True
        module = types.SimpleNamespace(mix_blocked=blocked)
        query = torch.zeros(rows, kv_heads * groups, new, 8)
        key = torch.zeros(rows, kv_heads, ctx + new, 8)
        allow = mix.exact_mask(module, query, key)
        self.assertEqual(tuple(allow.shape), (rows, kv_heads * groups, new, ctx + new))
        self.assertFalse(allow[0, 2:4][..., [3, 7, 49]].any())  # KV head 1 serves query heads 2 and 3
        self.assertTrue(allow[0, 0:2][..., [3, 7, 49]].all())
        self.assertFalse(allow[1, 0:2, :, 10].any())
        self.assertTrue(allow[2, :, :, :ctx].all())
        tail = allow[:, :, :, ctx:]
        self.assertTrue(torch.equal(tail, torch.ones(new, new, dtype=torch.bool).tril().expand_as(tail)))

    def test_exact_attention_puts_zero_weight_on_evicted_entries(self):
        g = torch.Generator().manual_seed(0)
        query, key, value = (torch.randn(1, 2, 3, 8, generator=g), torch.randn(1, 1, 20, 8, generator=g),
                             torch.randn(1, 1, 20, 8, generator=g))
        blocked = torch.zeros(1, 1, 17, dtype=torch.bool)
        blocked[0, 0, :12] = True
        value[0, 0, :12] = 1e6  # any weight on an evicted entry would show
        module = types.SimpleNamespace(mix_blocked=blocked, num_key_value_groups=2, is_causal=True)
        out, _ = mix.exact_attention(module, query, key, value, None, scaling=8 ** -0.5)
        self.assertLess(out.abs().max().item(), 100)


def pre_fix_attention(module, query, key, value, scaling):
    groups = query.shape[1] // key.shape[1]
    out = torch.nn.functional.scaled_dot_product_attention(
        query, key.repeat_interleave(groups, 1), value.repeat_interleave(groups, 1),
        attn_mask=mix.exact_mask(module, query, key), scale=scaling)
    return out.transpose(1, 2)


def case(rows, kv_heads, groups, ctx, new, masked_rows, seed=0, dtype=torch.float32):
    g = torch.Generator().manual_seed(seed)
    blocked = torch.rand(masked_rows, kv_heads, ctx - 3, generator=g) < 0.7 if masked_rows else None
    module = types.SimpleNamespace(mix_blocked=blocked, num_key_value_groups=groups)
    query = torch.randn(rows, kv_heads * groups, new, 16, generator=g, dtype=dtype)
    key, value = (torch.randn(rows, kv_heads, ctx + new, 16, generator=g, dtype=dtype) for _ in range(2))
    return module, query, key, value


CASES = [(3, 2, 2, 50, 4, 2), (7, 2, 4, 40, 9, 7), (2, 1, 3, 30, 1, 1), (4, 3, 1, 25, 6, 0), (5, 2, 2, 33, 1, 3)]


class ChunkedMaskTests(unittest.TestCase):
    def test_chunk_masks_reassemble_to_exact_mask(self):
        for seed, (rows, kv_heads, groups, ctx, new, masked_rows) in enumerate(CASES):
            module, query, key, _ = case(rows, kv_heads, groups, ctx, new, masked_rows, seed)
            k_len = ctx + new
            reference = mix.exact_mask(module, query, key)
            keep = mix.keep_mask(module, rows, kv_heads, k_len, 'cpu')
            for step in (1, 2, 3, new, new + 5):
                full = torch.zeros(reference.shape, dtype=torch.bool)
                for start in range(0, new, step):
                    stop = min(new, start + step)
                    part = mix.chunk_mask(keep, rows, kv_heads, groups, new, k_len, start, stop, 'cpu')
                    full[:, :, start:stop] = part.reshape(rows, kv_heads * groups, stop - start, k_len)
                self.assertTrue(torch.equal(full, reference), (rows, kv_heads, groups, ctx, new, step))

    def test_exact_attention_matches_the_pre_fix_attention(self):
        for seed, spec in enumerate(CASES):
            module, query, key, value = case(*spec, seed=seed, dtype=torch.float64)
            reference = pre_fix_attention(module, query, key, value, 0.25)
            for limit in (1, 97, mix.MASK_ELEMENTS):
                with mock.patch.object(mix, 'MASK_ELEMENTS', limit):
                    out, _ = mix.exact_attention(module, query, key, value, None, scaling=0.25)
                self.assertLess(float((out - reference).abs().max()), 1e-12, (spec, limit))

    def test_masks_stay_within_the_element_budget(self):
        module, query, key, value = case(7, 2, 4, 40, 9, 7)
        full = query.shape[0] * query.shape[1] * query.shape[2] * key.shape[2]
        sizes = []
        sdpa = torch.nn.functional.scaled_dot_product_attention

        def recording(*args, attn_mask=None, **kwargs):
            sizes.append(attn_mask.numel())
            return sdpa(*args, attn_mask=attn_mask, **kwargs)
        with mock.patch.object(mix, 'MASK_ELEMENTS', full // 4), \
                mock.patch.object(torch.nn.functional, 'scaled_dot_product_attention', recording):
            mix.exact_attention(module, query, key, value, None, scaling=0.25)
        self.assertEqual(len(sizes), 5)  # 9 queries, 2 per call
        self.assertLessEqual(max(sizes), full // 4)
        self.assertEqual(sum(sizes), full)


if __name__ == '__main__':
    unittest.main()
