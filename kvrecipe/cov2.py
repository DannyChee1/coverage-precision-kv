import time

import torch

from mix import N_SINK, layer_budget_bits
from improve import part_bits
import keys_ablation as ka
from stage2 import KEPT, SINK

KV2_CHUNK = 4096


def capture_class():
    from kvpress.presses.keydiff_press import KeyDiffPress
    from kvpress.presses.kvzip_press import KVzipPress
    from kvpress.utils import extract_keys_and_values

    class CaptureCov2(KVzipPress):
        kv2_rhos = {}

        def compress_post(self, model):
            self.captured = self.score_val.float().clone()  # L, 1, H, T
            self.captured_kv2, self.kv2_replayed, t0 = {}, {}, time.time()
            if not self.kv2_rhos:
                self.kv2_seconds = 0.0
                return
            plus, self.kvzip_plus_normalization = self.kvzip_plus_normalization, False
            hooks = [layer.self_attn.register_forward_hook(self.forward_hook, with_kwargs=True)
                     for layer in model.model.layers]
            try:
                proxies = self.kv2_proxies(model)
                for name, rho in self.kv2_rhos.items():
                    self.captured_kv2[name], self.kv2_replayed[name] = self.kv2_pass(model, proxies, rho)
            finally:
                for hook in hooks:
                    hook.remove()
                self.kvzip_plus_normalization = plus
            self.kv2_seconds = time.time() - t0

        def kv2_proxies(self, model):
            keydiff, out, pos = KeyDiffPress(), [], self.prefix_length
            for chunk in self._chunk_fn(self._context_ids[:, self.prefix_length:].cpu(), chunk_size=KV2_CHUNK):
                start, end = pos, pos + chunk.shape[1]
                per_layer = []
                for li, layer in enumerate(model.model.layers):
                    keys, values = extract_keys_and_values(self._cache, li)
                    s = keydiff.score(module=layer.self_attn, hidden_states=None, keys=keys[:, :, start:end],
                                      values=values[:, :, start:end], attentions=None, kwargs={})
                    per_layer.append(s.mean(dim=1))
                out.append((start, end, torch.stack(per_layer).mean(dim=0)))
                pos = end
            return out

        def kv2_pass(self, model, proxies, rho):
            self.score_val = torch.zeros((model.config.num_hidden_layers, 1, model.config.num_key_value_heads,
                                          self.context_length), dtype=model.dtype, device=model.device)
            self.score_val[..., :self.n_sink] = 1.0
            replayed = 0
            for start, end, proxy in proxies:
                self.start_idx, self.end_idx = start, end
                n = max(int(proxy.shape[-1] * rho), 1)
                positions = (proxy.topk(n, dim=-1).indices + start).squeeze(0).sort().values
                ids = self._context_ids.index_select(1, positions.to(self._context_ids.device))
                model.model(input_ids=ids, past_key_values=self._cache)
                replayed += n
            return self.score_val.float().clone(), replayed

    return CaptureCov2


def retained_share(ebits, overhead, n_heads, n_tokens, head_dim, budget):
    sinks = n_heads * min(N_SINK, n_tokens)
    left = (layer_budget_bits(n_heads, n_tokens, head_dim, budget) - sinks * 2 * part_bits(16, head_dim)
            - n_heads * overhead)
    kept = max(0, min(n_heads * (n_tokens - min(N_SINK, n_tokens)), int(left // ebits)))
    return (kept + sinks) / (n_heads * n_tokens)


def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def unrope(k, cos, sin):
    return k * cos - rotate_half(k) * sin


def rerope(k, cos, sin):
    return k * cos + rotate_half(k) * sin


@torch.no_grad()
def prerope_perchannel(keys, kept, bits, cos, sin):
    out = keys.clone()
    for h in range(keys.shape[0]):
        idx = torch.nonzero(kept[h], as_tuple=True)[0]
        if idx.numel():
            k = unrope(keys[h, idx].float(), cos[idx], sin[idx])
            out[h, idx] = rerope(ka.per_channel(k, bits), cos[idx], sin[idx]).to(keys.dtype)
    return out


def lowrank_costs(k_configs, layer, n_heads, head_dim):
    from mix_kvcobra import next_power_of_two
    key_bits, meta = [], []
    for h in range(n_heads):
        cfg = k_configs[(layer, h)]
        padded = next_power_of_two(int(cfg['V_r'].shape[-1]))
        key_bits.append(padded * int(cfg['bits']))
        meta.append(padded * 2 * 16)
    return key_bits, meta


def lowrank_entry_bits(key_bits, head_dim, vbits=2):
    return [k + part_bits(vbits, head_dim) for k in key_bits]


def allocate_lowrank(scores, entry, meta, head_dim, budget):
    n_heads, n_tokens = scores.shape
    plan = torch.zeros_like(scores, dtype=torch.int16)
    plan[:, :N_SINK] = SINK
    left = (layer_budget_bits(n_heads, n_tokens, head_dim, budget)
            - n_heads * min(N_SINK, n_tokens) * 2 * part_bits(16, head_dim) - sum(meta) - n_heads * ka.COUNTS)
    flat = scores[:, N_SINK:].reshape(-1)
    order = torch.sort(flat.float(), descending=True, stable=True).indices
    width = n_tokens - min(N_SINK, n_tokens)
    cost = torch.tensor(entry, dtype=torch.float64, device=scores.device)[order // max(width, 1)]
    k = int((torch.cumsum(cost, 0) <= left).sum().item()) if left > 0 else 0
    body = plan[:, N_SINK:].reshape(-1)
    body[order[:k]] = KEPT
    plan[:, N_SINK:] = body.view(n_heads, width)
    return plan


def lowrank_bits(plan, entry, meta, head_dim):
    kept = (plan == KEPT).sum(dim=1).tolist()
    sinks = int((plan == SINK).sum())
    return (sum(n * e for n, e in zip(kept, entry)) + sinks * 2 * part_bits(16, head_dim) + sum(meta)
            + plan.shape[0] * ka.COUNTS)


@torch.no_grad()
def lowrank_keys(keys, kept, k_configs, layer, cos, sin):
    from mix_kvcobra import KVCompressorHead
    out = keys.clone()
    for h in range(keys.shape[0]):
        idx = torch.nonzero(kept[h], as_tuple=True)[0]
        if idx.numel():
            k = unrope(keys[h, idx].float(), cos[idx], sin[idx])
            k_hat = KVCompressorHead.apply(k, k_configs[(layer, h)])
            out[h, idx] = rerope(k_hat, cos[idx], sin[idx]).to(keys.dtype)
    return out
