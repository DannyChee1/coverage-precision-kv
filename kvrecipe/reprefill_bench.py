import argparse
import json
from pathlib import Path
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).parent))
import mix
import controls as kc
from mix import N_SINK
import keys_ablation as ka
import pack as pk
from stage2 import KEPT, quantize_parts

LENGTHS = (4096, 8192, 16384, 32768)
REPEATS, PACK_LENGTH = 5, 16384


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timed(fn, repeats=REPEATS, warm=True):
    if warm:
        fn()
    out = []
    for _ in range(repeats):
        sync()
        t = time.perf_counter()
        fn()
        sync()
        out.append(time.perf_counter() - t)
    return out


def context_ids(tokenizer, rows_path, n_max):
    ids = []
    for line in open(rows_path):
        ids += tokenizer(json.loads(line)['context'], add_special_tokens=False).input_ids
        if len(ids) >= n_max:
            break
    return torch.tensor([ids[:n_max]])


def prefill(model, ids):
    from transformers import DynamicCache
    cache = DynamicCache()
    with torch.no_grad():
        model.model(input_ids=ids, past_key_values=cache)
    return cache


def score(model, ids, capture_kvzip):
    from transformers import DynamicCache
    cache = DynamicCache()
    press = capture_kvzip(compression_ratio=0.5, kvzip_plus_normalization=True, n_sink=N_SINK)
    with torch.no_grad(), press(model):
        model.model(input_ids=ids, past_key_values=cache)
    return cache, press.captured[:, 0]


def pack_cache(cache, scores, budget):
    n_layers = len(cache.layers)
    n_heads, n_tokens, d = cache.layers[0].keys.shape[1:]
    layers, accounted, worst = [], 0, 0.0
    for li in range(n_layers):
        keys, values = cache.layers[li].keys[0], cache.layers[li].values[0]
        plan = kc.allocate(scores[li], 'k3v2pc', d, budget)
        parts = pk.pack_layer(keys, values, plan)
        layers.append(parts)
        accounted += kc.plan_bits(plan, 'k3v2pc', d)
        if li == 0:
            sel = plan == KEPT
            want_k = ka.quantized_keys(keys, 'perchannel', 3, None, sel)
            want_v = quantize_parts(keys, values, (4,), (2,))[1][2]
            got_k, got_v = pk.restore_layer(parts, n_tokens, d, keys.dtype, keys.device)
            for got, want, ref in ((got_k, want_k, keys), (got_v, want_v, values)):
                diff = (got[sel].float() - want[sel].float()).abs().max()
                worst = max(worst, float(diff / (ref.float().amax() - ref.float().amin())))
    packed = sum(pk.head_bytes(p) for parts in layers for p in parts)
    bits = sum(pk.head_bits(p, d) for parts in layers for p in parts)
    return layers, {'packed_bytes': packed, 'packed_bits': bits, 'accounted_bytes': accounted / 8,
                    'accounted_bits': accounted, 'position_bitmap_bytes': n_layers * n_heads * -(-n_tokens // 8),
                    'code_round_trip_max_rel': worst}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--rows', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--lengths', type=int, nargs='+', default=list(LENGTHS))
    parser.add_argument('--pack-length', type=int, default=PACK_LENGTH)
    parser.add_argument('--repeats', type=int, default=REPEATS)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--dtype', default='bfloat16')
    args = parser.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import kvpress
    attn = 'flash_attention_2' if args.device.startswith('cuda') else 'sdpa'
    tok = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, dtype=getattr(torch, args.dtype),
                                                 attn_implementation=attn).to(args.device).eval()
    capture_kvzip = mix.capture_press_class()
    ids = context_ids(tok, args.rows, max(args.lengths + [args.pack_length])).to(args.device)
    device = torch.cuda.get_device_name() if torch.cuda.is_available() else 'cpu'
    out = {'model': model.config.name_or_path, 'device': device, 'context_source': str(args.rows),
           'repeats': args.repeats, 'prefill_seconds': {}, 'score_seconds': {}}
    for i, n in enumerate(args.lengths):
        out['prefill_seconds'][str(n)] = timed(lambda: prefill(model, ids[:, :n]), args.repeats, warm=i == 0)
        out['score_seconds'][str(n)] = timed(lambda: score(model, ids[:, :n], capture_kvzip), args.repeats,
                                             warm=i == 0)
        print(json.dumps({'done': n, 'prefill': min(out['prefill_seconds'][str(n)]),
                          'score': min(out['score_seconds'][str(n)])}), flush=True)
    cache, scores = score(model, ids[:, :args.pack_length], capture_kvzip)
    n_layers = len(cache.layers)
    n_heads, n_tokens, d = cache.layers[0].keys.shape[1:]
    full_bytes = n_layers * n_heads * n_tokens * d * 2 * 2
    out.update(pack_length=n_tokens, full_bytes=full_bytes, h2d_seconds={}, dequant_seconds={}, packing={})
    pinned = torch.empty(full_bytes, dtype=torch.uint8, pin_memory=torch.cuda.is_available())
    rates = []
    for name, size in [('full', full_bytes)] + [(b, int(full_bytes * share)) for b, share in kc.BUDGETS.items()]:
        out['h2d_seconds'][name] = timed(lambda: pinned[:size].to(args.device, non_blocking=True), args.repeats)
        rates.append(size / min(out['h2d_seconds'][name]) / 1e9)
    out['h2d_gbps'] = rates[0]
    for b, share in kc.BUDGETS.items():
        layers, stats = pack_cache(cache, scores, share)
        out['packing'][b] = stats
        out['dequant_seconds'][b] = min(timed(lambda: [pk.restore_layer(parts, n_tokens, d, cache.layers[0].keys.dtype,
                                                                         args.device) for parts in layers],
                                              args.repeats))
    out.update(packed_bytes={b: s['packed_bytes'] for b, s in out['packing'].items()},
               accounted_bytes={b: s['accounted_bytes'] for b, s in out['packing'].items()},
               position_bitmap_bytes={b: s['position_bitmap_bytes'] for b, s in out['packing'].items()},
               peak_gib=torch.cuda.max_memory_allocated() / 2 ** 30 if torch.cuda.is_available() else 0.0)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps({'done': 'pack', 'packed_bytes': out['packed_bytes'], 'dequant': out['dequant_seconds']}),
          flush=True)


if __name__ == '__main__':
    sys.exit(main())
