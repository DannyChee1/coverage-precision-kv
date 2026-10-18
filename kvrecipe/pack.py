import torch

from mix import GROUP
from keys_ablation import TOKENS
from stage2 import KEPT, SINK

KBITS, VBITS, COUNT_BITS = 3, 2, 32


def pack_codes(codes, bits):
    shifts = torch.arange(bits, device=codes.device)
    b = ((codes.reshape(-1, 1).long() >> shifts) & 1).reshape(-1)
    b = torch.cat([b, b.new_zeros((-b.numel()) % 8)])
    return (b.reshape(-1, 8) << torch.arange(8, device=codes.device)).sum(1).to(torch.uint8)


def unpack_codes(packed, bits, n):
    b = ((packed.reshape(-1, 1).long() >> torch.arange(8, device=packed.device)) & 1).reshape(-1)[:n * bits]
    return (b.reshape(n, bits) << torch.arange(bits, device=packed.device)).sum(1)


def minmax_codes(g, bits):
    lo, hi = g.amin(-1, keepdim=True), g.amax(-1, keepdim=True)
    scale = ((hi - lo) / (2 ** bits - 1)).clamp(min=1e-8)
    return ((g - lo) / scale).round().clamp(0, 2 ** bits - 1), scale, lo


def pack_head(keys, values, plan):
    idx = torch.nonzero(plan == KEPT, as_tuple=True)[0]
    n, d = idx.numel(), keys.shape[-1]
    k = keys[idx].float()
    pad = (-n) % TOKENS
    g = torch.cat([k, k[-1:].expand(pad, d)], 0) if pad and n else k
    kq, ks, kz = minmax_codes(g.reshape(-1, TOKENS, d).transpose(1, 2), KBITS)  # groups, D, 64
    vq, vs, vz = minmax_codes(values[idx].float().reshape(n, d // GROUP, GROUP), VBITS)
    sinks = torch.nonzero(plan == SINK, as_tuple=True)[0]
    return {'n': n, 'positions': idx,
            'key_codes': pack_codes(kq.transpose(1, 2).reshape(-1, d)[:n], KBITS),
            'key_scale': ks.half(), 'key_zero': kz.half(),
            'value_codes': pack_codes(vq, VBITS), 'value_scale': vs.half(), 'value_zero': vz.half(),
            'sink_keys': keys[sinks].half(), 'sink_values': values[sinks].half()}


def head_bits(part, d):
    n, groups = part['n'], part['key_scale'].shape[0]
    return (n * d * (KBITS + VBITS) + groups * d * 2 * 16 + n * (d // GROUP) * 2 * 16
            + part['sink_keys'].numel() * 16 + part['sink_values'].numel() * 16 + 2 * COUNT_BITS)


def head_bytes(part):
    return (sum(part[k].numel() * part[k].element_size() for k in
                ('key_codes', 'key_scale', 'key_zero', 'value_codes', 'value_scale', 'value_zero', 'sink_keys',
                 'sink_values')) + 2 * COUNT_BITS // 8)


def unpack_head(part, d, dtype):
    n = part['n']
    if not n:
        return torch.zeros(0, d, dtype=dtype), torch.zeros(0, d, dtype=dtype)
    kq = unpack_codes(part['key_codes'], KBITS, n * d).reshape(n, d).float()
    group = torch.arange(n, device=kq.device) // TOKENS
    k = kq * part['key_scale'][group, :, 0].float() + part['key_zero'][group, :, 0].float()
    vq = unpack_codes(part['value_codes'], VBITS, n * d).reshape(n, d // GROUP, GROUP).float()
    v = (vq * part['value_scale'].float() + part['value_zero'].float()).reshape(n, d)
    return k.to(dtype), v.to(dtype)


def pack_layer(keys, values, plan):
    return [pack_head(keys[h], values[h], plan[h]) for h in range(keys.shape[0])]


def restore_layer(parts, n_tokens, d, dtype, device):
    keys = torch.zeros(len(parts), n_tokens, d, dtype=dtype, device=device)
    values = torch.zeros_like(keys)
    for h, part in enumerate(parts):
        k, v = unpack_head(part, d, dtype)
        keys[h, part['positions']], values[h, part['positions']] = k, v
        s = part['sink_keys'].shape[0]
        keys[h, :s], values[h, :s] = part['sink_keys'].to(dtype), part['sink_values'].to(dtype)
    return keys, values
