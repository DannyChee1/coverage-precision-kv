from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import sys

import torch

_ROOT = str(Path(__file__).resolve().parents[1] / 'third_party' / 'kvcobra')
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from kvcobra.arch import ModelArch, locate_layers
from kvcobra.calibration import Calibration, _eigendecompose, load_calibration_texts
from kvcobra.compressor import svdq_compress
from kvcobra.hadamard import next_power_of_two
from kvcobra.method import KVCobraSpec, build_kv_cobra

SPEC = dict(variant='kl', kv_side='kv_symmetric', hadamard_seed=43, c1_rounding='floor')
META_BITS = 2 * 16  # a minimum and a maximum per padded coordinate and sequence
T_MIN = 2000
N_CAL, CAL_LEN = 32, 1024


def arch_of(model) -> ModelArch:
    cfg = model.config
    n_heads = cfg.num_attention_heads
    head_dim = getattr(cfg, 'head_dim', None) or cfg.hidden_size // n_heads
    return ModelArch(model_id=getattr(cfg, '_name_or_path', ''), n_layers=cfg.num_hidden_layers, n_heads=n_heads,
                     n_kv_heads=getattr(cfg, 'num_key_value_heads', n_heads), head_dim=head_dim,
                     hidden_size=cfg.hidden_size)


@torch.no_grad()
def collect(model, tokenizer, texts, max_len=CAL_LEN):
    arch = arch_of(model)
    nH, nQ, d = arch.n_kv_heads, arch.n_heads, arch.head_dim
    group = max(1, nQ // nH)
    k_data, v_data, q_data, hooks = {}, {}, {}, []

    def kv_hook(li, target):
        def fn(_mod, _args, out):
            x = out.reshape(out.shape[0], out.shape[1], nH, d)
            for h in range(nH):
                target.setdefault((li, h), []).append(x[0, :, h, :].detach().cpu())
        return fn

    def q_hook(li):
        def fn(_mod, _args, out):
            x = out.reshape(out.shape[0], out.shape[1], nQ, d)
            for kh in range(nH):
                q_data.setdefault((li, kh), []).append(x[0, :, kh * group:(kh + 1) * group, :].mean(dim=1).detach()
                                                       .cpu())
        return fn

    for li, layer in enumerate(locate_layers(model)):
        attn = layer.self_attn
        hooks.append(getattr(attn, 'k_norm', attn.k_proj).register_forward_hook(kv_hook(li, k_data)))
        hooks.append(attn.v_proj.register_forward_hook(kv_hook(li, v_data)))
        hooks.append(getattr(attn, 'q_norm', attn.q_proj).register_forward_hook(q_hook(li)))
    dev = next(model.parameters()).device
    try:
        for text in texts:
            ids = tokenizer(text, return_tensors='pt', max_length=max_len, truncation=True).input_ids.to(dev)
            if ids.shape[1] >= 10:
                model(input_ids=ids, use_cache=False)
    finally:
        for h in hooks:
            h.remove()
    cat = lambda data: {k: torch.cat(v, dim=0) for k, v in data.items()}
    k, v, q = cat(k_data), cat(v_data), cat(q_data)
    k_eig, k_basis = _eigendecompose(k, dev)
    v_eig, v_basis = _eigendecompose(v, dev)
    return Calibration(k_eig=k_eig, v_eig=v_eig, k_basis=k_basis, v_basis=v_basis,
                       n_cal_tokens=next(iter(k.values())).shape[0], raw_q=q)


@contextmanager
def query_variance_once(arch, calib):
    import kvcobra.method as method
    cached = getattr(calib, '_mix_q_var', None)
    if cached is None:
        cached = method.query_variance(calib.raw_q, calib.k_basis, arch.n_layers, arch.n_kv_heads)
        calib._mix_q_var = cached
    original = method.query_variance
    method.query_variance = lambda *args, **kwargs: cached
    try:
        yield
    finally:
        method.query_variance = original


def compressor(arch, calib, budget_bits_per_side):
    spec = KVCobraSpec(bits_per_dim=(budget_bits_per_side + 0.5) / arch.head_dim, **SPEC)
    with query_variance_once(arch, calib):
        return build_kv_cobra(arch, calib, spec)


def stored_bits(comp):
    per_token = meta = 0
    for cfgs in (comp.k_configs, comp.v_configs):
        for cfg in cfgs.values():
            padded = next_power_of_two(int(cfg['V_r'].shape[-1]))
            per_token += padded * int(cfg['bits'])
            meta += padded * META_BITS
    return per_token, meta


def layer_bits(comp, layer, n_tokens):
    total = 0
    for cfgs in (comp.k_configs, comp.v_configs):
        for (li, _), cfg in cfgs.items():
            if li == layer:
                padded = next_power_of_two(int(cfg['V_r'].shape[-1]))
                total += padded * (int(cfg['bits']) * n_tokens + META_BITS)
    return total


def model_budget_bits(arch, n_tokens, fraction):
    return fraction * arch.n_layers * arch.n_kv_heads * n_tokens * 2 * 16 * arch.head_dim


def side(arch, calib, which, budget_bits):
    spec = KVCobraSpec(bits_per_dim=(budget_bits + 0.5) / arch.head_dim,
                       **{**SPEC, 'kv_side': 'k_only' if which == 'k' else 'v_only'})
    with query_variance_once(arch, calib):
        comp = build_kv_cobra(arch, calib, spec)
    return comp.k_configs if which == 'k' else comp.v_configs


def side_bits(cfgs):
    per_token = sum(next_power_of_two(int(c['V_r'].shape[-1])) * int(c['bits']) for c in cfgs.values())
    meta = sum(next_power_of_two(int(c['V_r'].shape[-1])) * META_BITS for c in cfgs.values())
    return per_token, meta


def fit(arch, calib, fraction, t_min=T_MIN):
    nominal = int(round(fraction * 16 * arch.head_dim))
    limit = model_budget_bits(arch, t_min, fraction)
    options = {w: {b: side(arch, calib, w, b) for b in range(4, nominal + 1)} for w in 'kv'}
    used = {w: {b: side_bits(c) for b, c in options[w].items()} for w in 'kv'}
    best = None
    for bk, (kt, km) in used['k'].items():
        for bv, (vt, vm) in used['v'].items():
            total = (kt + vt) * t_min + km + vm
            if total <= limit:
                key = (total, -abs(bk - bv), bk)
                if best is None or key > best[0]:
                    best = (key, bk, bv)
    if best is None:
        raise ValueError(f'no KV-COBRA budget fits {fraction}')
    _, bk, bv = best
    from kvcobra.compressor import KVCompressor
    return (bk, bv), KVCompressor(options['k'][bk], options['v'][bv], arch.n_layers, arch.n_kv_heads, arch.head_dim)


def table_hash(comp):
    rows = [(side, li, hi, int(c['V_r'].shape[-1]), int(c['bits']))
            for side, cfgs in (('k', comp.k_configs), ('v', comp.v_configs)) for (li, hi), c in sorted(cfgs.items())]
    return hashlib.sha256(repr(rows).encode()).hexdigest()[:16]


def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def rope_angles(model, n_tokens, device):
    pos = torch.arange(n_tokens, device=device)[None]
    dummy = torch.zeros(1, n_tokens, 1, device=device, dtype=torch.float32)
    cos, sin = model.model.rotary_emb(dummy, pos)
    cos, sin = cos[0].float(), sin[0].float()
    if not torch.allclose(cos ** 2 + sin ** 2, torch.ones_like(cos), atol=1e-4):
        raise ValueError('RoPE with attention scaling: rotating the cached keys back would not be exact')
    return cos, sin


@torch.no_grad()
def compress_layer(keys, values, comp, layer, cos, sin):
    k_pre = keys.float() * cos - rotate_half(keys.float()) * sin
    k_out, v_out = keys.clone(), values.clone()
    for h in range(keys.shape[0]):
        kc, vc = comp.k_configs.get((layer, h)), comp.v_configs.get((layer, h))
        if kc is not None:
            k_hat = KVCompressorHead.apply(k_pre[h], kc)
            k_out[h] = (k_hat * cos + rotate_half(k_hat) * sin).to(keys.dtype)
        if vc is not None:
            v_out[h] = KVCompressorHead.apply(values[h].float(), vc).to(values.dtype)
    return k_out, v_out


class KVCompressorHead:

    @staticmethod
    def apply(x, cfg):
        signs = cfg.get('hadamard_signs')
        signs = signs.to(x.device) if signs is not None else None
        return svdq_compress(x, cfg['V_r'].to(x.device).float(), int(cfg['bits']), hadamard_signs=signs)


def calibration_texts():
    if os.environ.get('KVCOBRA_TEXTS'):
        return json.loads(Path(os.environ['KVCOBRA_TEXTS']).read_text())
    return load_calibration_texts(n_samples=N_CAL, min_length=200)
