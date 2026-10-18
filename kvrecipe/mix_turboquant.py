import math
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'third_party' / 'turboquant'))
from turboquant.compressors import TurboQuantCompressorMSE, TurboQuantCompressorV2

COMMIT = '999713889a18c0ffa20c62a65e7cbbe5746794e3'
NORM_BITS = 16
KINDS = ('mse', 'ip')
_COMPRESSORS = {}


def entry_bits(kbits, vbits, keys, head_dim=128):
    return head_dim * (kbits + vbits) + NORM_BITS * (3 if keys == 'ip' else 2)


def compressor(kind, bits, layer, head_dim, device):
    key = (kind, bits, layer, head_dim, str(device))
    if key not in _COMPRESSORS:
        cls = TurboQuantCompressorV2 if kind == 'ip' else TurboQuantCompressorMSE
        _COMPRESSORS[key] = cls(head_dim, bits, seed=layer, device=device)
    return _COMPRESSORS[key]


def keys_hat(x, bits, kind, layer):
    c = compressor(kind, bits, layer, x.shape[-1], x.device)
    if kind == 'mse':
        return c.decompress(c.compress(x[None, None].float()))[0, 0]
    out = c.compress(x[None, None].float())
    scale = math.sqrt(math.pi / 2) / c.S.shape[0]
    return (out['k_mse'][0, 0].float()
            + scale * out['residual_norm'][0, 0].float()[:, None] * (out['qjl_signs'][0, 0].float() @ c.S))


def values_hat(x, bits, layer):
    c = compressor('mse', bits, layer, x.shape[-1], x.device)
    return c.decompress(c.compress(x[None, None].float()))[0, 0]


def quantize(keys, values, sel, kbits, vbits, kind, layer):
    return (keys_hat(keys[sel], kbits, kind, layer).to(keys.dtype),
            values_hat(values[sel], vbits, layer).to(values.dtype))
