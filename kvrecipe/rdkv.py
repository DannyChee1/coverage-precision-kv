import torch
import torch.nn.functional as F

BITS = (0, 2, 4, 8, 16)
QBITS = (2, 4, 8)
META = 32  # FP16 scale + FP16 zero per quantized group
POOL = 5
ITERS = 64


def pool(w, kernel=POOL):
    shape = w.shape
    return F.avg_pool1d(w.reshape(-1, 1, shape[-1]).float(), kernel, stride=1, padding=kernel // 2).reshape(shape)


def minmax_quant(x, bits, dim):
    x = x.float()
    lo = x.amin(dim, keepdim=True).half().float()
    scale = ((x.amax(dim, keepdim=True) - lo) / (2 ** bits - 1)).clamp(min=1e-6).half().float()
    return ((x - lo) / scale).round().clamp(0, 2 ** bits - 1) * scale + lo


def unit_costs(n_values):
    return torch.tensor([b * n_values + (META if b in QBITS else 0) for b in BITS], dtype=torch.float64)


def mckp(weights, eps, cost, budget, iters=ITERS):
    w = weights.double()
    eps = eps.double().to(w.device)
    cost = cost.double().to(w.device)
    cost = cost.expand(w.shape[0], -1) if cost.dim() == 1 else cost
    budget = torch.as_tensor(budget, dtype=torch.float64, device=w.device).expand(w.shape[0])
    positive = torch.where(cost > 0, cost, torch.full_like(cost, float('inf'))).amin(-1)
    lo = torch.zeros_like(budget)
    hi = (w.amax(-1).clamp(min=1e-30) * 2 / positive).clamp(min=1e-30)  # every unit takes 0 bits at or above this

    def choose(lam):
        obj = w[..., None] * eps[:, None, :] + lam[:, None, None] * cost[:, None, :]
        return obj.argmin(-1)

    def total(idx):
        return torch.gather(cost, 1, idx).sum(-1)

    for _ in range(iters):
        mid = (lo + hi) / 2
        feasible = total(choose(mid)) <= budget
        hi = torch.where(feasible, mid, hi)
        lo = torch.where(feasible, lo, mid)
    return choose(hi)


def allocate_layer(keys, values, w_t, w_c, eps_v, eps_k, head_budget_bits, n_sink=0):
    H, T, D = keys.shape
    bits = torch.tensor(BITS, device=keys.device)
    sink_bits = n_sink * 2 * D * 16
    half = max(0.0, (head_budget_bits - sink_bits) / 2)
    cost_v = unit_costs(D)
    idx_v = mckp(w_t[:, n_sink:], eps_v, cost_v, torch.full((H,), half))
    vbits = torch.full((H, T), 16, dtype=torch.long, device=keys.device)
    vbits[:, n_sink:] = bits[idx_v]
    kept = vbits > 0
    values_q = values.clone()
    for b in QBITS:
        sel = vbits == b
        if sel.any():
            values_q[sel] = minmax_quant(values[sel], b, -1).to(values.dtype)
    used = H * sink_bits + float(cost_v.to(keys.device)[idx_v].sum())
    keys_q = keys.clone()
    kbits = torch.zeros(H, D, dtype=torch.long, device=keys.device)
    for h in range(H):
        tk = torch.nonzero(kept[h, n_sink:]).flatten() + n_sink
        if not len(tk):
            continue
        cost_k = unit_costs(len(tk))
        idx_k = mckp(w_c[h:h + 1], eps_k[h:h + 1], cost_k, torch.tensor([half]))[0]
        kbits[h] = bits[idx_k]
        used += float(cost_k.to(keys.device)[idx_k].sum())
        block = keys[h, tk]  # n, D
        out = block.clone()
        for b in QBITS:
            ch = torch.nonzero(kbits[h] == b).flatten()
            if len(ch):
                out[:, ch] = minmax_quant(block[:, ch], b, 0).to(keys.dtype)
        out[:, kbits[h] == 0] = 0
        keys_q[h, tk] = out
    return {'kept': kept, 'vbits': vbits, 'kbits': kbits, 'keys': keys_q, 'values': values_q, 'bits': used}


def allocate_model(w_t, w_c, eps_v, eps_k, head_budget_bits, n_sink=0):
    L, H, T = w_t.shape
    D = w_c.shape[-1]
    bits = torch.tensor(BITS, device=w_t.device)
    sink_bits = n_sink * 2 * D * 16
    half = max(0.0, (head_budget_bits - sink_bits) / 2)
    cost_v = unit_costs(D)
    idx_v = mckp(w_t[..., n_sink:].reshape(L * H, -1), eps_v.reshape(L * H, -1), cost_v,
                 torch.full((L * H,), half))
    vbits = torch.full((L, H, T), 16, dtype=torch.long, device=w_t.device)
    vbits[..., n_sink:] = bits[idx_v].reshape(L, H, -1)
    n_kept = (vbits[..., n_sink:] > 0).sum(-1).reshape(-1).double().cpu()
    cost_k = torch.stack([b * n_kept + (META if b in QBITS else 0) for b in BITS], -1)
    idx_k = mckp(w_c.reshape(L * H, D), eps_k.reshape(L * H, -1), cost_k, torch.full((L * H,), half))
    has = (n_kept > 0).to(idx_k.device)
    kbits = torch.where(has[:, None], bits[idx_k], 0).reshape(L, H, D)
    used = (L * H * sink_bits + float(cost_v.to(w_t.device)[idx_v].sum())
            + float((torch.gather(cost_k.to(idx_k.device), 1, idx_k).sum(-1) * has).sum()))
    return vbits, kbits, used


def plan_bits(vbits, kbits, n_sink=0):
    D = kbits.shape[-1]
    body = vbits[..., n_sink:]
    quantized = lambda x: sum((x == b).sum(-1) for b in QBITS).double()
    n = (body > 0).sum(-1).double()
    values = D * body.sum(-1).double() + META * quantized(body)
    keys = n * kbits.sum(-1).double() + META * quantized(kbits) * (n > 0)
    return n_sink * 2 * D * 16 + values + keys


def apply_plan(keys, values, vbits, kbits, n_sink=0):
    values_q = values.clone()
    for b in QBITS:
        sel = vbits == b
        if sel.any():
            values_q[sel] = minmax_quant(values[sel], b, -1).to(values.dtype)
    body = vbits > 0
    body[:, :n_sink] = False
    x = keys.float()
    big = torch.finfo(torch.float32).max
    lo = torch.where(body[..., None], x, big).amin(1, keepdim=True).half().float()
    hi = torch.where(body[..., None], x, -big).amax(1, keepdim=True)
    levels = (2 ** kbits.clamp(min=1, max=8) - 1).float()[:, None, :]
    scale = ((hi - lo) / levels).clamp(min=1e-6).half().float()
    q = torch.minimum(((x - lo) / scale).round().clamp(min=0), levels) * scale + lo
    q = torch.where(kbits[:, None, :] == 16, x, q)
    q = torch.where(kbits[:, None, :] == 0, torch.zeros_like(q), q)
    keys_q = torch.where(body[..., None], q.to(keys.dtype), keys)
    return keys_q, values_q


def calibration_sums(keys, values):
    v, k = values.float(), keys.float()
    err_v = torch.stack([((minmax_quant(v, b, -1) - v) ** 2).sum((1, 2)) for b in QBITS], -1)
    err_k = torch.stack([((minmax_quant(k, b, 1) - k) ** 2).sum((1, 2)) for b in QBITS], -1)
    return err_v, (v ** 2).sum((1, 2)), err_k, (k ** 2).sum((1, 2))


def eps_tables(err_v, energy_v, err_k, energy_k):
    def table(err, energy):
        mid = err / energy.clamp(min=1e-12)[:, None]
        return torch.cat([torch.ones_like(mid[:, :1]), mid, torch.zeros_like(mid[:, :1])], -1)
    return table(err_v, energy_v), table(err_k, energy_k)
