import math

import torch

from mix import GROUP, META, fake_quant
from keys_ablation import TOKENS, per_channel
from rdkv import pool

BITS_V, QV = (0, 2, 4, 8, 16), (2, 4, 8)
BITS_K, QK = (0, 2, 3, 4, 8, 16), (2, 3, 4, 8)
RDKV_K = (0, 1, 3, 4, 5)  # indices of RDKV's key widths {0, 2, 4, 8, 16} in BITS_K
SHARES = tuple(round(0.20 + 0.05 * i, 2) for i in range(13))
N_QUERIES = 64
ITERS = 64


def fixed_bits(head_dim):
    return 4 * 32 + 3 * head_dim


def value_costs(head_dim):
    return torch.tensor([0 if b == 0 else b * head_dim + (head_dim // GROUP * META if b in QV else 0)
                         for b in BITS_V], dtype=torch.float64)


def key_costs(n):
    n = n.double()
    groups = torch.ceil(n / TOKENS)
    return torch.stack([n * b + (META * groups if b in QK else 0) for b in BITS_K], -1)


def solve(value, cost, budget, iters=ITERS):
    v, c = value.double(), cost.double().expand_as(value)
    budget = torch.as_tensor(budget, dtype=torch.float64, device=v.device).expand(v.shape[0]).clamp(min=0)
    pos = torch.where(c > 0, c, torch.full_like(c, float('inf'))).amin(-1).amin(-1)
    spread = (v.amax(-1) - v.amin(-1)).amax(-1)
    lo, hi = torch.zeros_like(budget), (2 * spread / pos).clamp(min=1e-30)

    def choose(lam):
        return (v + lam[:, None, None] * c).argmin(-1)

    for _ in range(iters):
        mid = (lo + hi) / 2
        feasible = c.gather(-1, choose(mid)[..., None])[..., 0].sum(-1) <= budget
        hi, lo = torch.where(feasible, mid, hi), torch.where(feasible, lo, mid)
    return choose(hi)


def quant_values(values, vbits):
    out = values.clone()
    for b in QV:
        sel = vbits == b
        if sel.any():
            out[sel] = fake_quant(values[sel].float(), b).to(values.dtype)
    return out


def quant_keys(keys, body, kbits):
    out = keys.clone()
    for h in range(keys.shape[0]):
        idx = torch.nonzero(body[h]).flatten()
        if not len(idx):
            continue
        block = keys[h, idx].float()
        q = block.clone()
        for b in QK:
            ch = torch.nonzero(kbits[h] == b).flatten()
            if len(ch):
                q[:, ch] = per_channel(block[:, ch], b)
        q[:, kbits[h] == 0] = 0
        out[h, idx] = q.to(keys.dtype)
    return out


def eps_sums(keys, values):
    v, k = values.float(), keys.float()
    err_v = torch.stack([((fake_quant(v, b) - v) ** 2).sum((1, 2)) for b in QV], -1)
    err_k = torch.stack([torch.stack([((per_channel(k[h], b) - k[h]) ** 2).sum() for h in range(k.shape[0])])
                         for b in QK], -1)
    return err_v, (v ** 2).sum((1, 2)), err_k, (k ** 2).sum((1, 2))


def eps_tables(err_v, energy_v, err_k, energy_k):
    def table(err, energy):
        mid = err / energy.clamp(min=1e-12)[:, None]
        return torch.cat([torch.ones_like(mid[:, :1]), mid, torch.zeros_like(mid[:, :1])], -1)
    return table(err_v, energy_v), table(err_k, energy_k)


def sample_positions(n_tokens, n_sink, n=N_QUERIES):
    return torch.linspace(n_sink, n_tokens - 1, min(n, max(1, n_tokens - n_sink))).round().long().unique()


class Context:

    def __init__(self, keys, z, queries, eps_v, eps_k, flip, n_sink, chunk=8, qn=None):
        self.L, self.H, self.T, self.D = keys.shape
        self.n_sink, self.flip, self.chunk = n_sink, flip, chunk
        self.eps_v, self.eps_k = eps_v.double(), eps_k.double()
        u = z.float().clamp(min=0).clone()
        u[..., :n_sink] = 0
        self.u = (u / u.sum((1, 2), keepdim=True).clamp(min=1e-12)).double()
        pooled = pool(z.float().clamp(min=0))  # RDKV-hidden's value weights, sinks pooled in as there
        self.pooled_raw = pooled.double()
        pooled = pooled.clone()
        pooled[..., :n_sink] = 0
        self.u_pooled = (pooled / pooled.sum((1, 2), keepdim=True).clamp(min=1e-12)).double()
        if qn is not None:  # RDKV's linear key weight ||Q_c|| ||K_c|| / sqrt(d
            self.w_lin = (qn.double() * keys.float().norm(dim=2).double() / math.sqrt(keys.shape[-1]))
        g = queries.shape[1] // self.H
        q = queries.float().reshape(self.L, self.H, g * queries.shape[2], self.D)
        self.logits = (q @ keys.float().transpose(-1, -2) / math.sqrt(self.D)).half()
        self.qsq = (q ** 2).mean(2).double()
        self.ksq = (keys.float() ** 2).mean(2).double()
        self.zvar = self.logits.float().var((2, 3)).double().clamp(min=1e-12)
        self.w = self.qsq * self.ksq / self.D
        self.vcost = value_costs(self.D).to(keys.device)

    def budget(self, layer_bits):
        return layer_bits - self.H * (fixed_bits(self.D) + self.n_sink * 2 * self.D * 16)

    def margins(self, kept):
        out = []
        for s in range(0, self.L, self.chunk):
            lg = self.logits[s:s + self.chunk].float().masked_fill(~kept[s:s + self.chunk, :, None, :], float('-inf'))
            top = lg.topk(2, dim=-1).values
            out.append((top[..., 0] - top[..., 1]).nan_to_num(posinf=1e4, neginf=1e4).clamp(min=1e-6))
        return torch.cat(out).double()

    def flip_rate(self, m, sigma2, n):
        a, c = self.flip
        ratio = m / (sigma2.clamp(min=1e-30).sqrt() * torch.sqrt(2 * torch.log(n.clamp(min=2))))[..., None]
        return torch.sigmoid(a + c * torch.log10(ratio.clamp(min=1e-30)))

    def value_plan(self, alpha, nu, value_budget, weights='raw', per_head=False):
        if weights == 'pooled':  # unnormalized, as RDKV-hidden; each knapsack row is scale-free
            u = self.pooled_raw[..., self.n_sink:]
        else:
            u = self.u[..., self.n_sink:]
        rows = self.L * self.H if per_head else self.L
        u = u.reshape(rows, -1)
        eps = self.eps_v[:, :, None, :].expand(-1, -1, self.T - self.n_sink, -1).reshape(rows, -1, len(BITS_V))
        table = nu * u[..., None] * eps
        table[..., 0] = alpha * u
        idx = solve(table, self.vcost, value_budget)
        vbits = torch.full((self.L, self.H, self.T), 16, dtype=torch.long, device=u.device)
        vbits[..., self.n_sink:] = torch.tensor(BITS_V, device=u.device)[idx].reshape(self.L, self.H, -1)
        return vbits

    def key_plan(self, vbits, m, kappa, rho, key_budget, weights='squared', per_head=False, widths=None):
        widths = list(range(len(BITS_K))) if widths is None else list(widths)
        n_body = (vbits[..., self.n_sink:] > 0).sum(-1)
        n = (n_body + self.n_sink).double()
        if rho:
            s3 = (self.w * self.eps_k[..., 2:3]).sum(-1)
            p = self.flip_rate(m, s3, n)
            grad = -self.flip[1] / (2 * math.log(10) * s3.clamp(min=1e-30)) * (p * (1 - p)).mean(-1)
        else:
            grad = torch.zeros_like(self.zvar)
        weight = (kappa / self.zvar + rho * grad)[..., None] * self.w  # both terms are head means, so 1/H cancels
        if weights == 'linear':
            weight = self.w_lin if per_head else self.w_lin * (weight.sum(-1) / self.w_lin.sum(-1).clamp(min=1e-30))[..., None]
        rows = self.L * self.H if per_head else self.L
        eps = self.eps_k[..., widths]
        table = (weight[..., None] * eps[:, :, None, :]).reshape(rows, -1, len(widths))
        cost = key_costs(n_body)[..., widths][:, :, None, :].expand(-1, -1, self.D, -1).reshape(rows, -1, len(widths))
        idx = solve(table, cost, key_budget)
        kbits = torch.tensor([BITS_K[i] for i in widths], device=vbits.device)[idx].reshape(self.L, self.H, self.D)
        return torch.where((n_body > 0)[..., None], kbits, torch.zeros_like(kbits))

    def terms(self, vbits, kbits, m=None, weights='raw', routing=True):
        kept = vbits > 0
        m = self.margins(kept) if (m is None and routing) else m
        u = self.u_pooled if weights == 'pooled' else self.u
        idx_v = torch.searchsorted(torch.tensor(BITS_V, device=vbits.device), vbits)
        C = (u * (~kept)).sum((1, 2))
        V = (u * self.eps_v.gather(-1, idx_v) * kept).sum((1, 2))
        idx_k = torch.searchsorted(torch.tensor(BITS_K, device=kbits.device), kbits)
        body = kept[..., self.n_sink:].any(-1)
        sigma2 = torch.where(body, (self.w * self.eps_k.gather(-1, idx_k)).sum(-1), 0.0)  # sinks only: no quantized keys
        n = kept.sum(-1).double()
        S = (sigma2 / self.zvar).mean(-1)
        R = self.flip_rate(m, sigma2, n).mean(-1).mean(-1) if routing else torch.zeros_like(S)
        return C, V, S, R


VARIANT = {'key_weight': 'squared', 'budget': 'layer', 'split': 'search', 'key_bits': 'all', 'value_weight': 'raw'}


def plan(ctx, settings, layer_bits, shares=SHARES, cache=None):
    total = ctx.budget(layer_bits)
    cache, out = ({} if cache is None else cache), []
    for st in settings:
        var = {**VARIANT, **{k: st[k] for k in VARIANT if k in st}}
        per_head, routing = var['budget'] == 'head', bool(st['rho'])
        rows = ctx.L * ctx.H if per_head else ctx.L
        widths = RDKV_K if var['key_bits'] == 'rdkv' else None
        key = (st['alpha'], st['nu'], var['value_weight'], var['budget'])
        entries = cache.setdefault(key, {})
        best = None
        for s in ((0.5,) if var['split'] == 'fixed' else shares):
            if s not in entries:
                entries[s] = [ctx.value_plan(st['alpha'], st['nu'], torch.full((rows,), (1 - s) * total * ctx.L / rows),
                                             var['value_weight'], per_head), None]
            vbits = entries[s][0]
            if routing and entries[s][1] is None:
                entries[s][1] = ctx.margins(vbits > 0)
            m = entries[s][1]
            kbits = ctx.key_plan(vbits, m, st['kappa'], st['rho'], torch.full((rows,), s * total * ctx.L / rows),
                                 var['key_weight'], per_head, widths)
            C, V, S, R = ctx.terms(vbits, kbits, m, var['value_weight'], routing)
            loss = st['alpha'] * C + st['nu'] * V + st['kappa'] * S + st['rho'] * R
            cand = (loss, vbits, kbits, torch.full((ctx.L,), s, dtype=torch.float64, device=loss.device), (C, V, S, R))
            if best is None:
                best = cand
                continue
            take = cand[0] < best[0]
            best = (torch.where(take, cand[0], best[0]),
                    torch.where(take[:, None, None], cand[1], best[1]),
                    torch.where(take[:, None, None], cand[2], best[2]),
                    torch.where(take, cand[3], best[3]),
                    tuple(torch.where(take, a, b) for a, b in zip(cand[4], best[4])))
        out.append({'vbits': best[1], 'kbits': best[2], 'shares': best[3], 'terms': best[4]})
    return out


def plan_bits(vbits, kbits, n_sink, head_dim):
    H = vbits.shape[1]
    vc = value_costs(head_dim).to(vbits.device)
    idx_v = torch.searchsorted(torch.tensor(BITS_V, device=vbits.device), vbits[..., n_sink:].contiguous())
    n_body = (vbits[..., n_sink:] > 0).sum(-1)
    kc = key_costs(n_body)
    idx_k = torch.searchsorted(torch.tensor(BITS_K, device=kbits.device), kbits)
    keys = (kc[:, :, None, :].expand(-1, -1, head_dim, -1).gather(-1, idx_k[..., None])[..., 0]).sum(-1)
    return vc[idx_v].sum((1, 2)) + keys.sum(-1) + H * (fixed_bits(head_dim) + n_sink * 2 * head_dim * 16)


def rdkv_pc_plan(w_t, w_c, eps_v, eps_k, head_bits, n_sink, head_dim):
    L, H, T = w_t.shape
    half = max(0.0, (head_bits - fixed_bits(head_dim) - n_sink * 2 * head_dim * 16) / 2)
    vtable = (w_t[..., n_sink:].double()[..., None] * eps_v.double()[:, :, None, :]).reshape(L * H, T - n_sink, 5)
    idx_v = solve(vtable, value_costs(head_dim).to(w_t.device), torch.full((L * H,), half))
    vbits = torch.full((L, H, T), 16, dtype=torch.long, device=w_t.device)
    vbits[..., n_sink:] = torch.tensor(BITS_V, device=w_t.device)[idx_v].reshape(L, H, -1)
    n_body = (vbits[..., n_sink:] > 0).sum(-1).reshape(-1)
    ek = eps_k.double()[..., list(RDKV_K)]
    ktable = (w_c.double()[..., None] * ek[:, :, None, :]).reshape(L * H, head_dim, 5)
    kcost = key_costs(n_body)[:, list(RDKV_K)][:, None, :].expand(-1, head_dim, -1)
    idx_k = solve(ktable, kcost, torch.full((L * H,), half))
    kbits = torch.tensor([BITS_K[i] for i in RDKV_K], device=w_t.device)[idx_k].reshape(L, H, head_dim)
    return vbits, torch.where((n_body.reshape(L, H) > 0)[..., None], kbits, torch.zeros_like(kbits))


def histogram(vbits, kbits, n_sink):
    body = vbits[..., n_sink:]
    has = body.gt(0).any(-1, keepdim=True)
    return {'v': [int((body == b).sum()) for b in BITS_V], 'k': [int(((kbits == b) & has).sum()) for b in BITS_K]}
