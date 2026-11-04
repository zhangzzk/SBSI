#!/usr/bin/env python
"""Zero-shear alignment of measured with intrinsic shape per true size x measured size (rule amendment 3). CPU."""
import json
from pathlib import Path

import numpy as np

ROOT = Path('/project/ls-gruen/users/zekang.zhang/sbsi_caches/galsbi_size_transport_20261104')
z = np.load(ROOT/'transport.npz')
old = np.load(ROOT/'transport_17713980.npz')
nc = len(z['cases'])
W = np.vstack([np.ones(nc), np.random.default_rng(20260925).multinomial(nc, np.full(nc, 1/nc), size=10_000)])
own_r, ree = z['own_r_edges'], z['true_re_edges']
re_bins = np.concatenate([[0], ree, [np.inf]])
SIZE = {'Re<0.15': (0, 0.15), '0.15-0.25': (0.15, 0.25), '0.25-0.4': (0.25, 0.4), '>=0.4': (0.4, np.inf)}
out = {'validation': {}}
for k in old.files:
    if k.endswith(('_cuts', '_own', '_truth')):
        out['validation'][k] = float(np.abs(z[k] - old[k]).max() / max(np.abs(old[k]).max(), 1e-30))
    elif k.endswith('_joint'):
        out['validation'][k] = float(np.abs(z[k][..., :3] - old[k]).max() / max(np.abs(old[k]).max(), 1e-30))
worst = max(out['validation'].values())
print(f'validation: earlier fields repeat, max relative difference {worst:.1e}')
if worst > 1e-6:
    raise SystemExit('validation failed')


def align(name, sz):   # case x own radius bin, MAG_AUTO < 25 (zero leg): sums of x s0 and x x
    sel = [i for i in range(len(re_bins) - 1) if re_bins[i] >= sz[0] and re_bins[i + 1] <= sz[1]]
    a = z[f'{name}_joint'][:, sel][..., :2, :].sum((1, 2, 4))
    return a[..., 3], a[..., 4], a[..., 1]


lo = lambda t: int(np.flatnonzero(np.isclose(own_r, t))[0]) + 1
out['alignment'] = {}
for g, sz in SIZE.items():
    print(f'\n== true {g}: alignment slope sum(x s0)/sum(x x) at zero shear, by own measured radius '
          f'(MAG_AUTO < 25): sim | steps4 | nll_only | steps4/sim')
    rows = []
    xs_s, xx_s, n_s = align('sim', sz)
    for i in range(lo(2.2), lo(4.0) + 1):
        if n_s[:, i].sum() < 500:
            continue
        row = {'radius_lo': float(own_r[i - 1]), 'n_sim': float(n_s[:, i].sum())}
        txt = f'{own_r[i-1]:.1f} n {n_s[:, i].sum():7.0f}'
        for name in ('sim', 'steps4', 'nll_only'):
            xs, xx, _ = align(name, sz)
            v = (W @ xs[:, i]) / (W @ xx[:, i])
            row[name] = [float(v[0]), float(v[1:].std(ddof=1))]
            txt += f' | {v[0]:7.2f}±{v[1:].std(ddof=1):.2f}'
        r = ((W @ align('steps4', sz)[0][:, i]) / (W @ align('steps4', sz)[1][:, i])) / \
            ((W @ xs_s[:, i]) / (W @ xx_s[:, i]))
        row['ratio_steps4_sim'] = [float(r[0]), float(r[1:].std(ddof=1))]
        txt += f' | {r[0]:.3f}±{r[1:].std(ddof=1):.3f}'
        rows.append(row)
        print(txt)
    out['alignment'][g] = rows
(ROOT/'alignment.json').write_text(json.dumps(out, indent=1) + '\n')
