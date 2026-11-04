#!/usr/bin/env python
"""Per-galaxy shape response by own measured size and by true (size, magnitude); reads transport.npz (CPU)."""
import json
from pathlib import Path

import numpy as np

ROOT = Path('/project/ls-gruen/users/zekang.zhang/sbsi_caches/galsbi_size_transport_20261104')
z = np.load(ROOT/'transport.npz')
nc = len(z['cases'])
W = np.vstack([np.ones(nc), np.random.default_rng(20260925).multinomial(nc, np.full(nc, 1/nc), size=10_000)])
N, DS, Q, L, QQ, QL = 0, 7, 8, 9, 10, 11
out = {}


def ratio(a, b):   # case-weighted ratio of sums: point, se
    A, B = W @ a, W @ b
    v = A / B
    return v[0], v[1:].std(ddof=1)


def diff_ratio(a, b, c, d):   # A/B - C/D with shared case weights
    v = (W @ a) / (W @ b) - (W @ c) / (W @ d)
    return v[0], v[1:].std(ddof=1)


# 1. by own zero-leg FLUX_RADIUS, MAG_AUTO < 25 (each side binned by its own measurement)
re = z['own_r_edges']
print('== shape response per galaxy by own zero-leg FLUX_RADIUS (MAG_AUTO < 25); flow - sim')
rows = []
for j in range(1, len(re)):
    if not 2.0 <= re[j - 1] < 5.0:
        continue
    s = z['sim_own'][:, j, :2].sum(1)
    row = {'radius_lo': float(re[j - 1]), 'sim': ratio(s[:, DS], s[:, N]), 'sim_count': float(s[:, N].sum())}
    txt = f"{re[j-1]:.1f} n {s[:, N].sum():8.0f} sim {row['sim'][0]:+.3f}±{row['sim'][1]:.3f}"
    for f in ('steps4', 'nll_only'):
        a = z[f'{f}_own'][:, j, :2].sum(1)
        row[f] = diff_ratio(a[:, DS], a[:, N], s[:, DS], s[:, N])
        row[f'{f}_count_ratio'] = float(a[:, N].sum() / s[:, N].sum())
        txt += f" | {f} {row[f][0]:+.3f}±{row[f][1]:.3f} (n x{row[f'{f}_count_ratio']:.3f})"
    rows.append(row)
    print(txt)
out['own'] = rows

# 2. by true circularized Re x true magnitude (same galaxies on both sides)
ree, me = z['true_re_edges'], z['true_m_edges']
lab_r = ['<%g' % ree[0]] + [f'{a:g}-{b:g}' for a, b in zip(ree[:-1], ree[1:])] + ['>%g' % ree[-1]]
lab_m = ['<%g' % me[0]] + [f'{a:g}-{b:g}' for a, b in zip(me[:-1], me[1:])] + ['>%g' % me[-1]]
truth = {}
for f in ('steps4',):
    print(f'\n== shape response per galaxy by TRUE size (rows, arcsec) x TRUE mag (cols): sim | {f} - sim (sigma)')
    print(' ' * 10 + ''.join(f'{m:>22s}' for m in lab_m))
    for i, lr in enumerate(lab_r):
        txt = f'{lr:>10s}'
        for k in range(len(lab_m)):
            s, a = z['sim_truth'][:, i, k], z[f'{f}_truth'][:, i, k]
            if s[:, N].sum() < 2000:
                txt += f"{'-':>22s}"
                continue
            p, e = ratio(s[:, DS], s[:, N])
            d, de = diff_ratio(a[:, DS], a[:, N], s[:, DS], s[:, N])
            truth[f'{lr}|{lab_m[k]}'] = {'n': float(s[:, N].sum()), 'sim': [p, e], 'diff': [d, de]}
            txt += f'  {p:+.2f} | {d:+.3f}({d/de:+4.1f})'
        print(txt)
out['truth'] = truth

# 3. static: zero-leg ln FLUX_RADIUS vs |e_in|^2 slope, by true size (all magnitudes < 25)
print('\n== static slope d ln r / d|e_in|^2 by true size, true mag < 25 (sim / steps4)')
kk = slice(0, int(np.searchsorted(me, 25.0)) + 1)
for i, lr in enumerate(lab_r):
    vals = []
    for f in ('sim', 'steps4'):
        a = z[f'{f}_truth'][:, i, kk].sum(1)
        A = W @ a
        sl = (A[:, QL] - A[:, Q] * A[:, L] / A[:, N]) / (A[:, QQ] - A[:, Q] ** 2 / A[:, N])
        vals.append((sl[0], sl[1:].std(ddof=1)))
    print(f'{lr:>10s} sim {vals[0][0]:+.3f}±{vals[0][1]:.3f}  steps4 {vals[1][0]:+.3f}±{vals[1][1]:.3f}')
(ROOT/'shape_by_size.json').write_text(json.dumps(out, indent=1, default=float) + '\n')
