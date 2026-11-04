#!/usr/bin/env python
"""Pairing-free size-wave split by true population (rule amendment 2). Reads transport.npz (run 17713980), CPU."""
import json
from pathlib import Path

import numpy as np

DATA = Path('/project/ls-gruen/users/zekang.zhang/sbsi_caches')
ROOT = DATA/'galsbi_size_transport_20261104'
z = np.load(ROOT/'transport.npz')
old = np.load(ROOT/'transport_17713899.npz')
mp = np.load(DATA/'galsbi_guard_strength_20261103/map.npz')
nc = len(z['cases'])
W = np.vstack([np.ones(nc), np.random.default_rng(20260925).multinomial(nc, np.full(nc, 1/nc), size=10_000)])
FLOWS = ('nll_only', 'steps4', 'bands40')
own_r, ree, me = z['own_r_edges'], z['true_re_edges'], z['true_m_edges']
out = {'validation': {}}

# ---- validation: (a) everything from the previous run repeats; (b) joint summed over truth == map per radius bin
for k in old.files:
    if k.endswith(('_cuts', '_own', '_truth')):
        out['validation'][f'repeat_{k}'] = float(np.abs(z[k] - old[k]).max() / max(np.abs(old[k]).max(), 1e-30))
re_map = mp['r_edges']
l25 = int(np.flatnonzero(np.isclose(mp['m_edges'], 25.0))[0])
for name in ('sim', 'nll_only', 'steps4'):
    mtr = np.einsum('crmii->cr', mp[f'{name}_shape'][:, :, :l25 + 1])
    j = z[f'{name}_joint'][..., :2, 0].sum((1, 2, 4))                       # case, own radius bin
    worst = 0.0
    for i in range(1, len(own_r)):
        k = int(np.flatnonzero(np.isclose(re_map, own_r[i - 1]))[0]) + 1
        a, b = j[:, i].sum(), mtr[:, k].sum()
        worst = max(worst, abs(a - b) / max(abs(b), 1.0))
    out['validation'][f'joint_vs_map_{name}'] = worst
print('validation (max relative differences):', json.dumps(out['validation'], indent=0))
if max(out['validation'].values()) > 1e-6:
    raise SystemExit('validation failed')

# ---- true-size groups (circularized Re, arcsec) and true-magnitude groups
re_bins = np.concatenate([[0], ree, [np.inf]])
size_groups = {'Re<0.15': (0, 0.15), '0.15-0.25': (0.15, 0.25), '0.25-0.4': (0.25, 0.4), '>=0.4': (0.4, np.inf)}
mag_groups = {'mag<24': (0, 24), '24-25': (24, 25), '>=25': (25, np.inf)}
m_bins = np.concatenate([[0], me, [np.inf]])


def select(edges, lo, hi):
    return [i for i in range(len(edges) - 1) if edges[i] >= lo and edges[i + 1] <= hi]


def resp(name, sizes=None, mags=None):   # case x own radius bin, MAG_AUTO < 25, response sum (pairing-free)
    a = z[f'{name}_joint'][..., :2, :].sum(4)                                 # case, re, m, rbin, stat
    if sizes is not None:
        a = a[:, sizes]
    if mags is not None:
        a = a[:, :, mags]
    return a.sum((1, 2))


T = 3.0
lo = lambda t: int(np.flatnonzero(np.isclose(own_r, t))[0]) + 1         # first own bin with radius >= t
sim_all = resp('sim')                                                    # (case, rbin, 3)


def contributions(name, t, groups):
    """100 (flow - sim) / sim_total of cut radius >= t & MAG_AUTO < 25, per group; point, se."""
    tot = W @ sim_all[:, lo(t):, 0].sum(1)
    res = {}
    for g, (sz, mg) in groups.items():
        f, s = resp(name, sz, mg), resp('sim', sz, mg)
        d = W @ (f[:, lo(t):, 0].sum(1) - s[:, lo(t):, 0].sum(1))
        v = 100 * d / tot
        res[g] = [float(v[0]), float(v[1:].std(ddof=1))]
    return res


groups = {g: (select(re_bins, *b), None) for g, b in size_groups.items()}
groups_sm = {f'{g} {h}': (select(re_bins, *b), select(m_bins, *c)) for g, b in size_groups.items()
             for h, c in mag_groups.items()}
out['by_threshold'] = {}
print('\n== 100 (flow - sim)/sim at cut radius >= T & MAG_AUTO < 25, by true size (m_eq ~ minus the total)')
print('T    ' + ''.join(f'{g:>16s}' for g in groups) + '           total')
for t in np.round(np.arange(2.5, 4.01, 0.1), 2):
    for name in ('steps4',):
        c = contributions(name, t, groups)
        tot = contributions(name, t, {'all': (None, None)})['all']
        out['by_threshold'][f'{name}_{t:.1f}'] = {**c, 'total': tot}
        print(f'{t:.1f} ' + ''.join(f'{v[0]:+7.2f}±{v[1]:.2f}  ' for v in c.values()) + f'  {tot[0]:+6.2f}±{tot[1]:.2f}')

print('\n== same, size x magnitude groups, steps4, T = 2.8 / 3.0 / 3.2 / 3.6')
out['by_size_mag'] = {}
for t in (2.8, 3.0, 3.2, 3.6):
    c = contributions('steps4', t, groups_sm)
    out['by_size_mag'][f'{t:.1f}'] = c
    print(f'T {t}: ' + ', '.join(f'{g} {v[0]:+.2f}±{v[1]:.2f}' for g, v in c.items() if abs(v[0]) > 2 * v[1]))

# ---- where the populations sit in measured size, and their pairing-free response per galaxy per measured bin
print('\n== zero-leg count share by true size group per own radius bin (sim), and sim/flow response per galaxy')
print('bin   ' + ''.join(f'{g:>11s}' for g in size_groups) + '   | per-galaxy response sim / steps4 by group')
out['profile'] = []
for i in range(lo(2.3), lo(4.0) + 1):
    row = {'radius_lo': float(own_r[i - 1])}
    sh, rs = [], []
    n_all = sim_all[:, i, 1].sum()
    for g, (sz, _) in groups.items():
        s, f = resp('sim', sz)[:, i], resp('steps4', sz)[:, i]
        nn = (s[:, 1] + s[:, 2]).sum() / 2
        sh.append(s[:, 1].sum() / n_all)
        rs.append((s[:, 0].sum() / nn, f[:, 0].sum() / ((f[:, 1] + f[:, 2]).sum() / 2)))
        row[g] = {'share': float(sh[-1]), 'sim': float(rs[-1][0]), 'steps4': float(rs[-1][1])}
    out['profile'].append(row)
    print(f'{own_r[i-1]:.1f}  ' + ''.join(f'{v:11.3f}' for v in sh) + '   | ' +
          '  '.join(f'{a:+.2f}/{b:+.2f}' for a, b in rs))
(ROOT/'joint.json').write_text(json.dumps(out, indent=1) + '\n')
