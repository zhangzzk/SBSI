#!/usr/bin/env python
"""Analysis of the size-transport audit (rule.md here). CPU only; reads transport.npz and the earlier map files.

  python analyse.py            -> transport.json next to transport.npz (+ printed tables)
"""
import json
from pathlib import Path

import numpy as np

DATA = Path('/project/ls-gruen/users/zekang.zhang/sbsi_caches')
ROOT = DATA/'galsbi_size_transport_20261104'
MAPS = DATA/'galsbi_guard_strength_20261103'
BOOTSTRAP, SEED = 10_000, 20260925
FLOWS = ('nll_only', 'steps4', 'bands40')

z = np.load(ROOT/'transport.npz')
T, TM = z['t_radius'], z['t_magnitude']
nc = len(z['cases'])
W = np.vstack([np.ones(nc), np.random.default_rng(SEED).multinomial(nc, np.full(nc, 1/nc), size=BOOTSTRAP)])
cut_index = lambda r, m: int(np.argmin(abs(T - r))) * len(TM) + int(np.flatnonzero(TM == m)[0])
out = {'validation': {}, 'cuts': {}, 'transport_own': {}, 'transport_truth': {}, 'static_truth': {}}

# ---- validation: kept + entering - exiting == earlier map's bin sums over the same region
for name in ('sim',) + FLOWS:
    mp = np.load(MAPS/('map_bands.npz' if name == 'bands40' else 'map.npz'))
    tr = np.einsum('crmii->crm', mp[f'{name}_shape'])           # sum e . d, per case and bin
    re, me = mp['r_edges'], mp['m_edges']
    worst = 0.0
    for r in T:
        k = int(np.flatnonzero(np.isclose(re, r))[0])
        for m in (25.0, np.inf):
            l = len(me) if np.isinf(m) else int(np.flatnonzero(np.isclose(me, m))[0])
            ref = tr[:, k + 1:, :l + 1].sum()
            new = z[f'{name}_cuts'][:, cut_index(r, m), :3].sum()
            worst = max(worst, abs(new - ref) / abs(ref))
    out['validation'][name] = worst
    print(f'validation {name}: max relative difference to map {worst:.2e}')
if max(out['validation'].values()) > 1e-6:
    raise SystemExit('validation failed: cut totals do not reproduce the map')


def boot(f, *arrays):
    """f applied to case-weighted sums (first row = point estimate); returns point, se."""
    v = f(*[np.tensordot(W, a, axes=(1, 0)) for a in arrays])
    return v[0], v[1:].std(0, ddof=1)


# ---- 1. cut split, % of the simulation's total response at that cut
for m in TM:
    key = 'none' if np.isinf(m) else f'{m:g}'
    rows = []
    for r in T:
        i = cut_index(r, m)
        sim = z['sim_cuts'][:, i]
        row = {'radius': float(r), 'sim_parts_per_galaxy': None}
        n0 = sim[:, 5]
        p, s = boot(lambda a, n: np.stack([a[:, 0], a[:, 1], a[:, 2], a[:, 1] + a[:, 2]], -1) / n[:, None],
                    sim[:, :3], n0)
        row['sim_parts_per_galaxy'] = {'kept': p[0], 'entering': p[1], 'exiting': p[2], 'net_crossing': p[3]}
        row['sim_entries_exits_per_1000'] = (1000 * sim[:, 3].sum() / n0.sum(), 1000 * sim[:, 4].sum() / n0.sum())
        for name in FLOWS + tuple(f'{f}_mean' for f in FLOWS):
            fl = z[f'{name}_cuts'][:, i]

            def parts(a, b):
                tot = a[:, :3].sum(-1)
                d = 100 * (b[:, :3] - a[:, :3]) / tot[:, None]       # flow minus sim, % of sim total
                return np.concatenate([d, (d[:, 1] + d[:, 2])[:, None],
                                       (100 * (tot / b[:, :3].sum(-1) - 1))[:, None]], -1)
            p, s = boot(parts, sim, fl)
            row[name] = {k: [float(p[j]), float(s[j])] for j, k in
                         enumerate(('kept', 'entering', 'exiting', 'net_crossing', 'm_eq'))}
            row[name]['entries_exits_per_1000'] = (1000 * fl[:, 3].sum() / fl[:, 5].sum(),
                                                    1000 * fl[:, 4].sum() / fl[:, 5].sum())
        row['sim_parts_per_galaxy'] = {k: float(v) for k, v in row['sim_parts_per_galaxy'].items()}
        rows.append(row)
    out['cuts'][key] = rows


# ---- 2/3. transport and static slopes
def slopes(a):   # a (..., 12) sums -> transport slope, mean s0*y, shape response per galaxy, static slope
    n, x, y, xx, xy, zz, zy, ds, q, l, qq, ql = np.moveaxis(a, -1, 0)
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.stack([(xy - x * y / n) / (xx - x * x / n), zy / n, ds / n, (ql - q * l / n) / (qq - q * q / n),
                         n], -1)


names = ('transport_slope', 'mean_s0_times_y', 'shape_response', 'static_lnr_vs_e2_slope', 'count')
for name in ('sim',) + FLOWS + tuple(f'{f}_mean' for f in FLOWS):
    own = z[f'{name}_own']                              # case, rbin, mbin, stat
    own25 = own[:, :, :2].sum(2)                         # own MAG_AUTO < 25
    p, s = boot(slopes, own25)
    re = z['own_r_edges']
    out['transport_own'][name] = [{'radius_lo': float(re[j - 1]), **{k: [float(p[j, i]), float(s[j, i])]
                                   for i, k in enumerate(names)}} for j in range(1, len(re))]
    tr = z[f'{name}_truth']
    p, s = boot(slopes, tr)
    out['transport_truth'][name] = {'re_edges': z['true_re_edges'].tolist(), 'm_edges': z['true_m_edges'].tolist(),
                                    'point': p.tolist(), 'se': s.tolist(), 'columns': names}

(ROOT/'transport.json').write_text(json.dumps(out, indent=1, default=float) + '\n')

# ---- printed summary
for key in ('25', '25.8', 'none'):
    print(f'\n== cut FLUX_RADIUS > T AND MAG_AUTO < {key}: flow minus sim, % of sim total (kept / entering / '
          f'exiting / net crossing) and m_eq; sim per galaxy kept / net crossing')
    for row in out['cuts'][key]:
        if not 2.5 <= row['radius'] <= 4.0:
            continue
        sp = row['sim_parts_per_galaxy']
        txt = f"T {row['radius']:.1f} sim {sp['kept']:+.3f} {sp['net_crossing']:+.3f} |"
        for f in ('steps4', 'steps4_mean', 'nll_only'):
            v = row[f]
            txt += f" {f}: " + ' '.join(f"{v[k][0]:+5.2f}" for k in ('kept', 'entering', 'exiting',
                                                                     'net_crossing')) + \
                f" m {v['m_eq'][0]:+5.2f}±{v['m_eq'][1]:.2f} |"
        print(txt)
print('\n== transport slope dln r/|g| vs e_in.u, own MAG_AUTO < 25, by own zero-leg radius (sim / steps4 / '
      'steps4_mean / nll_only); mean s0*y; static slope')
for j, row in enumerate(out['transport_own']['sim']):
    if not 2.3 <= row['radius_lo'] <= 4.4:
        continue
    txt = f"{row['radius_lo']:.1f}"
    for f in ('sim', 'steps4', 'steps4_mean', 'nll_only'):
        r = out['transport_own'][f][j]
        txt += f" | {f} {r['transport_slope'][0]:+.3f}±{r['transport_slope'][1]:.3f} sy {r['mean_s0_times_y'][0]:+.3f}"
    print(txt)
