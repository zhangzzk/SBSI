"""Split the V3.9-like FS2 ConstGold m by truth properties and reweight FS2 to the GalSBI population mix.

usage (Slurm CPU, py31, after fs2_bins.py and galsbi_parent_bins.py): python fs2_split.py
m = 100 (R_sim/R_model - 1), each leg normalized by its own selected count (doc/V39_MODEL.md#development-evidence).
Contribution of bin b to m (as compare_cuts.py): 100 (R_sim,b - R_model,b)/R_model with R_x,b = (S+_b/N+ - S-_b/N-)/2h
and N the all-bin selected count of that leg; the contributions add up to m. Errors: case bootstrap, 10,000
multinomial resamples, seed 20260925. Reweighting: every FS2 galaxy in truth bin b gets weight
f_GalSBI(b)/f_FS2(b) (parent fractions); simulation and model sums are scaled alike, then m is recomputed.
"""
import json
from pathlib import Path

import numpy as np

S = Path(__file__).resolve().parent
H = .02
AX = json.loads((S/'fs2/axes.json').read_text())
Q = AX['Q']
q = {n: i for i, n in enumerate(Q)}
NAMES = ('mag', 'size', 'sersic', 'abs_e', 'crowd100', 'mag4_size', 'mag_size')
data = {n: [] for n in NAMES}
for case in range(40, 120):
    with np.load(S/f'fs2/case{case:03d}.npz') as z:
        for n in NAMES:
            data[n].append(z[n])
data = {n: np.array(v) for n, v in data.items()}   # (case, leg, cut, bin, Q)
W = np.random.default_rng(20260925).multinomial(80, np.full(80, 1/80), size=10000)
galsbi = dict(np.load(S/'galsbi_parent_bins.npz'))


def resp(n, e):   # n, e (..., leg, bin) -> per-bin response with all-bin leg normalization
    N = n.sum(-1, keepdims=True)
    r = e/N
    return (r[..., 0, :]-r[..., 1, :])/(2*H)


def parts(x, model):   # x (..., leg, bin, Q)
    rs = resp(x[..., q['sim_n']], x[..., q['sim_e1']])
    rm = resp(x[..., q[f'{model}_n']], x[..., q[f'{model}_e1' if model != 'flow' else 'c8_e1_flow']]) \
        if model != 'flow' else resp(x[..., q['c8_n']], x[..., q['c8_e1_flow']])
    tot = rm.sum(-1)
    return 100*(rs-rm)/tot[..., None], 100*(rs.sum(-1)/tot-1), rs, rm


def table(name, cut, model, weight=None):
    x = data[name][:, :, cut]
    if weight is not None:
        x = x*weight[None, None, :, None]
    c, m, rs, rm = parts(x.sum(0), model)
    bc, bm, brs, brm = parts(np.einsum('bc,c...->b...', W, x), model)
    with np.errstate(divide='ignore', invalid='ignore'):
        within = 100*(rs/rm-1)
        bwithin = 100*(brs/brm-1)
    return dict(m=float(m), m_se=float(bm.std(ddof=1)), contribution=c.tolist(), contribution_se=bc.std(0, ddof=1).tolist(),
                abs_sum=float(np.abs(c).sum()), within_bin_m=np.nan_to_num(within).tolist(),
                within_bin_se=np.nan_to_num(np.nanstd(bwithin, 0, ddof=1)).tolist(),
                selected_fraction=(x.sum(0)[0, :, q['sim_n']]/x.sum(0)[0, :, q['sim_n']].sum()).tolist(),
                parent_fraction=(x.sum(0)[0, :, q['parent']]/x.sum(0)[0, :, q['parent']].sum()).tolist()), bm


out = {'definitions': __doc__, 'axes': AX, 'results': {}, 'reweighted': {}}
for model in ('c8', 'tf', 'flow'):
    for cut, cname in ((1, 'main'), (2, 'ge0.75'), (0, 'uncut')):
        for name in NAMES:
            out['results'][f'{model}/{cname}/{name}'] = table(name, cut, model)[0]
r = out['results']
print(f"replay: C8 main m {r['c8/main/mag']['m']:+.6f} +- {r['c8/main/mag']['m_se']:.6f} (published +0.020770 +- 0.265152); "
      f"true flags {r['tf/main/mag']['m']:+.4f}")
if abs(r['c8/main/mag']['m']-0.020769635769424255) > 1e-6:
    raise ValueError('published V3.9-like m not reproduced')

# reweight FS2 to GalSBI parent fractions on several grids
for grid in ('mag', 'size', 'mag_size', 'mag4_size', 'sersic'):
    fs2_parent = data[grid][:, 0, 0, :, q['parent']].sum(0)
    g = galsbi[grid].sum(0)
    f_fs2, f_g = fs2_parent/fs2_parent.sum(), g/g.sum()
    with np.errstate(divide='ignore', invalid='ignore'):
        w = np.where(fs2_parent > 0, f_g/f_fs2, 0.)
    lost = float(f_g[fs2_parent == 0].sum())
    for model in ('c8', 'tf'):
        res, bm = table(grid, 1, model, w)
        base = table(grid, 1, model)[1]
        out['reweighted'][f'{model}/main/{grid}'] = dict(m=res['m'], m_se=res['m_se'],
            change_vs_unweighted=float(res['m']-r[f'{model}/main/{grid}']['m']),
            change_se=float((bm-base).std(ddof=1)), galsbi_parent_fraction_without_fs2_support=lost,
            max_weight=float(w.max()), contribution=res['contribution'], abs_sum=res['abs_sum'])
(S/'fs2_split.json').write_text(json.dumps(out, indent=1)+'\n')

for model in ('c8', 'tf'):
    for name in ('mag', 'size', 'sersic', 'abs_e', 'crowd100'):
        t = r[f'{model}/main/{name}']
        print(f"\n{model} main by {name}: m {t['m']:+.3f} +- {t['m_se']:.3f}, abs sum of contributions {t['abs_sum']:.2f}")
        for b, (c, e, wm, we, sf, pf) in enumerate(zip(t['contribution'], t['contribution_se'], t['within_bin_m'],
                                                       t['within_bin_se'], t['selected_fraction'], t['parent_fraction'])):
            if sf > 1e-4:
                print(f'  bin {b:2d}: contribution {c:+6.2f} +- {e:.2f}  within {wm:+7.2f} +- {we:6.2f}  sel.frac {sf:.3f}  parent {pf:.3f}')
for k, v in out['reweighted'].items():
    print(f"reweighted {k:22s} m {v['m']:+.3f} +- {v['m_se']:.3f}  change {v['change_vs_unweighted']:+.3f} +- {v['change_se']:.3f}  "
          f"abs sum {v['abs_sum']:.2f}  GalSBI parent without FS2 support {v['galsbi_parent_fraction_without_fs2_support']:.4f}  max w {v['max_weight']:.2f}")
