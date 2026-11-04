"""Shape part (galaxies/draws kept in both legs, true usable flags) by true-magnitude bin, GalSBI-dev run, 40 cases.

usage (CPU, py31): python split_by_mag.py
Per bin: R11 of galaxies kept in both legs (simulation) vs draws kept in both legs (model = flow + emulator shift),
the emulator share of the model, and the bin's contribution to the cut's shape part in pp of m (bin shape
difference x bin share of the simulation's both-leg count / cut model response). Errors: case bootstrap (10,000, seed 20260925).
Writes split_by_mag.json.
"""
import json
from pathlib import Path

import numpy as np

S = Path(__file__).resolve().parent
ROOT = Path('/project/ls-gruen/users/zekang.zhang/sbsi_caches/galsbi_dev_constant_grid_20261104')
CASES = [c for chunk in (0, 4, 8, 12) for c in range(1000+10*chunk, 1010+10*chunk)]
LABELS = ('<24', '24-25.5', '25.5-27.5', '>=27.5')
CUTS = ('mag25_radius_gt0.60', 'mag25_radius_gt0.62', 'mag25_radius_gt0.64', 'mag25_radius_ge0.75', 'mag25.8_radius_gt0.60')
H = .02


def mean_response(x):   # (..., leg, 3) -> R11 of the mean shape
    mean = x[..., 1]/x[..., 0]
    return (mean[..., 0]-mean[..., 1])/(2*H)


def main():
    model, sim = [], []
    for case in CASES:
        with np.load(ROOT/f'sums/case{case}.npz') as z:
            model.append(z['model'][1])
            sim.append(z['sim'])
            names = [str(c) for c in z['cuts']]
    model, sim = np.array(model), np.array(sim)       # case, leg, cut, bin, q
    n = len(CASES)
    W = np.random.default_rng(20260925).multinomial(n, np.full(n, 1/n), size=10000).astype(float)
    out = {}
    for cut in CUTS:
        k = names.index(cut)
        print(f'\n{cut} (GalSBI-dev, true flags), kept in both legs')

        def stats(agg):
            s_b = agg(sim[:, :, k, :, 3:6])                                   # leg, bin, 3
            m_b = agg(model[:, :, k, :, 5:10])                                # leg, bin, 5 (mass, flow e, blend e)
            m_tot = agg(np.concatenate([model[:, :, k, :, :1], model[:, :, k, :, 1:3]+model[:, :, k, :, 3:5]], -1)).sum(-2)
            rm_all = mean_response(m_tot)
            s_b, m_b = np.moveaxis(s_b, -3, -2), np.moveaxis(m_b, -3, -2)   # bin, leg, q
            rs = mean_response(s_b)
            rm = mean_response(np.concatenate([m_b[..., :1], m_b[..., 1:3]+m_b[..., 3:5]], -1))
            rf = mean_response(m_b[..., :3])
            count = s_b[..., 0, 0]
            total_count = count.sum(-1)
            contrib = 100*(rs-rm)*count/total_count[..., None]/rm_all[..., None]
            return np.stack([rs, rm, rf, rm-rf, 100*(rs/rm-1), contrib, count/total_count[..., None]])
        point = stats(lambda x: x.sum(0))
        boot = stats(lambda x: np.einsum('bc,c...->b...', W, x))
        se = np.std(boot, axis=1, ddof=1)
        rows = {}
        for b, label in enumerate(LABELS):
            r = {nm: [float(point[i, b]), float(se[i, b])] for i, nm in enumerate(
                ('R_sim', 'R_model', 'R_model_flow', 'R_model_emulator', 'm_in_bin', 'shape_contribution_pp', 'fraction'))}
            rows[label] = r
            print(f"  true mag {label:9s} fraction {r['fraction'][0]:.3f}  R sim {r['R_sim'][0]:.4f} model {r['R_model'][0]:.4f} "
                  f"(flow {r['R_model_flow'][0]:.4f} + emulator {r['R_model_emulator'][0]:.4f})  m in bin "
                  f"{r['m_in_bin'][0]:+.2f}±{r['m_in_bin'][1]:.2f}  contribution {r['shape_contribution_pp'][0]:+.2f}±"
                  f"{r['shape_contribution_pp'][1]:.2f} pp")
        out[cut] = rows
    (S/'split_by_mag.json').write_text(json.dumps(out, indent=1)+'\n')


if __name__ == '__main__':
    main()
