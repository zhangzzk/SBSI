"""Shape/selection split of the end-to-end m for both runs of this study (true usable flags, 40 cases).

usage (CPU, py31): python split.py
Same definition as scripts/evaluate_galsbi_constant.py summary/split: shape part = response of galaxies (simulation)
or draws (model) kept in both legs; selection part = the rest; both in pp of m, adding up to m. Also splits the model
side into flow and emulator (blend) shape shifts. Errors: case bootstrap (10,000, seed 20260925). Writes split.json.
"""
import json
from pathlib import Path

import numpy as np

S = Path(__file__).resolve().parent
D = Path('/project/ls-gruen/users/zekang.zhang/sbsi_caches')
RUNS = {'A_band_guard': D/'galsbi_bands40_constant_20261104', 'B_galsbi_dev': D/'galsbi_dev_constant_grid_20261104'}
CASES = [c for chunk in (0, 4, 8, 12) for c in range(1000+10*chunk, 1010+10*chunk)]
H = .02


def response(sums):   # (..., leg, 3): mass, e1, e2 -> R11
    mean = sums[..., 1:2]/sums[..., :1]
    return ((mean[..., 0, :]-mean[..., 1, :])/(2*H))[..., 0]


def main():
    n = len(CASES)
    W = np.random.default_rng(20260925).multinomial(n, np.full(n, 1/n), size=10000).astype(float)
    out = {}
    for run, root in RUNS.items():
        model, sim = [], []
        for case in CASES:
            with np.load(root/f'sums/case{case}.npz') as z:
                model.append(z['model'][1])          # true-flag weights: leg, cut, bin, 10
                sim.append(z['sim'])
                cuts = [str(c) for c in z['cuts']]
        model, sim = np.array(model).sum(-2), np.array(sim).sum(-2)   # case, leg, cut, q
        print(f'\n{run} (true flags): m = shape part + selection part [pp]; model shape part split flow / emulator')
        for k, cut in enumerate(cuts):
            def parts(agg):
                s_all, s_both = agg(sim[:, :, k, :3]), agg(sim[:, :, k, 3:6])
                m_all = agg(np.concatenate([model[:, :, k, :1], model[:, :, k, 1:3]+model[:, :, k, 3:5]], -1))
                m_both = agg(np.concatenate([model[:, :, k, 5:6], model[:, :, k, 6:8]+model[:, :, k, 8:10]], -1))
                m_both_flow = agg(np.concatenate([model[:, :, k, 5:6], model[:, :, k, 6:8]], -1))
                rs, rm, bs, bm, bf = (response(x) for x in (s_all, m_all, s_both, m_both, m_both_flow))
                return np.stack([100*(rs/rm-1), 100*(bs-bm)/rm, 100*((rs-bs)-(rm-bm))/rm,
                                 bs, bm, bf, bm-bf, rs-bs, rm-bm])
            point = parts(lambda x: x.sum(0))
            boot = parts(lambda x: np.einsum('bc,c...->b...', W, x))
            se = boot.std(-1, ddof=1)
            names = ('m', 'shape_part_pp', 'selection_part_pp', 'R_both_sim', 'R_both_model', 'R_both_model_flow',
                     'R_both_model_emulator', 'R_selection_sim', 'R_selection_model')
            out[f'{run}/{cut}'] = {nm: [float(p), float(e)] for nm, p, e in zip(names, point, se)}
            print(f'  {cut:24s} m {point[0]:+.2f}±{se[0]:.2f} = shape {point[1]:+.2f}±{se[1]:.2f} + selection '
                  f'{point[2]:+.2f}±{se[2]:.2f} | R_sel sim {point[7]:+.4f} model {point[8]:+.4f} | '
                  f'R_both sim {point[3]:.4f} model {point[4]:.4f} (flow {point[5]:.4f} + emulator {point[6]:.4f})')
    (S/'split.json').write_text(json.dumps(out, indent=1)+'\n')


if __name__ == '__main__':
    main()
