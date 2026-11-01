"""Check written GalSBI scenes and, when present, the BlendEMU render inputs made from them.

Per scene: renderer-convention identities (Re*sqrt(q) = circularized radius; q and position angle reproduce the
intrinsic e1, e2), Poisson count against the prior rate, and uniform positions on the sphere. Per shear leg: the
BlendEMU ``gals{case}_{g}.feather`` must hold the scene unchanged, row for row. Light enough for one srun task.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kstest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenes', type=Path, required=True)
    ap.add_argument('--sims', type=Path, help='BlendEMU output_path holding gals{case}_{g}.feather')
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()

    rows, expected, observed = [], 0.0, 0
    for meta_path in sorted(a.scenes.glob('scene*.json')):
        meta = json.loads(meta_path.read_text())
        case = meta['case']
        s = pd.read_feather(a.scenes/f'scene{case}.feather')
        q, beta = s['axis_ratio'].to_numpy(), np.deg2rad(s['position_angle'].to_numpy())
        eps = (1-q)/(1+q)
        row = dict(case=case, n=len(s),
                   circ_identity=float(np.max(np.abs(s['Re']*np.sqrt(q)/s['r50_circ']-1))),
                   e1_identity=float(np.max(np.abs(eps*np.cos(2*beta)-s['e1_intrinsic']))),
                   e2_identity=float(np.max(np.abs(eps*np.sin(2*beta)-s['e2_intrinsic']))),
                   ks_ra_p=float(kstest(s['RA']-180.0, 'uniform').pvalue),
                   ks_sindec_p=float(kstest(np.sin(np.deg2rad(s['DEC']))/np.sin(np.deg2rad(1.0)), 'uniform').pvalue),
                   mean_e1=float(s['e1_intrinsic'].mean()), mean_e2=float(s['e2_intrinsic'].mean()),
                   n_render_values=int(s['sersic_n'].nunique()),
                   max_render_n_rel_change_in_range=float(np.max(np.abs(
                       (s['sersic_n']/s['n']-1)[(s['n'] > 0.3) & (s['n'] < 6.2)]))),
                   frac_lsst_i_lt_25=float(np.mean(s['r'] < 25)), lsst_i_max=float(s['r'].max()))
        p = meta['prior']
        expected += p['surface_density_sr']*meta['area_sr']
        observed += len(s)
        if a.sims is not None:
            for path in sorted(a.sims.glob(f'gals{case}_*.feather')):
                g = pd.read_feather(path)
                same = len(g) == len(s) and all(np.array_equal(g[c].to_numpy(), s[c].to_numpy()) for c in s.columns)
                row[f'render_input_equals_scene[{path.name}]'] = bool(same)
                row[f'shear[{path.name}]'] = [float(g['g1'].mean()), float(g['g2'].mean())]
        rows.append(row)

    report = dict(cases=rows, n_total=observed, n_expected_from_prior=expected,
                  count_z=(observed-expected)/np.sqrt(expected),
                  mean_e1_all=float(np.average([r['mean_e1'] for r in rows], weights=[r['n'] for r in rows])),
                  mean_e2_all=float(np.average([r['mean_e2'] for r in rows], weights=[r['n'] for r in rows])))
    a.out.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'cases'}, indent=2))
    for r in rows:
        print(json.dumps(r))


if __name__ == '__main__':
    main()
