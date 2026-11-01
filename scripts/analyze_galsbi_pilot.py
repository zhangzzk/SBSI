"""Paired +-g pilot summary for the GalSBI re-simulation.

Reads BlendEMU's constant_response_catalogue_train.feather (galaxies detected and measured in both legs; failed fits
already dropped there) and joins each row to its scene row through input_index. Per measured-magnitude band (mean of
the two legs' MAG_AUTO, so binning does not depend on the shear sign):
  R      = <e_t(+g) - e_t(-g)> / 2g                    raw ngmix shear response along the applied shear
  c1, c2 = <(e(+g) + e(-g))/2 - e_intrinsic>           additive offset at zero shear (first order in g cancels)
Errors are the scatter of per-case means / sqrt(n_cases). No calibration, no selection model: this checks the images.
Also correlates sky pixels between legs and between cases to show the noise is paired within a case only.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

BANDS = [(14, 23), (23, 24), (24, 25), (25, 25.8), (14, 25.8)]


def per_case(df, value):
    means = df.groupby('case')[value].mean()
    return float(means.mean()), float(means.std(ddof=1)/np.sqrt(len(means))), int(len(df))


def pixel_correlations(sims, cases, g, patch=400):
    from astropy.io import fits

    def sky(case, label):
        name = sims/f'case{case}_{label}/real0/images/original/tile180.0_-0.5_bandr_rot0.fits'
        with fits.open(name, memmap=True) as h:
            data = h[0].data
            y0, x0 = data.shape[0]//2, data.shape[1]//2
            return np.asarray(data[y0:y0+patch, x0:x0+patch], float).ravel()

    plus, minus = f'{g:g}', f'{-g:g}'
    a, b, c = sky(cases[0], plus), sky(cases[0], minus), sky(cases[1], plus)
    return dict(same_case_opposite_leg=float(np.corrcoef(a, b)[0, 1]),
                different_case_same_leg=float(np.corrcoef(a, c)[0, 1]), patch_pixels=patch*patch)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sims', type=Path, required=True)
    ap.add_argument('--scenes', type=Path, required=True)
    ap.add_argument('--shear', type=float, default=0.02)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()

    df = pd.read_feather(a.sims/'constant_response_catalogue_train.feather')
    parts = []
    for case, rows in df.groupby('case'):
        scene = pd.read_feather(a.scenes/f'scene{case}.feather', columns=['e1_intrinsic', 'e2_intrinsic', 'r'])
        idx = rows['input_index'].to_numpy(int)
        rows = rows.assign(e1_int=scene['e1_intrinsic'].to_numpy()[idx], e2_int=scene['e2_intrinsic'].to_numpy()[idx],
                           r_scene=scene['r'].to_numpy()[idx])
        if not np.allclose(rows['r_scene'], rows['r_input_p']):
            raise RuntimeError(f'case {case}: input_index does not map to the scene row')
        parts.append(rows)
    df = pd.concat(parts, ignore_index=True)
    df['mag'] = 0.5*(df['measured_mag_auto_plus']+df['measured_mag_auto_minus'])
    df['resp'] = df['delta_et']/(2*a.shear)
    df['c1'] = 0.5*(df['measured_e1_plus']+df['measured_e1_minus'])-df['e1_int']
    df['c2'] = 0.5*(df['measured_e2_plus']+df['measured_e2_minus'])-df['e2_int']

    bands = []
    for lo, hi in BANDS:
        sel = df[(df['mag'] >= lo) & (df['mag'] < hi)]
        out = dict(band=[lo, hi])
        for key in ('resp', 'c1', 'c2'):
            mean, se, n = per_case(sel, key)
            out[key] = [mean, se]
            out['n'] = n
        bands.append(out)
    report = dict(n_rows=int(len(df)), n_cases=int(df['case'].nunique()),
                  rows_per_case=df.groupby('case').size().astype(int).to_dict(), bands=bands,
                  noise=pixel_correlations(a.sims, sorted(df['case'].unique()), a.shear))
    report['rows_per_case'] = {int(k): int(v) for k, v in report['rows_per_case'].items()}
    a.out.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
