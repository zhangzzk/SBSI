"""Write per-case GalSBI scenes for BlendEMU's ``catalog.type: sbsi_scene``.

Each case is one independent draw of the SBSI continuous GalSBI prior (F24 row 0) over RA 180-181, Dec 0-1 deg:
a Poisson number of proposals, thinned by the prior's own HSC-i 14-28 support, at uniform positions on the sphere.
Nothing else is cut or resampled. Shapes follow the UFig convention of ``native_shapes`` (semi-axes
r50*sqrt(1 +- a)); the renderer gets the semi-major axis as Re, q = (1-eps)/(1+eps) and position angle psi/2 in
degrees (GalSim beta, same frame as the applied shear). The rendered magnitude 'r' is Rubin LSST-i from the same SED
coefficients. Sersic n is written natively ('n') and, for rendering ('sersic_n'), clipped to GalSim's range and rounded
to a log grid so GalSim reuses its Sersic tables (as the FS2 catalogue does, 0.5-6 in 100 steps). Features must use
the rendered value on both the simulation and the model side. Run through the scheduler.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from check_galsbi_continuous_prior import build_prior
from sbsi.galsbi_summary import native_shapes

RA_DEG, DEC_DEG = (180.0, 181.0), (0.0, 1.0)
# MultiBand_ImSim/modules/ImSimObject.py SERSIC_N_MIN, SERSIC_N_MAX (GalSim Sersic support).
RENDER_N = (0.3, 6.2)
# Same log step as FS2's 100-value 0.5-6 grid (about 2.5%); nearest node in log n.
N_GRID = np.geomspace(*RENDER_N, 122)


def render_sersic_n(n):
    k = np.rint(np.log(np.clip(n, *RENDER_N)/RENDER_N[0])/np.log(N_GRID[1]/N_GRID[0])).astype(int)
    return N_GRID[k]


def main():
    from galsbi.ucat.magnitude_calculator import MagCalculatorTable

    ap = argparse.ArgumentParser()
    ap.add_argument('--cases', type=int, nargs=2, required=True, metavar=('START', 'STOP'))
    ap.add_argument('--seed', type=int, required=True, help='scene family seed; case k uses (seed, k)')
    ap.add_argument('--lsst-table', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()

    prior, cosmo, _ = build_prior()
    table = MagCalculatorTable(['LSST_i', 'HSC_i2'], str(a.lsst_table))
    area_sr = np.deg2rad(RA_DEG[1]-RA_DEG[0]) * (np.sin(np.deg2rad(DEC_DEG[1]))-np.sin(np.deg2rad(DEC_DEG[0])))
    a.out.mkdir(parents=True, exist_ok=True)

    for case in range(*a.cases):
        rng = np.random.default_rng([a.seed, case, 0])
        n_proposals = int(rng.poisson(prior.parent_rate*area_sr))
        d, meta = prior.sample(n_proposals, [a.seed, case, 1])
        n = len(d['z'])
        ra = rng.uniform(*RA_DEG, n)
        dec = np.rad2deg(np.arcsin(rng.uniform(*np.sin(np.deg2rad(DEC_DEG)), n)))

        zero = np.zeros(n)
        mags = table(redshifts=d['z'], excess_b_v=zero, coeffs=prior.apparent_coeffs(d['z'], d['M_B'], d['c']),
                     filter_names=['LSST_i', 'HSC_i2'])
        hsc_check = float(np.max(np.abs(mags['HSC_i2']-d['m_i'])))
        if hsc_check > 1e-9:
            raise RuntimeError(f'case {case}: LSST table does not reproduce prior HSC i ({hsc_check:g} mag)')

        shapes = native_shapes(d, lambda z: cosmo.background.dist_ang_a(a=1/(1+z)))
        eps = np.hypot(shapes['e1'], shapes['e2'])
        scene = pd.DataFrame({
            'RA': ra, 'DEC': dec,
            'Re': shapes['semimajor'], 'axis_ratio': (1-eps)/(1+eps),
            'position_angle': np.rad2deg(d['psi']/2) % 180.0,
            'sersic_n': render_sersic_n(d['n']),
            'redshift': d['z'], 'r': mags['LSST_i'],
            'e1_intrinsic': shapes['e1'], 'e2_intrinsic': shapes['e2'], 'r50_circ': shapes['circularized'],
            'type': d['type'], 'M_B': d['M_B'], **{f'c{j}': d['c'][:, j] for j in range(d['c'].shape[1])},
            'R_kpc': d['R_kpc'], 'n': d['n'], 'a': d['a'], 'psi': d['psi'],
            'm_r_hsc': d['m_r'], 'm_i_hsc': d['m_i'], 'M_i': d['M_i'], 'proposal_index': d['proposal_index'],
        })
        if not np.isfinite(scene.drop(columns=['type', 'proposal_index']).to_numpy(float)).all():
            raise RuntimeError(f'case {case}: non-finite scene values')
        scene.to_feather(a.out/f'scene{case}.feather')
        report = dict(case=case, seed=[a.seed, case], area_sr=area_sr, n_proposals=n_proposals, n_galaxies=n,
                      hsc_i_table_max_abs_diff=hsc_check,
                      n_clipped_low=int(np.sum(d['n'] < RENDER_N[0])), n_clipped_high=int(np.sum(d['n'] > RENDER_N[1])),
                      lsst_i_range=[float(scene['r'].min()), float(scene['r'].max())],
                      re_max_arcsec=float(scene['Re'].max()), q_min=float(scene['axis_ratio'].min()),
                      prior=dict(meta), lsst_table=str(a.lsst_table))
        (a.out/f'scene{case}.json').write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
