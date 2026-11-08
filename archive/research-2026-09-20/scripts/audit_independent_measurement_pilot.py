"""Compare old/new per-leg marginals on independent constant-shear scenes.

Uses independent usability in each measurement, not their intersection.
This audits measurement provenance; it never computes a fitted calibration.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.audit_common_optimizer_pilot import paired_marginal_summary
from scripts.train_fixed_g0_flow import sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot-root', type=Path, required=True)
    parser.add_argument('--raw-root', type=Path, required=True)
    parser.add_argument('--cases', type=int, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('measurement catalogue audit requires Slurm')
    if args.output.exists():
        raise FileExistsError(args.output)
    if not args.cases or len(set(args.cases)) != len(args.cases):
        raise ValueError('unique nonempty pilot cases required')
    reports = []
    for case in args.cases:
        manifest_path = args.pilot_root/f'case{case:03d}_manifest.json'
        record = json.loads(manifest_path.read_text())
        if (record['catalogue_family'] != 'constant' or record['role'] != 'secondary'
                or record['centering'] != 'subpixel' or record['amplitudes'] != [.05, -.05]):
            raise ValueError('unexpected independent measurement contract')
        truth_path = args.raw_root/f'gals{case}_0.05.feather'
        if sha256(truth_path) != record['truth_sha256']:
            raise ValueError('changed generated truth')
        truth = pd.read_feather(truth_path, columns=['index', 'r'])
        expected = np.sort(truth.loc[(truth['index'] >= len(truth)//2) & (truth['r'] < 26), 'index'].to_numpy(np.int64))
        with np.load(args.pilot_root/f'case{case:03d}_fit_status.npz') as fit:
            ids, fitted = fit['ids'], fit['shapes']
        np.testing.assert_array_equal(ids, expected)
        marginals = []
        for hi, leg in enumerate(record['case_reports']):
            for path, digest in ((leg['source_shapes'], leg['source_shapes_sha256']), (leg['output'], leg['output_sha256'])):
                if sha256(path) != digest:
                    raise ValueError('changed measurement source/output')
            old, new = [pd.read_feather(leg[key]) for key in ('source_shapes', 'output')]
            pd.testing.assert_frame_equal(old.drop(columns=['NGMIX_G1', 'NGMIX_G2']),
                                          new.drop(columns=['NGMIX_G1', 'NGMIX_G2']), check_exact=True)
            path = args.raw_root/f'case{case}_{leg["h"]}/real0/catalogues/CrossMatch/tile180.0_-0.5_rot0_matched.feather'
            if sha256(path) != leg['cross_sha256']:
                raise ValueError('changed crossmatch')
            cross = pd.read_feather(path, columns=['id_input', 'id_detec'])
            if cross['id_input'].duplicated().any() or cross['id_detec'].duplicated().any():
                raise ValueError('nonunique crossmatch')
            aligned = [cross.merge(frame, left_on='id_detec', right_on='NUMBER', validate='one_to_one')
                       .set_index('id_input').reindex(ids) for frame in (old, new)]
            shapes = [frame[['NGMIX_G1', 'NGMIX_G2']].to_numpy(float) for frame in aligned]
            np.testing.assert_allclose(shapes[1], fitted[:, hi], rtol=0, atol=0, equal_nan=True)
            radius, mag = aligned[0]['FLUX_RADIUS'].to_numpy(float), aligned[0]['MAG_AUTO'].to_numpy(float)
            with np.errstate(over='ignore', invalid='ignore'):
                flux = 10**(-.4*(mag-30))
            phot = np.isfinite(radius) & (radius > 0) & np.isfinite(flux) & (flux > 0)
            cuts = {'uncut': phot, 'mag25.8_radius_gt0.60': phot & (radius > 3) & (mag < 25.8),
                    'mag25.8_radius_ge0.75': phot & (radius >= 3.75) & (mag < 25.8)}
            marginals.append({'h': leg['h'], 'cuts': {name: paired_marginal_summary(*shapes, selected)
                                                    for name, selected in cuts.items()},
                              'cross_without_shape': len(cross)-int(cross['id_detec'].isin(old['NUMBER']).sum())})
        reports.append({'case': case, 'truth_parent_rows': len(ids), 'per_leg_marginals': marginals,
                        'pilot_manifest_sha256': sha256(manifest_path)})
        print(f'INDEPENDENT_MEASUREMENT_AUDIT case={case} complete', flush=True)
    write_json(args.output, {'reports': reports, 'script_sha256': sha256(__file__),
                            'limitations': ['two pilot scenes cannot establish distribution equivalence or calibration',
                                            'paired object errors condition on each scene; no independent-scene SE',
                                            'no model fit, response offset or science-cut tuning']})
    print(f'INDEPENDENT_MEASUREMENT_AUDIT_COMPLETE {args.output}', flush=True)


if __name__ == '__main__':
    main()
