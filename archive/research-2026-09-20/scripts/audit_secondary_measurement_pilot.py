"""Check harmonized secondary measurements against current flow targets.

Compare g0 marginals and matched random-direction self-response labels.
Three reused scenes cannot establish calibration or density equivalence.
"""
import os
from pathlib import Path
import json

import numpy as np
import pandas as pd

from scripts.audit_common_optimizer_pilot import paired_marginal_summary
from scripts.train_fixed_g0_flow import sha256, write_json


def main():
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('secondary measurement audit requires Slurm')
    data = Path('/project/ls-gruen/users/zekang.zhang')
    pilot = data/'blendemu_runs/common_optimizer_secondary_pilot_20260917_v1'
    output = data/'sbsi_caches/output_conditioned_response_20260917_v1/secondary_measurement_pilot.json'
    if output.exists():
        raise FileExistsError(output)
    reports = []
    for case in (0, 40, 41):
        manifest_path = pilot/f'case{case:03d}_manifest.json'
        record = json.loads(manifest_path.read_text())
        if record['amplitudes'] != [0., .05] or record['centering'] != 'subpixel' or record['role'] != 'secondary':
            raise ValueError('unexpected secondary measurement contract')
        with np.load(pilot/f'case{case:03d}_fit_status.npz') as arrays:
            ids, fitted = arrays['ids'], arrays['shapes']
        truth = pd.read_feather(data/f'lsst_sims_fs2_25876/gals{case}_0.05.feather').set_index('index')
        expected_ids = np.sort(truth.index[(truth.index >= len(truth)//2) & (truth['r'] < 26)].to_numpy())
        np.testing.assert_array_equal(ids, expected_ids)
        directions = truth.reindex(ids)[['g1', 'g2']].to_numpy(float)/.05
        np.testing.assert_allclose(np.linalg.norm(directions, axis=1), 1., atol=1e-10)
        old = np.full_like(fitted, np.nan)
        masks, marginals = [], []
        for hi, leg in enumerate(record['case_reports']):
            for path, digest in ((leg['source_shapes'], leg['source_shapes_sha256']), (leg['output'], leg['output_sha256'])):
                if sha256(path) != digest:
                    raise ValueError('changed pilot measurement')
            original = pd.read_feather(leg['source_shapes'])
            candidate = pd.read_feather(leg['output'])
            pd.testing.assert_frame_equal(original.drop(columns=['NGMIX_G1', 'NGMIX_G2']),
                                          candidate.drop(columns=['NGMIX_G1', 'NGMIX_G2']), check_exact=True)
            cross_path = data/f'lsst_sims_fs2_25876/case{case}_{leg["h"]}/real0/catalogues/CrossMatch/tile180.0_-0.5_rot0_matched.feather'
            if sha256(cross_path) != leg['cross_sha256']:
                raise ValueError('changed crossmatch')
            cross = pd.read_feather(cross_path, columns=['id_input', 'id_detec'])
            if cross['id_input'].duplicated().any() or cross['id_detec'].duplicated().any():
                raise ValueError('nonunique crossmatch')
            aligned = cross.merge(original, left_on='id_detec', right_on='NUMBER', validate='one_to_one').set_index('id_input').reindex(ids)
            old[:, hi] = aligned[['NGMIX_G1', 'NGMIX_G2']].to_numpy(float)
            radius, mag = aligned['FLUX_RADIUS'].to_numpy(float), aligned['MAG_AUTO'].to_numpy(float)
            phot = np.isfinite(radius) & (radius > 0) & np.isfinite(mag)
            selections = (phot, phot & (radius > 3) & (mag < 25.8), phot & (radius >= 3.75) & (mag < 25.8))
            masks.append(selections)
            marginals.append([paired_marginal_summary(old[:, hi], fitted[:, hi], keep) for keep in selections])
        common = np.isfinite(old).all(axis=(1, 2)) & np.isfinite(fitted).all(axis=(1, 2))
        common &= (np.square(old).sum(-1) < 1).all(1) & (np.square(fitted).sum(-1) < 1).all(1)
        response = []
        for selection in masks[0]:
            keep = selection & common
            n = int(keep.sum())
            if n < 2:
                raise ValueError('insufficient matched self-response labels')
            labels = [2*np.einsum('ni,nj->nij', (values[keep, 1]-values[keep, 0])/.05, directions[keep])
                      for values in (old, fitted)]
            delta = labels[1]-labels[0]
            scalar = [np.trace(values, axis1=1, axis2=2)/2 for values in labels]
            response.append({'baseline_selected': int(selection.sum()), 'common_rows': n,
                             'invalid_or_unmatched_dropped': int((selection & ~common).sum()),
                             'old_tensor': labels[0].mean(0), 'new_tensor': labels[1].mean(0),
                             'tensor_difference': delta.mean(0),
                             'tensor_difference_paired_object_se': delta.std(0, ddof=1)/np.sqrt(n),
                             'old_scalar_label_variance': scalar[0].var(ddof=1),
                             'new_scalar_label_variance': scalar[1].var(ddof=1)})
        reports.append({'case': case, 'per_leg_marginals': marginals, 'response_common_fixed_g0_cuts': response,
                        'pilot_manifest_sha256': sha256(manifest_path), 'truth_parent_rows': len(ids)})
        print(f'SECONDARY_PILOT_AUDIT case={case} complete', flush=True)
    write_json(output, {'reports': reports, 'amplitudes': [0., .05],
                        'cuts': ['uncut', 'radius_gt3_mag_lt25.8', 'radius_ge3.75_mag_lt25.8'],
                        'script_sha256': sha256(__file__),
                        'limitations': ['three reused scenes; paired object errors are not independent-scene acceptance',
                                        'forward random-direction self response on common usable objects and fixed g0 cuts',
                                        'intentional centering harmonization at g0.05; not a variance-only change']})
    print(f'SECONDARY_PILOT_AUDIT_COMPLETE {output}', flush=True)


if __name__ == '__main__':
    main()
