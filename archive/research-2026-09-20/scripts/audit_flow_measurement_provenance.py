"""Verify that sampled flow-domain targets are the historical raw shapes."""
import os
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.train_fixed_g0_flow import sha256, write_json


def main():
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('catalogue provenance audit requires Slurm')
    data = Path('/project/ls-gruen/users/zekang.zhang')
    domain = data/'sbsi_caches/full_domain_true_mag26_guard_v1/domain'
    output = data/'sbsi_caches/output_conditioned_response_20260917_v1/flow_measurement_provenance.json'
    if output.exists():
        raise FileExistsError(output)
    reports = []
    for case in (0, 1):
        for leg, h in (('g0', 0.), ('g05', .05)):
            raw = data/f'lsst_sims_fs2_25876/case{case}_{h}/real0/catalogues'
            shape_path = raw/'Shapes/shape_catalogue_detect_position_secondaries_tile180.0_-0.5.feather'
            cross_path = raw/'CrossMatch/tile180.0_-0.5_rot0_matched.feather'
            truth = pd.read_feather(raw/'input/gals_info_tile180.0_-0.5.feather', columns=['index_input'])
            shapes = pd.read_feather(shape_path, columns=['NUMBER', 'NGMIX_G1', 'NGMIX_G2'])
            cross = pd.read_feather(cross_path, columns=['id_input', 'id_detec'])
            if (shapes['NUMBER'].duplicated().any() or cross['id_input'].duplicated().any()
                    or cross['id_detec'].duplicated().any()):
                raise ValueError('nonunique raw measurements')
            joined = cross.merge(shapes, left_on='id_detec', right_on='NUMBER', validate='one_to_one').set_index('id_input')
            path = domain/f'flow/{leg}/case{case:03d}.npz'
            with np.load(path) as rows:
                ids, targets = rows['input_index'], rows['target'][:, :2]
                aligned = joined.reindex(ids)[['NGMIX_G1', 'NGMIX_G2']].to_numpy(float)
                if (ids < len(truth)//2).any():
                    raise ValueError('flow target is not in the secondary role')
                missing = ~np.isfinite(aligned).all(1)
                if missing.any():
                    raise ValueError(f'{int(missing.sum())} unmatched/invalid raw flow shapes')
                difference = aligned-targets
                np.testing.assert_allclose(aligned, targets, atol=4e-8, rtol=0)
                reports.append({'case': case, 'leg': leg, 'rows': len(ids), 'unmatched': int(missing.sum()),
                                'maximum_shape_difference': float(np.max(np.abs(difference))),
                                'raw_shape_path': str(shape_path), 'raw_shape_sha256': sha256(shape_path),
                                'cross_sha256': sha256(cross_path), 'domain_sha256': sha256(path)})
            print(f'FLOW_MEASUREMENT_PROVENANCE case={case} leg={leg} rows={len(ids)}', flush=True)
    write_json(output, {'comparisons': reports, 'script_sha256': sha256(__file__),
                        'limitations': ['two training-family cases; verifies target provenance, not model calibration',
                                        'centering identity established by separate historical and image-fit checks']})
    print(f'FLOW_MEASUREMENT_PROVENANCE_COMPLETE {output}', flush=True)


if __name__ == '__main__':
    main()
