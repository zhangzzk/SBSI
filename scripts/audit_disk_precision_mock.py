"""Gate a model-mock inference on ordinary and rounded-boundary observations.

Run under Slurm with SBSI_STUDY_ROOT pointing at a frozen precision-enabled
study. Tests the full sampler on every rounded boundary observation, using
its original object ID and the study's same random stream.
"""
import importlib.util
import os
from pathlib import Path
import time

import numpy as np


def main():
    root = Path(os.environ['SBSI_STUDY_ROOT'])
    spec = importlib.util.spec_from_file_location('frozen_study', root/'run_v39_sampling_study.py')
    study = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(study)
    execution = study.identity()
    jobs = study.read(root/'jobs.json')
    study.verify(__file__, jobs['precision_audit_script_sha256'])
    likelihood, proposal, mock, receipt = study.load_inference()
    likelihood.compile_precision_density()
    raw = mock.measurements[list(likelihood.target_names)].to_numpy()
    base = mock.truth[['base_e1', 'base_e2']].to_numpy()
    boundary = np.flatnonzero((np.square(raw[:, :2]).sum(1) >= 1)
                              | (np.square(base).sum(1) >= 1))
    selected = np.unique(np.r_[np.arange(4), boundary])
    checked = []
    for row in selected:
        started = time.monotonic()
        window = study.MockCatalogue(mock.measurements.iloc[[row]].reset_index(drop=True),
                                     mock.truth.iloc[[row]].reset_index(drop=True))
        atoms = window.truth.scene_row.to_numpy(dtype=int)
        density = likelihood.conditional_log_likelihood(window.measurements, *receipt['center'], atom_indices=atoms)
        assert np.isfinite(density).all()
        moments = study.run_adaptive_section5(likelihood, window, proposal, center=receipt['center'], h=.001,
            draw_ladder=(16384,), n_candidates=1024, proposal_prefilter_candidates=1024,
            epsilon=.1, proposal_seed=8701, min_ess=32., max_weight_fraction=.5,
            candidate_backend='torch', estimator_mode='tilted_stratified', tilt_delta=.1,
            retain_full_ladder=True, full_information=True, object_id_offset=int(row),
            object_chunk=1, atom_chunk=4096)
        assert np.isfinite(moments.ladder_score).all() and np.isfinite(moments.ladder_information).all()
        checked.append(dict(row=int(row), rounded_boundary=bool(row in boundary),
                            score=moments.ladder_score.tolist(), information=moments.ladder_information.tolist(),
                            conditional_log_density=density.tolist(), seconds=time.monotonic()-started))
        print(f'PRECISION_INFERENCE_AUDIT row={row} boundary={row in boundary} seconds={time.monotonic()-started:.1f}', flush=True)
    study.write_json(root/'precision_inference_audit.json', dict(status='passed', execution=execution,
        mock_manifest_sha256=study.file_hash(root/'mock/manifest.json'), boundary_count=len(boundary),
        checked=checked, dropped_or_replaced_observations=0))
    print('PRECISION_INFERENCE_AUDIT_PASS', flush=True)


if __name__ == '__main__':
    main()
