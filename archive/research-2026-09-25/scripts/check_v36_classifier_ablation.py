"""Slurm preflight: replay retained probabilities and exercise summary end to end."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import torch

from sbsi.selection_model import load_selection_model
from scripts.run_v36_classifier_ablation import check_protocol, frame_inputs
from scripts import train_smooth_classifier as trainer


def main():
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Slurm required')
    root = Path('/project/ls-gruen/users/zekang.zhang/sbsi_caches/v36_classifier_occam_20260925_v1')
    p = check_protocol(root)
    torch.set_num_threads(4)
    # Exact initial function matching, despite removal of five input columns.
    full, _ = trainer.initial_model(17)
    reduced, _ = trainer.initial_model(12)
    x = torch.randn(100, 17)
    torch.testing.assert_close(full(x), reduced(x[:, [*range(8), *range(13, 17)]]), atol=0, rtol=0)
    model = load_selection_model(p['old_classifier'], device='cpu')
    with np.load(Path(p['f8_root'])/'evaluation/chunk0/rows/case040.npz') as z:
        frames, _ = frame_inputs(p, 40, z)
        for leg in ('plus', 'minus'):
            actual = model.predict_proba(frames[leg])
            np.testing.assert_allclose(actual, z[f'probability_modeled_smooth_{leg}'], atol=1e-6, rtol=2e-6)
    print('RETAINED_PROBABILITY_REPLAY_PASS case=40 both_legs', flush=True)
    with tempfile.TemporaryDirectory(dir=root, prefix='synthetic_summary_') as tmp:
        test = Path(tmp)
        (test/'protocol.json').write_text(json.dumps(p))
        from scripts.train_full_parent_classifier import sha256
        model_names = ['retained', *[f'{arm}_{seed}' for seed in p['seeds'] for arm in p['arms']]]
        done = (Path(p['old_classifier']).parent/'completed.json').read_text()
        for name in model_names[1:]:
            (test/name).mkdir()
            (test/name/'completed.json').write_text(done)
        sums = np.ones((2, 3, 4, 3))
        sums[..., 0] = 100
        sums[0, ..., 1] = 2
        sums[1, ..., 1] = -2
        for chunk in range(8):
            records = [dict(case=case, sums={k: sums.tolist() for k in ['measured', *model_names]},
                scores={k: [dict(bce=.2, brier=.05, calibration=.01)]*2 for k in model_names})
                for case in range(40+10*chunk, 50+10*chunk)]
            (test/f'evaluation_{chunk}.json').write_text(json.dumps(dict(records=records,
                classifier_hashes={k: 'synthetic' for k in model_names}, protocol_sha256=sha256(test/'protocol.json'))))
        subprocess.run([sys.executable, '-m', 'scripts.summarize_v36_classifier_ablation', '--root', str(test)],
                       check=True, stdout=subprocess.DEVNULL)
        assert json.loads((test/'comparison.json').read_text())['development_non_degradation_pass']
    (root/'preflight.json').write_text(json.dumps(dict(initial_function_match=True,
        retained_probability_replay=True, summary_identical_models_pass=True, job=os.environ['SLURM_JOB_ID']))+'\n')
    print('CLASSIFIER_PREFLIGHT_PASS', flush=True)


if __name__ == '__main__':
    main()
