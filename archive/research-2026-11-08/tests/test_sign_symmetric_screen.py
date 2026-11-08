"""The automated external replay must remain behind its validation gates."""
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

spec = importlib.util.spec_from_file_location('signsym_screen',
    Path(__file__).resolve().parents[1] / 'scripts/screen_sign_symmetric_candidate.py')
screen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(screen)


def setup_screen(tmp_path, monkeypatch, *, backtracking=False, **changes):
    root, old, forward = [tmp_path / name for name in ('new', 'old', 'forward')]
    new_dir = root / ('refinement' if backtracking else 'joint/joint_probability_refinement_full')
    old_dir = old / 'joint/joint_probability_refinement_full'
    for p in (new_dir, old_dir, forward):
        p.mkdir(parents=True)
    state = dict(target_transform=dict(means=np.zeros(4), scales=np.ones(4)))
    for p in (new_dir, old_dir):
        torch.save(state, p / 'selected.pt')
    reference = dict(selected_full_validation_nll=-4., selected_guard_loss=.03,
                     selected_selection_mass=dict(residual=[.1, -.1]))
    candidate = dict(reference, selected_full_validation_nll=-4.01, selected_guard_loss=.02,
        selected_sha256=screen.digest(new_dir / 'selected.pt'), best_step=1,
        eligible_for_joint_validation=True, selected_selection_mass=dict(residual=[.05, -.05]))
    candidate.update(changes)
    (new_dir / 'completed.json').write_text(json.dumps(candidate))
    (old_dir / 'completed.json').write_text(json.dumps(reference))
    (root / ('refinement_symmetry.json' if backtracking else 'final_symmetry.json')).write_text('{"status":"passed"}')
    (forward / 'protocol.json').write_text('{"hashes":{}}')
    for name in ('evaluate.py', 'job.sh', 'detect_resources.py'):
        (forward / name).write_text('# test fixture\n')
    args = ['screen', '--root', str(root), '--reference', str(old), '--forward-reference', str(forward)]
    if backtracking:
        args += ['--backtracking', '--evaluation-after', '12345']
    monkeypatch.setattr(sys, 'argv', args)
    monkeypatch.setenv('SLURM_JOB_ID', 'test')
    submissions = []
    def submit(command, **kwargs):
        submissions.append(command)
        return str(20000 + len(submissions)) + '\n'
    monkeypatch.setattr(screen.subprocess, 'check_output', submit)
    return root, old, submissions


@pytest.mark.parametrize('changes', [dict(selected_full_validation_nll=-4.001),
                                   dict(selected_guard_loss=.031)])
def test_asinh_failure_does_not_launch_external_replay(tmp_path, monkeypatch, changes):
    root, _, calls = setup_screen(tmp_path, monkeypatch, **changes)
    screen.main()
    assert not json.loads((root / 'validation_screen.json').read_text())['passed']
    assert not calls


def test_backtracking_probability_failure_does_not_launch(tmp_path, monkeypatch):
    root, _, calls = setup_screen(tmp_path, monkeypatch, backtracking=True,
        selected_selection_mass=dict(residual=[.05, -.11]))
    screen.main()
    assert not json.loads((root / 'validation_screen.json').read_text())['passed']
    assert not calls


def test_backtracking_replay_preserves_runtime_and_resource_dependency(tmp_path, monkeypatch):
    root, old, calls = setup_screen(tmp_path, monkeypatch, backtracking=True)
    screen.main()
    assert json.loads((root / 'validation_screen.json').read_text())['passed']
    assert len(calls) == 2
    assert '--array=0-7%1' in calls[0]
    assert '--dependency=afterany:12345' in calls[0]
    assert calls[0][-2:] == [str(old), 'evaluate']
    assert '--dependency=afterok:20001' in calls[1]
