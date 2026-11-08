"""Candidate publication must bind the base checkpoint to its passing audit."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    'signsym_runner', REPO / 'scripts/run_sign_symmetric_flow.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


@pytest.mark.parametrize('corruption', [None, 'checkpoint', 'audit_status', 'audit_path'])
def test_candidate_requires_matching_passed_audit(tmp_path, corruption):
    flow = tmp_path / 'base/flow/guard/selected.pt'
    flow.parent.mkdir(parents=True)
    flow.write_bytes(b'original checkpoint')
    audit = dict(status='passed', flow=str(flow), flow_sha256=runner.sha256(flow))
    if corruption == 'checkpoint':
        flow.write_bytes(b'changed after audit')
    elif corruption == 'audit_status':
        audit['status'] = 'failed'
    elif corruption == 'audit_path':
        audit['flow'] = str(tmp_path / 'another.pt')
    (tmp_path / 'base_symmetry.json').write_text(json.dumps(audit))
    experiment = dict(fixed_components={'classifier_sha256': 'unchanged'})
    if corruption:
        with pytest.raises(ValueError, match='audit mismatch'):
            runner.record_candidate(tmp_path, experiment)
        assert not (tmp_path / 'trained_candidate.json').exists()
    else:
        runner.record_candidate(tmp_path, experiment)
        candidate = json.loads((tmp_path / 'trained_candidate.json').read_text())
        assert candidate['flow'] == str(flow)
        assert candidate['flow_sha256'] == runner.sha256(flow)
        assert candidate['fixed_components'] == experiment['fixed_components']
        assert candidate['selection_stage'] == 'base_guard'
        assert not candidate['later_refinement']
        assert not candidate['production_accepted']


@pytest.mark.parametrize('stage', ['labels', 'refine', 'evaluate'])
def test_removed_stages_reject_before_accessing_study(tmp_path, stage):
    for command in (
        ['bash', str(REPO / 'jobs/job_sign_symmetric_flow.sh'), str(tmp_path), stage],
        [sys.executable, str(REPO / 'scripts/run_sign_symmetric_flow.py'),
         '--root', str(tmp_path), '--stage', stage],
    ):
        result = subprocess.run(command, capture_output=True, text=True)
        assert result.returncode == 2
    assert not list(tmp_path.iterdir())
