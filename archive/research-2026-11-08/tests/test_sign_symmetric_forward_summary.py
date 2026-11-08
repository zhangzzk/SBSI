"""Check response signs, own-leg normalization and the published-control gate."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location('signsym_forward',
    Path(__file__).resolve().parents[1] / 'scripts/evaluate_sign_symmetric_forward.py')
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)


def fixture_records(root):
    (root / 'cases').mkdir()
    (root / 'protocol.json').write_text('{}')
    for case in range(40, 120):
        entries = {}
        for name, response in [('measured', .5), ('old_C8', .4), ('new_C8', .5),
                               ('old_true_flags', .4), ('new_true_flags', .5)]:
            rows = np.zeros((2, 4, 3))
            # Deliberately unequal selected counts between legs and bins.
            rows[0, :, 0] = np.array([1, 3, 2, 4]) * (case + 1)
            rows[1, :, 0] = np.array([4, 2, 1, 2]) * (case + 1)
            rows[0, :, 1] = rows[0, :, 0] * (.01 + .02 * response)
            rows[1, :, 1] = rows[1, :, 0] * (.01 - .02 * response)
            entries[name] = rows.tolist()
        ev.write(root / 'cases' / f'case{case:03d}.json',
                 dict(protocol_sha256=ev.digest(root / 'protocol.json'), sums=entries))


def test_known_response_and_paired_difference(tmp_path):
    fixture_records(tmp_path)
    ev.summarize(tmp_path, dict(published_control=dict(m_percent=25., case_se_pp=0.), limitations=[]))
    result = json.loads((tmp_path / 'summary.json').read_text())['results']
    assert result['old_C8']['m_percent'] == pytest.approx(25.)
    assert result['new_C8']['m_percent'] == pytest.approx(0., abs=1e-10)
    assert result['new_C8']['data_response'][0] == pytest.approx(.5)
    assert result['new_minus_old_C8']['m_difference_pp'] == pytest.approx(-25.)
    assert result['new_minus_old_C8']['paired_case_se_pp'] < 1e-10
    assert sum(result['old_C8']['magnitude_bin_bias_contributions_pp']) == pytest.approx(25.)


def test_wrong_control_blocks_summary(tmp_path):
    fixture_records(tmp_path)
    with pytest.raises(AssertionError):
        ev.summarize(tmp_path, dict(published_control=dict(m_percent=0., case_se_pp=0.), limitations=[]))
    assert not (tmp_path / 'summary.json').exists()
