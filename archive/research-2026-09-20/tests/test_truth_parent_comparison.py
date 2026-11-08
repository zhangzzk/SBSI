import copy
import json
import sys

import numpy as np
import pytest

from scripts.compare_truth_parent_response import main, require_comparable, require_latent_repeat


def fixture_result(response):
    stats = {}
    for source, value in (("measured", 1.), ("model", response)):
        stats[source] = {direction: {"numerator": [[[sign*.02*value, 0]], [[sign*.02*value, 0]]],
                                     "denominator": [[1.], [1.]]}
                         for direction, sign in (("plus", 1), ("minus", -1))}
    key = "truth_parent/uncut/actual_usable_flags"
    return {"cases": [1, 2], "h": .02, "draws": 64, "seed": 7301, "population": "test",
            "cohorts": ["truth_parent"], "cuts": ["uncut"], "usability": "test", "case_reports": {},
            "inputs": {"parent_manifest_sha256": "test", "classifiers": [], "rblend": []},
            "per_case_sufficient": {key: stats}, "summary": {key: {"m_percent": 100*(1/response-1)}}}


def test_paired_delta_retains_common_case_cancellation(tmp_path, monkeypatch):
    paths = [tmp_path / name for name in ("baseline.json", "candidate.json", "output.json")]
    for path, response in zip(paths[:2], (1., 1.1)):
        path.write_text(json.dumps(fixture_result(response)))
    monkeypatch.setattr(sys, "argv", ["compare", "--baseline", str(paths[0]), "--candidate", str(paths[1]), "--output", str(paths[2])])
    main()
    result = next(iter(json.loads(paths[2].read_text())["comparisons"].values()))
    np.testing.assert_allclose(result["delta_m_percentage_points"], -100/11)
    assert result["paired_delta_se_pp"] < 1e-12


def test_different_latent_stream_cannot_be_claimed_as_paired():
    baseline = fixture_result(1)
    candidate = copy.deepcopy(baseline)
    candidate["seed"] += 1
    with pytest.raises(ValueError, match="seed"):
        require_comparable(baseline, candidate)


def test_latent_repeat_rejects_model_changes_and_same_seed():
    baseline = fixture_result(1)
    with pytest.raises(ValueError, match="distinct seeds"):
        require_latent_repeat(baseline, baseline)
    repeat = copy.deepcopy(baseline)
    repeat["seed"] += 1
    require_latent_repeat(baseline, repeat)
    repeat["inputs"]["flow_sha256"] = "changed"
    with pytest.raises(ValueError, match="identical model"):
        require_latent_repeat(baseline, repeat)


def test_latent_repeat_rejects_response_change():
    baseline = fixture_result(1)
    baseline["experimental_response"] = {"metadata": {"model_sha256": "frozen"},
                                         "case_reports": {"1": {"pairs": {"sha256": "pair"}}}}
    repeat = copy.deepcopy(baseline)
    repeat["seed"] += 1
    repeat["experimental_response"]["case_reports"]["1"]["support"] = {"random_count": 3}
    require_latent_repeat(baseline, repeat)
    repeat["experimental_response"]["metadata"]["model_sha256"] = "changed"
    with pytest.raises(ValueError, match="response differs"):
        require_latent_repeat(baseline, repeat)


def test_g2_comparison_uses_r22_for_bootstrap_not_cross_response(tmp_path, monkeypatch):
    paths = [tmp_path/name for name in ('baseline.json', 'candidate.json', 'output.json')]
    for path, predicted in zip(paths, (1., 1.1)):
        result = fixture_result(predicted)
        result['shear_axis'] = 2
        result['response_vector_order'] = ['e1', 'e2']
        for stats in result['per_case_sufficient'].values():
            for source, value in [('measured', 1.), ('model', predicted)]:
                for direction, sign in [('plus', 1), ('minus', -1)]:
                    stats[source][direction]['numerator'] = [
                        [[sign*.02*(.2 if source == 'measured' else .1), sign*.02*value]],
                        [[sign*.02*(.4 if source == 'measured' else .1), sign*.02*value]]]
        path.write_text(json.dumps(result))
    monkeypatch.setattr(sys, 'argv', ['compare', '--baseline', str(paths[0]), '--candidate', str(paths[1]), '--output', str(paths[2])])
    main()
    result = json.loads(paths[2].read_text())
    assert result['shear_axis'] == 2 and result['diagonal_response_component'] == 'R22'
    values = next(iter(result['comparisons'].values()))
    np.testing.assert_allclose(values['paired_delta_ci95_pp'], [-100/11, -100/11])
    assert values['paired_delta_se_pp'] < 1e-12


def test_axis_and_component_order_mismatches_rejected():
    base = fixture_result(1.)
    candidate = copy.deepcopy(base)
    candidate['shear_axis'] = 2
    with pytest.raises(ValueError, match='shear_axis'):
        require_comparable(base, candidate)
    candidate['shear_axis'] = 1
    candidate['response_vector_order'] = ['e2', 'e1']
    with pytest.raises(ValueError, match='response-vector order'):
        require_comparable(base, candidate)


def test_single_case_operational_output_has_no_bootstrap_comparison(tmp_path, monkeypatch):
    result = fixture_result(1.)
    result['operational_single_case'] = True
    result['cases'] = [1]
    source = tmp_path/'input.json'
    source.write_text(json.dumps(result))
    monkeypatch.setattr(sys, 'argv', ['compare', '--baseline', str(source), '--candidate', str(source), '--output', str(tmp_path/'output.json')])
    with pytest.raises(ValueError, match='at least two'):
        main()
