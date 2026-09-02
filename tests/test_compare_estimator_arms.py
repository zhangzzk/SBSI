import json
from hashlib import sha256

import numpy as np
import pytest

from _script_loader import load_script_module


MODULE = load_script_module("compare_estimator_arms.py")


def _hash(path):
    return sha256(path.read_bytes()).hexdigest()


def _write_arm(root, *, mode, drift, ladder=(64, 256, 1024), n=400, seed=0,
               diagnostics=True, **payload_overrides):
    """Write one synthetic arm whose ladder drifts by a known amount.

    Object-level noise is shared across arms at a fixed seed, so the arms are
    paired exactly as a common-random-number screen makes them.
    """

    root.mkdir(parents=True)
    rng = np.random.default_rng(seed)
    noise = rng.normal(scale=0.02, size=(n, 2))
    ladder_score = np.empty((len(ladder), n, 2))
    ladder_information = np.empty((len(ladder), n, 2, 2))
    for index, _ in enumerate(ladder):
        # Unit information, so the one-step estimate is just the mean score.
        ladder_information[index] = np.broadcast_to(np.eye(2), (n, 2, 2))
        # A rung-dependent perturbation with exactly zero mean gives the paired
        # difference a realistic nonzero variance without moving any estimate.
        wobble = rng.normal(scale=0.004, size=(n, 2))
        wobble -= wobble.mean(axis=0)
        ladder_score[index] = noise + drift * index + (0.0 if index == 0 else wobble)
    moments = root / "one_step_moments.npz"
    arrays = {
        "score": ladder_score[-1],
        "information": ladder_information[-1],
        "draw_counts": np.full(n, ladder[-1]),
        "unique_counts": np.full(n, 7),
        "ladder_score": ladder_score,
        "ladder_information": ladder_information,
    }
    if diagnostics:
        arrays.update(
            weight_diagnostic_draws=np.asarray(ladder, dtype=np.int64),
            weight_ess=np.full((len(ladder), n), 10.0),
            weight_max_fraction=np.full((len(ladder), n), 0.2),
            weight_relative_error=np.full((len(ladder), n), 0.3),
            weight_pareto_k=np.full((len(ladder), n), 0.4),
        )
    np.savez_compressed(moments, **arrays)
    weight_rows = [
        {
            "draws": int(value),
            "ess_percentiles": [1.0, 2.0, 3.0, 4.0, 5.0],
            "ess_fraction_percentiles": [0.1, 0.2, 0.3, 0.4, 0.5],
            "max_weight_fraction_percentiles": [0.1, 0.2, 0.3, 0.4, 0.5],
            "relative_standard_error_percentiles": [0.1, 0.2, 0.3, 0.4, 0.5],
            "pareto_k_percentiles": [0.1, 0.2, 0.9, 1.1, 1.4],
            "pareto_k_undefined": 3,
            "pareto_k_threshold": 0.7,
            "pareto_k_above_threshold": 0.8,
        }
        for value in ladder
    ]
    payload = {
        "pipeline_release": "v1.1-infer" if mode == "mixture" else "custom",
        "pipeline_config": {
            "proposal": {"candidates": 16384},
            "estimator": {"draw_ladder": list(ladder), **(
                {} if mode == "mixture" else {"estimator_mode": mode}
            )},
        },
        "likelihood_release": "v3.2-like",
        "likelihood_config_sha256": "l",
        "likelihood_component_sha256": {"flow": "f"},
        "initial_center": {"center": [0.0, 0.0]},
        "injected_shear": [0.02, 0.0],
        "observation_partition": {"start": 0, "stop": n, "n_partition": n},
        "model_sha256": {"m": "1"},
        "model_cache_sha256": {"c": "1"},
        "scene_sha256": {"s": "1"},
        "proposal_cache_sha256": {"p": "1"},
        "mock_input_sha256": {"k": "1"},
        "implementation_sha256": {"i": "1"},
        "one_step_moments": {"path": moments.name, "sha256": _hash(moments)},
        "result": {
            "center": [0.0, 0.0],
            "draw_ladder": list(ladder),
            "estimator_mode": mode,
            "weight_diagnostics": weight_rows if diagnostics else None,
        },
    }
    payload.update(payload_overrides)
    (root / "result.json").write_text(json.dumps(payload, indent=2))
    return root


def test_paired_ladder_and_arm_differences_match_a_direct_computation(tmp_path):
    ladder = (64, 256, 1024)
    left = _write_arm(tmp_path / "mixture", mode="mixture", drift=1e-3, ladder=ladder)
    right = _write_arm(
        tmp_path / "stratified", mode="stratified", drift=1e-4, ladder=ladder
    )
    arms = [MODULE.load_arm("mixture", left), MODULE.load_arm("stratified", right)]
    pairing = MODULE.check_pairing(arms, ["estimator"])
    assert pairing["differing_config_keys"] == ["estimator"]
    assert pairing["n_objects"] == 400

    reports = {arm["label"]: MODULE.ladder_report(arm, 0) for arm in arms}
    # Unit information makes the one-step estimate the mean score exactly.
    for label, drift in (("mixture", 1e-3), ("stratified", 1e-4)):
        rows = reports[label]["rows"]
        assert len(rows) == len(ladder)
        expected = rows[0]["estimate"] + 2 * drift
        assert reports[label]["ladder_drift"]["shift"] == pytest.approx(
            2 * drift, rel=1e-9
        )
        assert rows[-1]["estimate"] == pytest.approx(expected, rel=1e-9)

    # The arms share their object noise, so the drift difference is resolved far
    # better than either arm's own robust error.
    drift_error = reports["mixture"]["ladder_drift"]["paired_standard_error"]
    assert drift_error > 0
    assert drift_error < 0.5 * reports["mixture"]["rows"][-1]["robust_standard_error"]

    cross = MODULE.cross_arm_report(reports, "mixture", 0)
    rows = cross["stratified"]
    assert [row["draws"] for row in rows] == list(ladder)
    assert rows[0]["shift"] == pytest.approx(0.0, abs=1e-12)
    assert rows[-1]["shift"] == pytest.approx(2 * (1e-4 - 1e-3), rel=1e-9)


def test_comparison_refuses_arms_that_are_not_paired(tmp_path):
    left = _write_arm(tmp_path / "a", mode="mixture", drift=0.0)
    right = _write_arm(
        tmp_path / "b", mode="stratified", drift=0.0, mock_input_sha256={"k": "2"}
    )
    arms = [MODULE.load_arm("a", left), MODULE.load_arm("b", right)]
    with pytest.raises(RuntimeError, match="not paired"):
        MODULE.check_pairing(arms, ["estimator"])


def test_comparison_refuses_an_undeclared_configuration_difference(tmp_path):
    left = _write_arm(tmp_path / "a", mode="mixture", drift=0.0)
    right = _write_arm(tmp_path / "b", mode="stratified", drift=0.0)
    payload = json.loads((right / "result.json").read_text())
    payload["pipeline_config"]["proposal"]["candidates"] = 65536
    (right / "result.json").write_text(json.dumps(payload))
    arms = [MODULE.load_arm("a", left), MODULE.load_arm("b", right)]
    with pytest.raises(RuntimeError, match="undeclared configuration keys"):
        MODULE.check_pairing(arms, ["estimator"])
    # Declaring it is enough; the guard exists to make the choice explicit.
    assert MODULE.check_pairing(arms, ["estimator", "proposal"])[
        "differing_config_keys"
    ] == ["estimator", "proposal"]


def test_comparison_requires_a_retained_ladder(tmp_path):
    root = _write_arm(tmp_path / "a", mode="mixture", drift=0.0)
    moments = root / "one_step_moments.npz"
    arrays = dict(np.load(moments))
    arrays.pop("ladder_score")
    np.savez_compressed(moments, **arrays)
    payload = json.loads((root / "result.json").read_text())
    payload["one_step_moments"]["sha256"] = _hash(moments)
    (root / "result.json").write_text(json.dumps(payload))
    with pytest.raises(RuntimeError, match="did not retain the full ladder"):
        MODULE.load_arm("a", root)


def test_comparison_detects_a_tampered_moment_file(tmp_path):
    root = _write_arm(tmp_path / "a", mode="mixture", drift=0.0)
    (root / "one_step_moments.npz").write_bytes(b"not the moments")
    with pytest.raises(RuntimeError, match="moment hash mismatch"):
        MODULE.load_arm("a", root)


def test_main_renders_both_tables_and_writes_json(tmp_path, capsys):
    left = _write_arm(tmp_path / "mixture", mode="mixture", drift=1e-3)
    right = _write_arm(tmp_path / "stratified", mode="stratified", drift=1e-4)
    output = tmp_path / "comparison.json"
    MODULE.main(
        [
            "--arm", f"mixture={left}",
            "--arm", f"stratified={right}",
            "--component", "g1",
            "--output", str(output),
        ]
    )
    text = capsys.readouterr().out
    assert "Draw ladder, g1" in text
    assert "Arm difference at each rung" in text
    assert "NOT on ESS/M" in text
    # The k warning must fire when most fitted objects are above threshold.
    assert "more draws cannot fix this arm" in text
    payload = json.loads(output.read_text())
    assert payload["reference"] == "mixture"
    assert set(payload["arms"]) == {"mixture", "stratified"}
    assert payload["arms"]["stratified"]["estimator_mode"] == "stratified"
    assert payload["arm_difference"]["stratified"][-1]["shift"] < 0


def test_missing_weight_diagnostics_are_reported_not_faked(tmp_path):
    root = _write_arm(tmp_path / "a", mode="mixture", drift=0.0, diagnostics=False)
    arm = MODULE.load_arm("a", root)
    assert arm["weight_diagnostics"] is None
    assert "predates the diagnostics" in MODULE.render_weights([arm])


def test_a_zero_paired_error_is_capped_rather_than_printed_as_a_huge_pull():
    capped = MODULE._format_pull({"pull": 1.3e16})
    assert capped.strip() == ">+9999"
    assert MODULE._format_pull({"pull": -1.3e16}).strip() == "<-9999"
    assert MODULE._format_pull({"pull": None}).strip() == ""
    assert MODULE._format_pull({"pull": float("inf")}).strip() == "inf"
    assert MODULE._format_pull({"pull": -3.25}).strip() == "-3.25"
