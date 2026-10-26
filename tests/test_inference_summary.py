from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from _script_loader import load_script_module


MODULE = load_script_module("summarize_inference.py")


def _sha256(path):
    digest = sha256()
    digest.update(Path(path).read_bytes())
    return digest.hexdigest()


def _write_result(
    root,
    *,
    n_objects=2,
    epsilon=0.1,
    bandwidth=1.0,
    implementation="implementation-a",
    proposal_method="distance_kernel",
):
    output = root / "run"
    mock = output / "mock"
    mock.mkdir(parents=True)
    pd.DataFrame(
        {
            "measured_ngmix_g1": [0.01, -0.02],
            "measured_ngmix_g2": [0.0, 0.01],
        }
    ).to_parquet(mock / "measurements.parquet", index=False)
    pd.DataFrame(
        {
            "source_case": [40, 41],
            "injected_g1": [0.02, 0.02],
            "injected_g2": [0.0, 0.0],
        }
    ).to_parquet(mock / "truth.parquet", index=False)
    mock_hashes = {name: _sha256(mock / name) for name in ("measurements.parquet", "truth.parquet")}
    payload = {
        "pipeline_release": "v1.1-infer",
        "pipeline_config_sha256": "pipeline-config",
        "pipeline_resolved_config_sha256": "resolved-config",
        "likelihood_release": "v3.2-like",
        "likelihood_config_sha256": "likelihood-config",
        "likelihood_component_sha256": {"measurement": "model"},
        "likelihood_implementation_sha256": {"likelihood": "implementation"},
        "pipeline_implementation_sha256": {
            "scripts/run_inference.py": implementation
        },
        "injected_shear": [0.02, 0.0],
        "config": {
            "proposal_flow_samples": 16,
            "proposal_statistic": "median",
            "proposal_row_chunk": 8192,
            "proposal_coordinate_seed": 8201,
            "initial": [0.0, 0.0],
            "max_iterations": 10,
            "tolerance": 1.0e-4,
            "max_step": 0.01,
            "shear_bound": 0.1,
            "max_backtracks": 8,
        },
        "selection": None,
        "r_blend": {"cache_sha256": None},
        "model_sha256": {"measurement": "model"},
        "model_cache_sha256": {
            "manifest.json": "model-manifest",
            "flow_zero.parquet": "model-flow",
            "detection_zero.parquet": "model-detection",
        },
        "scene_sha256": {"manifest.json": "scene"},
        "proposal_cache_sha256": {
            "manifest.json": "proposal-manifest",
            "coordinates.npz": "proposal-values",
        },
        "mock_sha256": mock_hashes,
        "implementation_sha256": {"scripts/run_inference.py": implementation},
        "result": {
            "converged": True,
            "estimate": [0.0201, -0.0002],
            "reason": "newton_step_below_tolerance",
            "h": 0.005,
            "n_objects": n_objects,
            "n_draws": 8192,
            "n_candidates": 16384,
            "proposal_seed": 8701,
            "epsilon": epsilon,
            "bandwidth": bandwidth,
            "elapsed_seconds": 12.5,
            "flow_evaluations": 4096,
            "importance": {
                "mean_ess": 120.0,
                "median_ess": 100.0,
                "p10_ess": 30.0,
                "mean_ess_fraction": 0.0146,
                "p90_max_weight_fraction": 0.12,
                "mean_outside_local_contribution": 0.4,
                "mean_global_draw_contribution": 0.3,
            },
            "iterations": [{"information": [[100.0, 0.0], [0.0, 80.0]]}],
        },
    }
    if proposal_method == "initial_center_posterior_adapted":
        payload["config"]["proposal_method"] = proposal_method
        payload["result"].update(
            {
                "proposal_method": proposal_method,
                "proposal_reference_shear": [0.0, 0.0],
                "initial_likelihood_reused": True,
                "proposal_candidate_flow_evaluations": 4,
                "proposal_reuse_flow_evaluations": 2,
                "proposal_flow_evaluations": 6,
                "bandwidth": None,
            }
        )
    result_path = output / "result.json"
    result_path.write_text(json.dumps(payload) + "\n")
    return result_path


def _row(block, sign, *, slope_shift=0.0):
    amplitude = 0.02
    slope = np.array([1.002 + slope_shift, 0.03])
    intercept = np.array([0.0001, -0.0002])
    return {
        "path": f"block{block}_{sign}",
        "cases": (40 + 5 * block, 41 + 5 * block),
        "n_objects": 2000,
        "injected": np.array([sign * amplitude, 0.0]),
        "estimate": intercept + sign * amplitude * slope,
        "covariance": np.diag([4.0e-8, 9.0e-8]),
        "importance": {
            "mean_ess": 100.0,
            "median_ess": 80.0,
            "p10_ess": 25.0,
            "mean_ess_fraction": 0.012,
            "p90_max_weight_fraction": 0.15,
            "mean_outside_local_contribution": 0.4,
            "mean_global_draw_contribution": 0.3,
        },
        "elapsed_seconds": 10.0,
        "flow_evaluations": 1000,
        "identity": {"model": "same", "selection": None},
        "mock_sha256": {"truth": f"block{block}_{sign}"},
    }


def test_paired_numerical_summary_uses_equal_case_blocks():
    rows = []
    for block, shift in enumerate((-0.001, 0.0, 0.001)):
        rows.extend((_row(block, -1, slope_shift=shift), _row(block, 1, slope_shift=shift)))
    result = MODULE.summarize_paired_blocks(rows, injected_component=0)
    assert result["n_case_blocks"] == 3
    assert result["multiplicative_bias"]["estimate"] == pytest.approx(0.002)
    assert result["additive_bias"]["estimate"] == pytest.approx(0.0001)
    assert result["cross_response"]["estimate"] == pytest.approx(0.03)
    assert result["multiplicative_bias"]["case_block_se"] > 0
    assert result["multiplicative_bias"]["conservative_se"] >= result["multiplicative_bias"]["curvature_se"]
    assert result["sampler_diagnostics"]["min_mean_ess"] == pytest.approx(100.0)
    assert result["sampler_diagnostics"]["total_flow_evaluations"] == 6000


def test_paired_numerical_summary_rejects_a_missing_arm():
    with pytest.raises(ValueError, match="lacks a matched"):
        MODULE.summarize_paired_blocks([_row(0, 1)], injected_component=0)


def test_load_summary_verifies_mock_and_retains_complete_sampler_identity(tmp_path):
    path = _write_result(tmp_path)
    row = MODULE.load_inference_summary(path)
    assert row["n_objects"] == 2
    assert row["cases"] == (40, 41)
    assert row["identity"]["epsilon"] == 0.1
    assert row["identity"]["bandwidth"] == 1.0
    assert row["identity"]["proposal_method"] == "distance_kernel"
    assert row["identity"]["pipeline_release"] == "v1.1-infer"
    assert row["identity"]["likelihood_release"] == "v3.2-like"
    assert row["identity"]["implementation_sha256"] == {
        "scripts/run_inference.py": "implementation-a"
    }
    assert row["identity"]["proposal_cache_sha256"]["coordinates.npz"] == ("proposal-values")


def test_load_summary_retains_posterior_adapted_proposal_identity(tmp_path):
    path = _write_result(
        tmp_path,
        proposal_method="initial_center_posterior_adapted",
    )
    row = MODULE.load_inference_summary(path)
    assert row["identity"]["proposal_method"] == ("initial_center_posterior_adapted")
    assert row["identity"]["proposal_reference_shear"] == [0.0, 0.0]
    assert row["identity"]["initial_likelihood_reused"]
    assert row["proposal_flow_evaluations"] == 6


def test_load_summary_rejects_tampered_analysis_mock(tmp_path):
    path = _write_result(tmp_path)
    truth_path = path.parent / "mock" / "truth.parquet"
    truth = pd.read_parquet(truth_path)
    truth.loc[0, "source_case"] = 99
    truth.to_parquet(truth_path, index=False)
    with pytest.raises(ValueError, match="mock hashes do not match"):
        MODULE.load_inference_summary(path)


def test_load_summary_rejects_result_object_count_mismatch(tmp_path):
    path = _write_result(tmp_path, n_objects=3)
    with pytest.raises(ValueError, match="row count does not match n_objects"):
        MODULE.load_inference_summary(path)


def test_load_summary_requires_implementation_identity(tmp_path):
    path = _write_result(tmp_path)
    payload = json.loads(path.read_text())
    payload.pop("implementation_sha256")
    payload.pop("pipeline_implementation_sha256")
    path.write_text(json.dumps(payload) + "\n")
    with pytest.raises(ValueError, match="lacks implementation hashes"):
        MODULE.load_inference_summary(path)


def test_paired_numerical_summary_rejects_overlapping_case_blocks():
    rows = [_row(0, -1), _row(0, 1), _row(1, -1), _row(1, 1)]
    rows[2]["cases"] = (41, 42)
    rows[3]["cases"] = (41, 42)
    with pytest.raises(ValueError, match="source case 41 appears in overlapping blocks"):
        MODULE.summarize_paired_blocks(rows, injected_component=0)
