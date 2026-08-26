import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_infer_v1_is_the_frozen_profiled_setup():
    config = json.loads((ROOT / "configs" / "infer_v1.json").read_text())

    assert config["name"] == "Infer V1"
    assert config["prior"] == "configs/default_catalogue_prior.json"
    assert config["proposal"] == {
        "flow_samples": 128,
        "location_statistic": "mean",
        "dispersion_statistic": "std",
        "candidates": 16384,
        "prefilter_candidates": 131072,
        "draws": 16384,
        "epsilon": 0.1,
        "method": "initial_center_posterior_adapted",
    }
    assert config["estimator"] == {
        "initial_strategy": "mean_observed_shape",
        "step": "one_step_full_2d",
        "finite_difference_h": 0.001,
    }
    assert config["execution"] == {
        "precision": "fp32",
        "compile_flow": True,
        "candidate_backend": "torch",
        "object_chunk": 128,
        "atom_chunk": 4096,
    }


def test_infer_v1_job_records_the_named_setup():
    job = (ROOT / "jobs" / "job_infer_v1.sh").read_text()

    for setting in (
        "${PROPOSAL_FLOW_SAMPLES:=128}",
        "${PROPOSAL_STATISTIC:=mean}",
        "${PROPOSAL_DISPERSION_STATISTIC:=std}",
        "${PROPOSAL_CANDIDATES:=16384}",
        "${PROPOSAL_PREFILTER_CANDIDATES:=131072}",
        "${DRAWS:=16384}",
        "${PROPOSAL_EPSILON:=0.1}",
        "${INITIAL_STRATEGY:=mean_observed_shape}",
        "${ADAPTIVE_ONE_STEP:=1}",
        "${RETAIN_FULL_LADDER:=1}",
        "${CANDIDATE_BACKEND:=torch}",
    ):
        assert setting in job
    assert '--inference-version "Infer V1"' in job


def test_candidate_backend_benchmark_is_matched_to_infer_v1():
    job = (ROOT / "jobs" / "job_benchmark_infer_v1_candidates.sh").read_text()

    for setting in (
        "PROPOSAL_FLOW_SAMPLES=128",
        "PROPOSAL_STATISTIC=mean PROPOSAL_DISPERSION_STATISTIC=std",
        "PROPOSAL_CANDIDATES=16384 PROPOSAL_PREFILTER_CANDIDATES=131072",
        "PROPOSAL_EPSILON=0.1 PROPOSAL_SEED=8701",
        'ADAPTIVE_DRAW_LADDER="512 1024 2048 4096 8192 16384"',
        "RETAIN_FULL_LADDER=1 COMPILE_FLOW=1",
        'CANDIDATE_BACKEND="$backend" OBJECT_CHUNK=128 ATOM_CHUNK=4096',
    ):
        assert setting in job
    assert "gpu:v100:1" in job
    assert "OBSERVATION_START=0" in job


def test_efficiency_mock_reuses_the_frozen_generation_streams():
    job = (ROOT / "jobs" / "job_prepare_infer_v1_efficiency_mock.sh").read_text()

    assert "gpu:v100:1" in job
    assert "N_DETECTED=100000 INJECTED_G1=0.02 INJECTED_G2=0.0" in job
    assert "SCENE_SEED=12001 DETECTION_SEED=12002 FLOW_SEED=12003" in job
    assert "INITIAL_STRATEGY=mean_observed_shape PREPARE_ONLY=1 COMPILE_FLOW=0" in job
