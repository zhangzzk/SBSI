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
    ):
        assert setting in job
    assert '--inference-version "Infer V1"' in job
