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


def test_one_million_closure_uses_infer_v1_on_two_v100s():
    prepare = (ROOT / "jobs" / "job_infer_v1_closure_n1m_prepare.sh").read_text()
    infer = (ROOT / "jobs" / "job_infer_v1_closure_n1m.sh").read_text()
    combine = (ROOT / "jobs" / "job_infer_v1_closure_n1m_combine.sh").read_text()

    assert "N_DETECTED=1000000 INJECTED_G1=0.02 INJECTED_G2=0.0" in prepare
    assert "SCENE_SEED=12001 DETECTION_SEED=12002 FLOW_SEED=12003" in prepare
    assert "gpu:v100:2" in infer
    assert "PROPOSAL_CANDIDATES=16384 PROPOSAL_PREFILTER_CANDIDATES=131072" in infer
    assert 'ADAPTIVE_DRAW_LADDER="512 1024 2048 4096 8192 16384"' in infer
    assert "RETAIN_FULL_LADDER=1 COMPILE_FLOW=1 CANDIDATE_BACKEND=torch" in infer
    assert "run_partition 0 0 500000" in infer
    assert "run_partition 1 500000 1000000" in infer
    assert "observations_000000_499999" in combine
    assert "observations_500000_999999" in combine


def test_sampling_visualization_uses_nested_infer_v1_draws_on_v100():
    job = (ROOT / "jobs" / "job_infer_v1_sampling_visualization.sh").read_text()

    assert "gpu:v100:1" in job
    assert "infer_v1_fs2_n1m_closure_v1" in job
    assert '"$closure/combined/result.json"' in job
    assert "--pool-size 64 --pool-seed 9101 --examples 3" in job
    assert "--ladder 4096 8192 16384 --device cuda" in job


def test_exact_proposal_target_scans_second_example_on_v100():
    job = (ROOT / "jobs" / "job_infer_v1_exact_proposal_target.sh").read_text()

    assert "gpu:v100:1" in job
    assert '"$closure/combined/result.json"' in job
    assert "--exact-object-id 514716 --device cuda" in job


def test_proposal_sweep_is_paired_to_the_exact_second_example():
    job = (ROOT / "jobs" / "job_infer_v1_proposal_sweep.sh").read_text()

    assert "gpu:a40-16gb:1" in job
    assert '"$closure/combined/result.json"' in job
    assert "--exact-object-id 514716 --proposal-sweep --device cuda" in job
    assert "--proposal-k-ladder 16384 32768 65536 131072 262144 524288" in job
    assert "--proposal-prefilter 4194304" in job
    assert "--proposal-epsilon-ladder 0.1 0.2 0.3 0.4 0.6 1.0" in job
    assert "--proposal-replicates 256 --proposal-mc-seed 9917" in job


def test_large_proposal_sweep_doubles_to_two_million_candidates():
    job = (ROOT / "jobs" / "job_infer_v1_proposal_sweep_large.sh").read_text()

    assert "gpu:a40-16gb:1" in job
    assert "--proposal-k-ladder 524288 1048576 2097152" in job
    assert "--proposal-prefilter 8388608" in job
    assert "--proposal-epsilon-ladder 0.4 0.6 0.75 0.85 0.95 1.0" in job
    assert "--proposal-replicates 256 --proposal-mc-seed 9917" in job


def test_proposal_diversification_screen_keeps_infer_v1_frozen():
    job = (ROOT / "jobs" / "job_infer_v1_proposal_diversification.sh").read_text()

    assert "gpu:a40-16gb:1" in job
    assert "--score-pool 4194304 --score-prefilter 8388608" in job
    assert "--k 32768 65536 --core-fractions 0.75 0.5" in job
    assert "--temperatures 1.5 2 4 --epsilons 0.1 0.2 0.3" in job
    assert "--candidate-seeds 7301 7302 7303 7304 7305" in job
    assert "--query-counts 8 16 --log-strata 16" in job
    assert "--mc-replicates 256 --mc-draws 16384" in job


def test_proposal_histograms_reuse_the_frozen_diversification_screen():
    job = (ROOT / "jobs" / "job_infer_v1_proposal_histograms.sh").read_text()

    assert "gpu:a40-16gb:1" in job
    assert '"$closure/combined/result.json"' in job
    assert '"$root/screen/result.json"' in job
    assert '"$root/likelihood_histograms_v1"' in job
    assert "--object-id 514716 --device cuda --bins 90" in job
