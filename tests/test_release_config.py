import json
from pathlib import Path

from _script_loader import load_script_module


ROOT = Path(__file__).resolve().parents[1]
RUNNER = load_script_module("run_inference.py")


def test_release_configs_name_and_pin_the_current_pipeline():
    inference = json.loads((ROOT / "configs" / "inference.json").read_text())
    likelihood = json.loads((ROOT / "configs" / "likelihood.json").read_text())

    assert inference["release"] == "v1.1-infer"
    assert inference["proposal"]["candidates"] == 16384
    assert inference["proposal"]["prefilter_candidates"] == 131072
    assert inference["estimator"]["draw_ladder"][-1] == 16384
    assert inference["estimator"]["step"] == "one_step_full_2d"
    assert inference["execution"] == {
        "precision": "fp32",
        "compile_flow": True,
        "candidate_backend": "torch",
        "object_chunk": 128,
        "atom_chunk": 4096,
    }

    assert likelihood["release"] == "v3.2-like"
    assert likelihood["shear_transform"] == "intrinsic_ellipticity_only_v1"
    assert likelihood["measurement_model"]["sha256"] == (
        "38a76bb9bbece61f403ce2781883f38779b419101d0e690439edffb82b93c51e"
    )
    assert likelihood["emulator"]["model"]["sha256"] == (
        "01decd1335ce1c23aac1ef6ba055ae01c3950e47c4046345dcb6a1813033c21f"
    )


def test_runner_consumes_release_defaults():
    args = RUNNER.parse_args(
        [
            "--scene-store",
            "scene",
            "--measurement-model",
            "flow.pt",
            "--model-cache",
            "model-cache",
            "--proposal-cache",
            "proposal-cache",
            "--output",
            "output",
        ]
    )
    source = RUNNER._load_release_config(args.inference_config, kind="inference")
    assert RUNNER._resolved_pipeline_config(args, source) == source
    assert args.proposal_flow_samples == 128
    assert args.proposal_candidates == 16384
    assert tuple(args.adaptive_draw_ladder) == (512, 1024, 2048, 4096, 8192, 16384)
    assert args.compile_flow and args.retain_full_ladder


def test_reference_job_only_supplies_runtime_paths():
    job = (ROOT / "jobs" / "job_inference.sh").read_text()
    assert "v1.1-infer with v3.2-like" in job
    assert '"$repo/configs/inference.json"' in job
    assert '"$repo/configs/likelihood.json"' in job
    assert '"$repo/scripts/run_inference.py"' in job
    for duplicated_setting in (
        "PROPOSAL_CANDIDATES",
        "PROPOSAL_EPSILON",
        "ADAPTIVE_DRAW_LADDER",
        "CANDIDATE_BACKEND",
    ):
        assert duplicated_setting not in job
