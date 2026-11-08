import json
from pathlib import Path

import pytest

from _script_loader import load_script_module


ROOT = Path(__file__).resolve().parents[1]
RUNNER = load_script_module("run_inference.py")


def test_default_release_configs_name_and_pin_the_current_pipeline():
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


def test_v35_like_pins_joint_flow_trial9_and_nine_input_u_ensemble():
    config_path = ROOT / "configs" / "likelihood_v3_5_like.json"
    likelihood = json.loads(config_path.read_text())

    assert likelihood["release"] == "v3.5-like"
    assert "has not demonstrated 0.3% calibration" in likelihood["description"]
    assert likelihood["measurement_model"] == {
        "path": None,
        "sha256": "9c5bb028437a4714b454d2bcd202243a2eda6263c36b4c70c46de3d12f725a6f",
        "target_names": [
            "measured_ngmix_g1",
            "measured_ngmix_g2",
            "measured_flux_radius",
            "measured_flux_from_mag_auto",
        ],
    }
    assert likelihood["emulator"]["model"]["sha256"] == (
        "9723589880234fa6825ae242d9b978120a6385faf5ccc9942d1993696952e46d"
    )
    assert likelihood["emulator"]["metadata"]["sha256"] == (
        "48908a9cebc7475be83eef767c95175694531d78b1dd789c512fc49c0699f2b4"
    )
    detector = likelihood["emulator"]["detection_model"]
    assert detector["backend"] == "sbsi_selection_model_ensemble"
    assert detector["aggregation"] == "arithmetic_mean_probability"
    assert [Path(member["path"]).parent.name for member in detector["members"]] == [
        "seed20260913",
        "seed20260914",
        "seed20260915",
    ]
    assert [member["sha256"] for member in detector["members"]] == [
        "bbbd27a86c18bffb3e8b98e46c455ebcd2a27844ead363913bb5289ae19e6685",
        "031498aa4140be4d2dfc7605f5211a1d2379e14c16f3fbc23b79f2f7f9e028cd",
        "dab1e4cddf3b61da4cc6dee810091209b29df4dbf5dcc86cd20d3e23eebbd702",
    ]
    assert likelihood["blend_response"] == "required_fixed_atom_cache"
    assert likelihood["measured_selection"] == {
        "mode": "output_cut_with_population_normalization",
        "bounds": [
            "measured_flux_radius:3.75:",
            "measured_flux_from_mag_auto:47.8630092322638:",
        ],
        "definition": (
            "MAG_AUTO < 25.8 and convolved SExtractor FLUX_RADIUS >= 0.75 "
            "arcsec at 0.2 arcsec/pixel; no measured-|e| cut"
        ),
    }

    args = RUNNER.parse_args(
        [
            "--likelihood-config",
            str(config_path),
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
    assert args.flow_neighbour_radius_arcsec == 7.0
    assert args.crowding_near_arcsec == 3.0
    assert args.crowding_far_arcsec == 7.0
    assert args.cut_abs_ehat is None
    assert args.cut_bound == [
        "measured_flux_radius:3.75:",
        "measured_flux_from_mag_auto:47.8630092322638:",
    ]


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


def test_v1_2_names_the_tilted_stratified_trimmed_arm():
    config = json.loads((ROOT / "configs" / "inference_v1_2.json").read_text())
    baseline = json.loads((ROOT / "configs" / "inference.json").read_text())
    assert config["release"] == "v1.2-infer"
    assert config["proposal"]["candidates"] == 1024
    assert config["proposal"]["prefilter_candidates"] == 131072
    assert config["estimator"]["draw_ladder"][-1] == 8192
    assert config["estimator"]["estimator_mode"] == "tilted_stratified"
    assert config["estimator"]["tilt_delta"] == 0.1
    assert config["estimator"]["tilt_temperature"] == 1.0
    assert config["estimator"]["allocation"] == "production_prefix"
    assert config["estimator"]["retain_full_ladder"] is True
    assert config["estimator"]["bias_correction"] == "none"
    assert config["execution"] == baseline["execution"]
    assert config["prior"] == baseline["prior"]
    assert config["estimator"]["step"] == baseline["estimator"]["step"]


def _resolve(config_name):
    args = RUNNER.parse_args(
        [
            "--inference-config",
            str(ROOT / "configs" / config_name),
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
    return args, source, RUNNER._resolved_pipeline_config(args, source)


def test_plain_v1_2_run_reports_its_release():
    args, source, resolved = _resolve("inference_v1_2.json")
    assert args.estimator_mode == "tilted_stratified"
    assert args.proposal_candidates == 1024
    assert tuple(args.adaptive_draw_ladder) == (512, 1024, 2048, 4096, 8192)
    assert resolved == source


def test_v1_1_run_reports_its_release():
    args, source, resolved = _resolve("inference.json")
    assert "estimator_mode" not in source["estimator"]
    assert args.estimator_mode == "mixture"
    assert resolved == source


def test_overriding_estimator_mode_marks_run_custom():
    args = RUNNER.parse_args(
        [
            "--inference-config",
            str(ROOT / "configs" / "inference_v1_2.json"),
            "--estimator-mode",
            "mixture",
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
    assert RUNNER._resolved_pipeline_config(args, source) != source


def test_tilt_on_mode_without_tilt_is_refused(tmp_path):
    config = json.loads((ROOT / "configs" / "inference_v1_2.json").read_text())
    config["estimator"]["estimator_mode"] = "stratified"
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="does not take a complement tilt"):
        RUNNER._inference_defaults(RUNNER._load_release_config(path, kind="inference"))


def test_draw_budget_follows_release_ladder():
    v1_1_args, _, _ = _resolve("inference.json")
    v1_2_args, _, _ = _resolve("inference_v1_2.json")
    assert v1_1_args.draws == v1_1_args.adaptive_draw_ladder[-1] == 16384
    assert v1_2_args.draws == v1_2_args.adaptive_draw_ladder[-1] == 8192
