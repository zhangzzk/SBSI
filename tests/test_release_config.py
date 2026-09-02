import json
from pathlib import Path

import pytest

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


def test_whole_catalogue_proxy_shortlist_is_recorded_as_custom():
    args = RUNNER.parse_args(
        [
            "--inference-config",
            str(ROOT / "configs" / "inference_v1_2.json"),
            "--scene-store",
            "scene",
            "--measurement-model",
            "flow.pt",
            "--model-cache",
            "model-cache",
            "--proposal-cache",
            "proposal-cache",
            "--proposal-candidate-source",
            "whole_catalogue_gaussian_proxy",
            "--output",
            "output",
        ]
    )
    source = RUNNER._load_release_config(args.inference_config, kind="inference")
    resolved = RUNNER._resolved_pipeline_config(args, source)

    assert resolved != source
    assert resolved["proposal"]["candidate_source"] == (
        "whole_catalogue_gaussian_proxy"
    )
    assert resolved["proposal"]["prefilter_candidates"] is None


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
    """cont.344: the retained screen arm gets a release identity of its own.

    v1.2-infer is the same likelihood and the same one-step update as
    v1.1-infer.  What it changes is how the per-object sum is estimated: the
    candidate shortlist is trimmed from 16,384 to 1,024 and summed exactly,
    and the complement is drawn from the tilted whole-catalogue proposal
    rather than carried by the flat defensive tail.  Measured against the
    v1.1 arm on the same 25,000-object window, that moved median Pareto k-hat
    from 2.01 to 0.292 and relative error from 0.1264 to 0.0260.
    """

    config = json.loads((ROOT / "configs" / "inference_v1_2.json").read_text())
    assert config["release"] == "v1.2-infer"
    assert config["proposal"]["candidates"] == 1024
    assert config["proposal"]["prefilter_candidates"] == 131072
    assert config["estimator"]["draw_ladder"][-1] == 8192
    assert config["estimator"]["estimator_mode"] == "tilted_stratified"
    assert config["estimator"]["tilt_delta"] == 0.1
    assert config["estimator"]["tilt_temperature"] == 1.0
    # The split is only unbiased if every draw is retained and no finite-draw
    # correction is applied on top of it.
    assert config["estimator"]["allocation"] == "production_prefix"
    assert config["estimator"]["retain_full_ladder"] is True
    assert config["estimator"]["bias_correction"] == "none"
    # Everything outside the estimator is deliberately v1.1.
    v1_1 = json.loads((ROOT / "configs" / "inference.json").read_text())
    assert config["execution"] == v1_1["execution"]
    assert config["prior"] == v1_1["prior"]
    assert config["proposal"]["method"] == v1_1["proposal"]["method"]
    assert config["estimator"]["step"] == v1_1["estimator"]["step"]
    assert (
        config["estimator"]["finite_difference_h"]
        == v1_1["estimator"]["finite_difference_h"]
    )


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


def test_a_plain_v1_2_run_reports_its_own_release_not_custom():
    """The identity is worthless if an ordinary run of it still says custom.

    Before cont.344 the estimator mode reached the runner only as a command
    line flag, so every tilted-stratified run resolved to a document the
    release file did not contain and was recorded as `custom`.
    """

    args, source, resolved = _resolve("inference_v1_2.json")
    assert args.estimator_mode == "tilted_stratified"
    assert args.estimator_tilt_delta == 0.1
    assert args.estimator_tilt_temperature == 1.0
    assert args.proposal_candidates == 1024
    assert tuple(args.adaptive_draw_ladder) == (512, 1024, 2048, 4096, 8192)
    assert resolved == source


def test_v1_1_still_resolves_to_itself_and_declares_no_mode():
    """The new optional keys must not perturb the released default."""

    args, source, resolved = _resolve("inference.json")
    assert "estimator_mode" not in source["estimator"]
    assert args.estimator_mode == "mixture"
    assert resolved == source
    assert "estimator_mode" not in resolved["estimator"]


def test_overriding_the_mode_on_the_command_line_marks_the_run_custom():
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


def test_a_tilt_on_a_mode_that_has_none_is_refused(tmp_path):
    config = json.loads((ROOT / "configs" / "inference_v1_2.json").read_text())
    config["estimator"]["estimator_mode"] = "stratified"
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="does not take a complement tilt"):
        RUNNER._inference_defaults(
            RUNNER._load_release_config(path, kind="inference")
        )


def test_the_draw_budget_follows_the_release_ladder():
    """`--draws` is hidden and defaulted to 16,384, which only ever matched
    v1.1-infer's ladder top by coincidence.  A release whose ladder stops at
    8,192 must not be handed a deeper production budget than it declares."""

    v1_1_args, _, _ = _resolve("inference.json")
    v1_2_args, _, _ = _resolve("inference_v1_2.json")
    assert v1_1_args.draws == v1_1_args.adaptive_draw_ladder[-1] == 16384
    assert v1_2_args.draws == v1_2_args.adaptive_draw_ladder[-1] == 8192
