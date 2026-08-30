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


def _minimal_runner_arguments(*extra):
    return [
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
        *extra,
    ]


def test_tilted_screen_records_the_fixed_tail_recipe_without_mutating_release():
    args = RUNNER.parse_args(
        _minimal_runner_arguments("--estimator-mode", "tilted_stratified")
    )
    source = RUNNER._load_release_config(args.inference_config, kind="inference")
    original = json.loads(json.dumps(source))
    resolved = RUNNER._resolved_pipeline_config(args, source)

    assert source == original
    assert "tail_proposal" not in source["estimator"]
    assert resolved["estimator"]["estimator_mode"] == "tilted_stratified"
    assert resolved["estimator"]["tail_proposal"] == json.loads(
        json.dumps(RUNNER.asdict(RUNNER.TAIL_PROPOSAL_RECIPE))
    )
    assert (
        resolved["execution"]["tail_proposal_atom_chunk"]
        == RUNNER.TAIL_PROPOSAL_ATOM_CHUNK
    )
    assert (
        resolved["execution"]["tail_proposal_object_chunk"]
        == RUNNER.TAIL_PROPOSAL_OBJECT_CHUNK
    )
    assert resolved != source


@pytest.mark.parametrize(
    "retired",
    ("--estimator-tilt-delta", "--estimator-tilt-temperature"),
)
def test_tilted_estimator_recipe_has_no_cli_tuning_knobs(retired):
    with pytest.raises(SystemExit):
        RUNNER.parse_args(_minimal_runner_arguments(retired, "0.5"))


def test_exact_oracle_cli_is_one_boolean_without_selection_knobs():
    args = RUNNER.parse_args(
        _minimal_runner_arguments(
            "--diagnose-complement",
            "diagnostic",
            "--diagnose-complement-tail-proposal",
            "--diagnose-complement-exact-oracle",
        )
    )
    assert args.diagnose_complement_exact_oracle
    RUNNER._validate_exact_oracle_request(args)
    help_text = (ROOT / "scripts" / "run_inference.py").read_text()
    assert "--diagnose-complement-exact-oracle" in help_text
    for retired_knob in (
        "--exact-oracle-seed",
        "--exact-oracle-count",
        "--exact-oracle-atom-chunk",
        "--exact-oracle-object-chunk",
    ):
        assert retired_knob not in help_text


def test_pilot_stratum_cli_is_one_boolean_with_code_owned_streams():
    args = RUNNER.parse_args(
        _minimal_runner_arguments(
            "--diagnose-complement",
            "diagnostic",
            "--diagnose-complement-tail-proposal",
            "--diagnose-complement-pilot-stratum",
        )
    )
    assert args.diagnose_complement_pilot_stratum
    RUNNER._validate_pilot_stratum_request(args)
    assert RUNNER.PILOT_STRATUM_DIAGNOSTIC_SEED_OFFSET == 1_000_003
    assert RUNNER.COMPLEMENT_DIAGNOSTIC_SEED_OFFSET == 7_000_003
    assert 8701 + RUNNER.PILOT_STRATUM_DIAGNOSTIC_SEED_OFFSET == 1_008_704
    assert 8701 + RUNNER.COMPLEMENT_DIAGNOSTIC_SEED_OFFSET == 7_008_704

    help_text = (ROOT / "scripts" / "run_inference.py").read_text()
    assert "--diagnose-complement-pilot-stratum" in help_text
    for forbidden_knob in (
        "--pilot-stratum-draws",
        "--pilot-stratum-anchors",
        "--pilot-stratum-promotions",
        "--pilot-stratum-neighbors",
        "--pilot-stratum-seed",
    ):
        assert forbidden_knob not in help_text


@pytest.mark.parametrize(
    "extra, message",
    (
        (("--diagnose-complement-pilot-stratum",), "requires --diagnose-complement"),
        (
            (
                "--diagnose-complement",
                "diagnostic",
                "--diagnose-complement-pilot-stratum",
            ),
            "requires --diagnose-complement-tail-proposal",
        ),
        (
            (
                "--diagnose-complement",
                "diagnostic",
                "--diagnose-complement-tail-proposal",
                "--diagnose-complement-pilot-stratum",
                "--diagnose-complement-exact-oracle",
            ),
            "cannot be combined",
        ),
    ),
)
def test_pilot_stratum_cli_rejects_incomplete_or_oracle_modes(extra, message):
    args = RUNNER.parse_args(_minimal_runner_arguments(*extra))
    with pytest.raises(SystemExit, match=message):
        RUNNER._validate_pilot_stratum_request(args)


@pytest.mark.parametrize(
    "selection_arguments",
    (
        ("--cut-abs-ehat", "0.8"),
        ("--cut-bound", "measured_ngmix_g1:-1:1"),
        (
            "--cut-abs-ehat",
            "0.8",
            "--selection-cache",
            "selection-cache",
        ),
    ),
)
def test_exact_oracle_rejects_measured_selection(selection_arguments):
    args = RUNNER.parse_args(
        _minimal_runner_arguments(
            "--diagnose-complement",
            "diagnostic",
            "--diagnose-complement-tail-proposal",
            "--diagnose-complement-exact-oracle",
            *selection_arguments,
        )
    )
    with pytest.raises(SystemExit, match="frozen absolute row ids"):
        RUNNER._validate_exact_oracle_request(args)


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


def test_heldout_tail_jobs_pin_the_isolated_checkout_and_fixed_recipe():
    inference_job = (ROOT / "jobs" / "job_inference_stratified_screen.sh").read_text()
    diagnostic_job = (ROOT / "jobs" / "job_complement_mass_diagnostic.sh").read_text()
    for job in (inference_job, diagnostic_job):
        assert "/home/z/Zekang.Zhang/SBSI-inference-sampler-night" in job
        assert "refusing to overwrite" in job
        assert "must be an unsigned integer" in job
        assert "PROPOSAL_SEED=${PROPOSAL_SEED:-8701}" in job
        assert "seed${PROPOSAL_SEED}" in job
        assert '--proposal-seed "$PROPOSAL_SEED"' in job
        assert 'cd "$repo"' in job
        for reserved in ("TEMP", "TMP"):
            assert f"{reserved}=" not in job
            assert f"${reserved}" not in job
        assert "--estimator-tilt-delta" not in job
        assert "--estimator-tilt-temperature" not in job
        assert "dm3_t1_t2_prior_4_3_1_v2" in job
    assert "OBSERVATION_START=${OBSERVATION_START:-26024}" in inference_job
    assert "OBSERVATION_STOP=${OBSERVATION_STOP:-51024}" in inference_job
    assert "OBSERVATION_START=${OBSERVATION_START:-25000}" in diagnostic_job
    assert "OBSERVATION_STOP=${OBSERVATION_STOP:-25512}" in diagnostic_job
    assert "--diagnose-complement-tail-proposal" in diagnostic_job
    assert "16121540--16121542" in diagnostic_job
    assert "TILT_TEMPERATURE" in diagnostic_job


def test_exact_oracle_job_pins_fresh_universe_and_manifest():
    job = (ROOT / "jobs" / "job_complement_exact_oracle.sh").read_text()
    assert "/home/z/Zekang.Zhang/SBSI-inference-sampler-night" in job
    assert "OBSERVATION_START=25512" in job
    assert "OBSERVATION_STOP=26024" in job
    assert "GLOBAL_DRAWS=131072" in job
    assert "PROPOSAL_SEED=9701" in job
    assert "GLOBAL_DRAWS=${" not in job
    assert "PROPOSAL_SEED=${" not in job
    assert "SOURCE_MOCK" not in job
    assert "dm3_t1_t2_prior_4_3_1_v2_exact64" in job
    assert "dm4" not in job.lower()
    assert "--initial-strategy fixed" in job
    assert "--initial 0,0" in job
    assert "--candidate-backend torch" in job
    assert "--object-chunk 64" in job
    assert "asdict(TAIL_PROPOSAL_RECIPE)" in job
    assert 'EXACT_ORACLE_CONFIRMATION["tail_proposal_recipe"]' in job
    assert "--diagnose-complement-tail-proposal" in job
    assert "--diagnose-complement-exact-oracle" in job
    assert "fc44a632ed4758440a3e241ffeab9fba087f58397f114cfbc38aecc7feedcb8b" in job
    assert "refusing to overwrite" in job


def test_pilot_stratum_job_pins_consumed_development_window_and_recipe():
    job = (ROOT / "jobs" / "job_complement_pilot_stratum.sh").read_text()
    assert "/home/z/Zekang.Zhang/SBSI-inference-sampler-night" in job
    assert "OBSERVATION_START=25000" in job
    assert "OBSERVATION_STOP=25512" in job
    assert "K=16384" in job
    assert "PREFILTER=131072" in job
    assert "GLOBAL_DRAWS=16384" in job
    assert "PROPOSAL_SEED=8701" in job
    assert "OBSERVATION_START=${" not in job
    assert "OBSERVATION_STOP=${" not in job
    assert "GLOBAL_DRAWS=${" not in job
    assert "PROPOSAL_SEED=${" not in job
    assert "pilot8192_promote4096_meanobs_dm3" in job
    assert "--initial-strategy mean_observed_shape" in job
    assert "--initial-strategy fixed" not in job
    assert "--initial 0,0" not in job
    assert "--diagnose-complement-tail-proposal" in job
    assert "--diagnose-complement-pilot-stratum" in job
    assert "--diagnose-complement-exact-oracle" not in job
    assert "PILOT_STRATUM_DIAGNOSTIC_SEED_OFFSET != 1_000_003" in job
    assert "COMPLEMENT_DIAGNOSTIC_SEED_OFFSET != 7_000_003" in job
    assert "PILOT_STRATUM_RECIPE.atom_chunk != TAIL_PROPOSAL_ATOM_CHUNK" in job
    assert "PILOT_STRATUM_RECIPE.object_chunk != TAIL_PROPOSAL_OBJECT_CHUNK" in job
    assert "b4349271763311c2127d785c79fc637e115f421a31fb821e6db6e0a592b8d410" in job
    assert "refusing to overwrite" in job
