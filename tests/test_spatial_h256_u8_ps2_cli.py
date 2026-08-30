from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import pytest

from _script_loader import load_script_module


ROOT = Path(__file__).resolve().parents[1]
RUNNER = load_script_module("run_inference.py")


def _minimal_arguments(*extra):
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


@pytest.fixture
def partition_root(tmp_path):
    root = tmp_path / "partition"
    root.mkdir()
    (root / RUNNER.SPATIAL_NEYMAN_PARTITION_FILE).write_bytes(b"partition")
    (root / RUNNER.SPATIAL_NEYMAN_PARTITION_MANIFEST).write_text("{}\n")
    return root


def _freeze(monkeypatch, partition_root):
    manifest = partition_root / RUNNER.SPATIAL_NEYMAN_PARTITION_MANIFEST
    cache = partition_root / RUNNER.SPATIAL_NEYMAN_PARTITION_FILE
    protocol = dict(RUNNER.SPATIAL_H256_U8_PS2_PROTOCOL)
    protocol.update(
        {
            "partition_sha256": "1" * 64,
            "partition_cache_sha256": sha256(cache.read_bytes()).hexdigest(),
            "partition_manifest_sha256": sha256(manifest.read_bytes()).hexdigest(),
        }
    )
    monkeypatch.setattr(RUNNER, "SPATIAL_H256_U8_PS2_PROTOCOL", protocol)
    monkeypatch.setattr(
        RUNNER,
        "_mock_input_hashes",
        lambda path: dict(protocol["mock_input_sha256"]),
    )
    return protocol


def _arguments(partition_root, *extra, mock_input="frozen-mock"):
    result = _minimal_arguments(
        "--diagnose-complement",
        "diagnostic",
        "--diagnose-complement-exact-oracle",
        "--diagnose-complement-spatial-h256-u8-ps2",
        "--spatial-h256-u8-ps2-partition-cache",
        str(partition_root),
        "--observation-start",
        "25000",
        "--observation-stop",
        "25512",
        "--proposal-candidates",
        "16384",
        "--proposal-prefilter-candidates",
        "131072",
        "--diagnose-complement-global-draws",
        "16384",
        "--proposal-seed",
        "8701",
        "--initial-strategy",
        "mean_observed_shape",
        "--candidate-backend",
        "torch",
        "--object-chunk",
        "32",
        "--atom-chunk",
        "4096",
        "--spatial-h256-u8-ps2-object-chunk",
        "32",
        "--spatial-h256-u8-ps2-atom-chunk",
        "262144",
    )
    if mock_input is not None:
        result.extend(("--mock-input", str(mock_input)))
    result.extend(extra)
    return result


def test_protocol_pins_integer_law_core_contract_counts_and_zero_pilot():
    protocol = RUNNER.SPATIAL_H256_U8_PS2_PROTOCOL
    assert protocol["universe_start"] == 25_000
    assert protocol["universe_stop"] == 25_512
    assert protocol["n_candidates"] == 16_384
    assert protocol["prefilter_candidates"] == 131_072
    assert protocol["n_global_draws"] == 16_384
    assert protocol["proposal_seed"] == 8_701
    assert protocol["production_seed"] == 7_008_704
    assert protocol["diagnostic_object_chunk"] == 32
    assert protocol["flow_atom_chunk"] == 4_096
    assert protocol["spatial_object_chunk"] == 32
    assert protocol["spatial_atom_chunk"] == 262_144
    assert protocol["cells_per_slab"] == 4
    assert protocol["source_microstrata"] == [16, 12, 4]
    assert protocol["source_draws_per_cell"] == [32, 24, 8]
    assert protocol["cdf_ticks_per_microstratum"] == 1 << 48
    assert protocol["balance_total_ticks"] == 1 << 53
    assert protocol["pilot"]["used"] is False
    assert protocol["pilot"]["draws"] == 0
    assert protocol["pilot"]["target_flow_evaluations"] == 0
    assert protocol["stratum_flow_evaluations"] == 8_388_608
    assert protocol["production_flow_evaluations"] == 8_388_608
    assert protocol["exact_flow_evaluations"] == 816_703_360
    assert protocol["run_flow_evaluations"] == 833_480_576


def test_cli_accepts_only_complete_reviewed_frozen_contract(partition_root, monkeypatch):
    _freeze(monkeypatch, partition_root)
    args = RUNNER.parse_args(_arguments(partition_root))
    RUNNER._validate_exact_oracle_request(args)
    RUNNER._validate_spatial_h256_u8_ps2_request(args)


@pytest.mark.parametrize(
    "extra,field",
    (
        (("--proposal-candidates", "16383"), "n_candidates"),
        (("--proposal-prefilter-candidates", "131071"), "prefilter_candidates"),
        (("--diagnose-complement-global-draws", "8192"), "n_global_draws"),
        (("--observation-start", "24999"), "universe_start"),
        (("--observation-stop", "25513"), "universe_stop"),
        (("--proposal-seed", "8702"), "proposal_seed"),
        (("--candidate-backend", "scipy"), "candidate_backend"),
        (("--object-chunk", "16"), "diagnostic_object_chunk"),
        (("--atom-chunk", "8192"), "flow_atom_chunk"),
        (("--spatial-h256-u8-ps2-object-chunk", "16"), "spatial_object_chunk"),
        (("--spatial-h256-u8-ps2-atom-chunk", "131072"), "spatial_atom_chunk"),
        (("--initial-strategy", "fixed"), "initial_strategy"),
        (("--diagnose-complement-tail-proposal",), "tail_proposal"),
        (("--diagnose-complement-pilot-stratum",), "pilot_stratum"),
        (("--diagnose-complement-spatial-neyman",), "spatial_neyman"),
        (("--spatial-neyman-partition-cache", "old"), "old_partition_cache"),
    ),
)
def test_cli_rejects_every_protocol_change(
    partition_root, monkeypatch, extra, field
):
    _freeze(monkeypatch, partition_root)
    args = RUNNER.parse_args(_arguments(partition_root, *extra))
    with pytest.raises(SystemExit, match=field):
        RUNNER._validate_spatial_h256_u8_ps2_request(args)


@pytest.mark.parametrize(
    "extra",
    (
        ("--cut-abs-ehat", "0.8"),
        ("--cut-bound", "measured_ngmix_g1:-1:1"),
        ("--selection-cache", "selection-cache"),
    ),
)
def test_cli_rejects_selection(partition_root, monkeypatch, extra):
    _freeze(monkeypatch, partition_root)
    args = RUNNER.parse_args(_arguments(partition_root, *extra))
    with pytest.raises(SystemExit, match="measured cuts.*selection caches"):
        RUNNER._validate_spatial_h256_u8_ps2_request(args)


def test_exact_oracle_accepts_h256_route_without_tail_proposal():
    args = RUNNER.parse_args(
        _minimal_arguments(
            "--diagnose-complement",
            "diagnostic",
            "--diagnose-complement-exact-oracle",
            "--diagnose-complement-spatial-h256-u8-ps2",
        )
    )
    RUNNER._validate_exact_oracle_request(args)


def test_partition_audit_rejects_h256_science_flags():
    args = RUNNER.parse_args(
        _minimal_arguments(
            "--audit-spatial-neyman-partition",
            "--diagnose-complement-spatial-h256-u8-ps2",
        )
    )
    with pytest.raises(
        SystemExit, match="--diagnose-complement-spatial-h256-u8-ps2"
    ):
        RUNNER._validate_spatial_neyman_partition_audit_request(args)


def test_runner_wires_separate_partition_protocol_dispatch_summary_and_hashes():
    source = (ROOT / "scripts" / "run_inference.py").read_text()
    assert "diagnose_spatial_h256_u8_ps2_complement(" in source
    assert "protocol=(" in source
    assert "SPATIAL_H256_U8_PS2_PROTOCOL" in source
    assert "partition=spatial_partition" in source
    assert "spatial_h256_u8_ps2_center_summary(diagnostic)" in source
    assert 'diagnostic_summary["gates"] = spatial_ps_summary["gates"]' in source
    assert '"sbsi/spatial_h256_u8_ps2_diagnostic.py"' in source
    assert '"jobs/job_complement_spatial_h256_u8_ps2_center.sh"' in source


def test_launcher_is_clean_detached_a40_exact_no_pilot_contract():
    source = (
        ROOT / "jobs" / "job_complement_spatial_h256_u8_ps2_center.sh"
    ).read_text()
    for required in (
        "#SBATCH --gres=gpu:a40:1",
        "SBSI_COMMIT",
        "symbolic-ref -q HEAD",
        "status --porcelain --untracked-files=normal",
        "refusing to overwrite",
        "OBSERVATION_START=25000",
        "OBSERVATION_STOP=25512",
        "K=16384",
        "PREFILTER=131072",
        "GLOBAL_DRAWS=16384",
        "PROPOSAL_SEED=8701",
        "--diagnose-complement-exact-oracle",
        "--diagnose-complement-spatial-h256-u8-ps2",
        "--spatial-h256-u8-ps2-partition-cache",
        "--object-chunk 32",
        "--atom-chunk 4096",
        "--spatial-h256-u8-ps2-object-chunk 32",
        "--spatial-h256-u8-ps2-atom-chunk 262144",
        'exec "$python" "$repo/scripts/run_inference.py"',
    ):
        assert required in source
    for forbidden in (
        "--diagnose-complement-tail-proposal",
        "--diagnose-complement-pilot-stratum",
        "--diagnose-complement-spatial-neyman ",
        "--spatial-h256-u8-ps2-seed",
        "--spatial-h256-u8-ps2-cells",
        "--spatial-h256-u8-ps2-microstrata",
        "--spatial-h256-u8-ps2-replicates",
        "--spatial-h256-u8-ps2-ticks",
    ):
        assert forbidden not in source
