from hashlib import sha256
import json
from pathlib import Path

import numpy as np
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


def _freeze_partition_protocol(monkeypatch, partition_root):
    manifest_path = partition_root / RUNNER.SPATIAL_NEYMAN_PARTITION_MANIFEST
    partition_path = partition_root / RUNNER.SPATIAL_NEYMAN_PARTITION_FILE
    protocol = dict(RUNNER.SPATIAL_NEYMAN_PROTOCOL)
    protocol.update(
        {
            "partition_sha256": "1" * 64,
            "partition_cache_sha256": sha256(partition_path.read_bytes()).hexdigest(),
            "partition_manifest_sha256": sha256(manifest_path.read_bytes()).hexdigest(),
        }
    )
    monkeypatch.setattr(RUNNER, "SPATIAL_NEYMAN_PROTOCOL", protocol)
    monkeypatch.setattr(
        RUNNER,
        "_mock_input_hashes",
        lambda path: dict(protocol["mock_input_sha256"]),
    )
    return protocol


def _spatial_arguments(partition_root, *extra, mock_input="frozen-mock"):
    arguments = _minimal_arguments(
        "--diagnose-complement",
        "diagnostic",
        "--diagnose-complement-exact-oracle",
        "--diagnose-complement-spatial-neyman",
        "--spatial-neyman-partition-cache",
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
        "8",
        "--atom-chunk",
        "4096",
        "--spatial-neyman-object-chunk",
        "8",
        "--spatial-neyman-atom-chunk",
        "262144",
    )
    if mock_input is not None:
        arguments.extend(("--mock-input", str(mock_input)))
    arguments.extend(extra)
    return arguments


@pytest.fixture
def partition_root(tmp_path):
    root = tmp_path / "partition"
    root.mkdir()
    (root / RUNNER.SPATIAL_NEYMAN_PARTITION_FILE).write_bytes(b"partition fixture")
    (root / RUNNER.SPATIAL_NEYMAN_PARTITION_MANIFEST).write_text("{}\n")
    return root


def test_spatial_protocol_pins_every_public_execution_choice():
    protocol = RUNNER.SPATIAL_NEYMAN_PROTOCOL
    assert protocol["universe_start"] == 25_000
    assert protocol["universe_stop"] == 25_512
    assert protocol["n_candidates"] == 16_384
    assert protocol["prefilter_candidates"] == 131_072
    assert protocol["n_global_draws"] == 16_384
    assert protocol["pilot_draws"] == 4_096
    assert protocol["proposal_seed"] == 8_701
    assert protocol["pilot_seed"] == 1_008_704
    assert protocol["production_seed"] == 7_008_704
    assert protocol["candidate_backend"] == "torch"
    assert protocol["diagnostic_object_chunk"] == 8
    assert protocol["flow_atom_chunk"] == 4_096
    assert protocol["spatial_object_chunk"] == 8
    assert protocol["spatial_atom_chunk"] == 262_144
    assert protocol["n_cells"] == 256
    assert protocol["pilot_cycles_per_cell"] == 2
    assert protocol["production_reservoir_cycles_per_cell"] == 64
    assert protocol["allocation_min_cycles"] == 2
    assert protocol["allocation_max_cycles"] == 64
    assert protocol["allocation_total_cycles"] == 2_048
    assert protocol["active_atoms"] == 12_760_990


def test_partition_audit_cli_is_outcome_blind_and_torch_only():
    args = RUNNER.parse_args(
        _minimal_arguments("--audit-spatial-neyman-partition", "--candidate-backend", "torch")
    )
    RUNNER._validate_spatial_neyman_partition_audit_request(args)

    args = RUNNER.parse_args(
        _minimal_arguments("--audit-spatial-neyman-partition", "--candidate-backend", "scipy")
    )
    with pytest.raises(SystemExit, match="requires --candidate-backend torch"):
        RUNNER._validate_spatial_neyman_partition_audit_request(args)


@pytest.mark.parametrize(
    "extra, conflict",
    (
        (("--prepare-only",), "--prepare-only"),
        (("--mock-input", "mock"), "--mock-input"),
        (("--diagnose-complement", "diagnostic"), "--diagnose-complement"),
        (
            ("--spatial-neyman-partition-cache", "partition"),
            "--spatial-neyman-partition-cache",
        ),
        (
            ("--diagnose-complement-learned-center-checkpoint", "model.pt"),
            "--diagnose-complement-learned-center-checkpoint",
        ),
    ),
)
def test_partition_audit_rejects_science_and_prepare_modes(extra, conflict):
    args = RUNNER.parse_args(_minimal_arguments("--audit-spatial-neyman-partition", *extra))
    with pytest.raises(SystemExit, match=conflict):
        RUNNER._validate_spatial_neyman_partition_audit_request(args)


def test_partition_audit_exit_precedes_mock_candidate_and_target_work():
    source = (ROOT / "scripts" / "run_inference.py").read_text()
    prework = source[
        source.index("# The spatial partition is outcome-blind") : source.index(
            "generated_mock = args.mock_input is None"
        )
    ]
    assert "build_coordinate_tree=False" in prework
    assert "if args.audit_spatial_neyman_partition:" in prework
    assert "_write_spatial_partition_audit(" in prework
    assert "return 0" in prework
    assert "generate_mock_catalogue" not in prework
    assert ".candidates(" not in prework
    assert "observed_target_tensor" not in prework


def test_spatial_science_cli_accepts_only_reviewed_frozen_artifacts(partition_root, monkeypatch):
    _freeze_partition_protocol(monkeypatch, partition_root)
    args = RUNNER.parse_args(_spatial_arguments(partition_root))

    RUNNER._validate_exact_oracle_request(args)
    RUNNER._validate_spatial_neyman_request(args)


def test_spatial_science_cli_refuses_unfrozen_partition_hashes(partition_root):
    args = RUNNER.parse_args(_spatial_arguments(partition_root))
    with pytest.raises(SystemExit, match="lacks frozen partition hashes"):
        RUNNER._validate_spatial_neyman_request(args)


@pytest.mark.parametrize(
    "extra, field",
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
        (("--spatial-neyman-object-chunk", "16"), "spatial_object_chunk"),
        (("--spatial-neyman-atom-chunk", "131072"), "spatial_atom_chunk"),
        (("--initial-strategy", "fixed"), "initial_strategy"),
        (("--diagnose-complement-tail-proposal",), "tail_proposal"),
        (("--diagnose-complement-pilot-stratum",), "pilot_stratum"),
    ),
)
def test_spatial_science_cli_rejects_protocol_changes(partition_root, monkeypatch, extra, field):
    _freeze_partition_protocol(monkeypatch, partition_root)
    args = RUNNER.parse_args(_spatial_arguments(partition_root, *extra))
    with pytest.raises(SystemExit, match=field):
        RUNNER._validate_spatial_neyman_request(args)


@pytest.mark.parametrize(
    "extra",
    (
        ("--cut-abs-ehat", "0.8"),
        ("--cut-bound", "measured_ngmix_g1:-1:1"),
        ("--selection-cache", "selection-cache"),
    ),
)
def test_spatial_science_cli_rejects_selection(partition_root, monkeypatch, extra):
    _freeze_partition_protocol(monkeypatch, partition_root)
    args = RUNNER.parse_args(_spatial_arguments(partition_root, *extra))
    with pytest.raises(SystemExit, match="measured cuts.*selection caches"):
        RUNNER._validate_spatial_neyman_request(args)


def test_spatial_exact_oracle_is_valid_without_ordinary_tail_proposal():
    args = RUNNER.parse_args(
        _minimal_arguments(
            "--diagnose-complement",
            "diagnostic",
            "--diagnose-complement-exact-oracle",
            "--diagnose-complement-spatial-neyman",
        )
    )
    RUNNER._validate_exact_oracle_request(args)

    args = RUNNER.parse_args(
        _minimal_arguments(
            "--diagnose-complement",
            "diagnostic",
            "--diagnose-complement-exact-oracle",
        )
    )
    with pytest.raises(SystemExit, match="tail-proposal or"):
        RUNNER._validate_exact_oracle_request(args)


class _SizedArray:
    def __init__(self, size):
        self.size = size

    def __len__(self):
        return self.size


class _FakePartition:
    def __init__(self, *, minimum_count=20_000, maximum_count=20_000):
        self.cell_id = _SizedArray(256 * 20_000)
        self.counts = np.full(256, 20_000, dtype=np.uint64)
        self.counts[0] = minimum_count
        self.counts[-1] = maximum_count
        self.detected_mass = np.linspace(0.001, 0.002, 256)
        self.prior_mass = np.linspace(0.002, 0.003, 256)
        self.partition_sha256 = "1" * 64
        self.recipe_sha256 = "2" * 64
        self.hash_contract = "fixture-contract"
        self.active_indices_sha256 = "3" * 64
        self.coordinates_sha256 = "4" * 64
        self.coordinate_names_sha256 = "5" * 64
        self.detected_weights_sha256 = "6" * 64
        self.prior_weights_sha256 = "7" * 64
        self.assignment_sha256 = "8" * 64
        self.counts_sha256 = "9" * 64
        self.detected_mass_sha256 = "a" * 64
        self.prior_mass_sha256 = "b" * 64
        self.metadata = {"fixture": True}

    def save(self, path):
        Path(path).write_bytes(b"partition artifact")


def _manifest_for(partition, cache_hash, provenance=None):
    return RUNNER._spatial_partition_manifest_payload(
        partition,
        partition_cache_sha256=cache_hash,
        provenance={} if provenance is None else provenance,
    )


def test_partition_audit_writes_three_identities_ranges_and_zero_science_counts(tmp_path, monkeypatch):
    partition = _FakePartition()

    class PartitionFactory:
        @staticmethod
        def build(proposal, *, n_cells):
            assert n_cells == 256
            return partition

    monkeypatch.setattr(RUNNER, "SpatialCellPartition", PartitionFactory)
    output = tmp_path / "audit"
    output.mkdir()
    audit = RUNNER._write_spatial_partition_audit(
        output,
        type("Proposal", (), {"tree": None})(),
        provenance={"coordinate_tree_built": False},
    )
    manifest_path = output / RUNNER.SPATIAL_NEYMAN_PARTITION_MANIFEST
    manifest = json.loads(manifest_path.read_text())

    assert audit["partition_sha256"] == partition.partition_sha256
    assert (
        audit["partition_cache_sha256"]
        == sha256((output / RUNNER.SPATIAL_NEYMAN_PARTITION_FILE).read_bytes()).hexdigest()
    )
    assert audit["partition_manifest_sha256"] == sha256(manifest_path.read_bytes()).hexdigest()
    for payload in (manifest, audit):
        assert payload["cell_count_minimum"] == 20_000
        assert payload["cell_count_maximum"] == 20_000
        assert payload["detected_mass_minimum"] == pytest.approx(0.001)
        assert payload["detected_mass_maximum"] == pytest.approx(0.002)
        assert payload["prior_mass_minimum"] == pytest.approx(0.002)
        assert payload["prior_mass_maximum"] == pytest.approx(0.003)
        assert payload["science_rows_read"] == 0
        assert payload["candidate_queries"] == 0
        assert payload["target_flow_evaluations"] == 0


def test_science_loader_verifies_bytes_manifest_live_proposal_and_cell_bounds(tmp_path, monkeypatch):
    root = tmp_path / "partition"
    root.mkdir()
    partition = _FakePartition()
    partition_path = root / RUNNER.SPATIAL_NEYMAN_PARTITION_FILE
    partition_path.write_bytes(b"partition artifact")
    cache_hash = sha256(partition_path.read_bytes()).hexdigest()
    manifest = _manifest_for(partition, cache_hash)
    manifest_path = root / RUNNER.SPATIAL_NEYMAN_PARTITION_MANIFEST
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    protocol = dict(RUNNER.SPATIAL_NEYMAN_PROTOCOL)
    protocol.update(
        {
            "partition_sha256": partition.partition_sha256,
            "partition_cache_sha256": cache_hash,
            "partition_manifest_sha256": sha256(manifest_path.read_bytes()).hexdigest(),
            "partition_recipe_sha256": partition.recipe_sha256,
            "active_atoms": len(partition.cell_id),
        }
    )
    monkeypatch.setattr(RUNNER, "SPATIAL_NEYMAN_PROTOCOL", protocol)
    loaded = []

    class PartitionLoader:
        @staticmethod
        def load(path, proposal=None):
            loaded.append((Path(path), proposal))
            return partition

    monkeypatch.setattr(RUNNER, "SpatialCellPartition", PartitionLoader)
    proposal = object()
    observed, elapsed = RUNNER._load_verified_spatial_partition(root, proposal)
    assert observed is partition
    assert elapsed >= 0.0
    assert loaded == [(partition_path, proposal)]

    partition.counts[0] = 16_384
    manifest = _manifest_for(partition, cache_hash)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    protocol["partition_manifest_sha256"] = sha256(manifest_path.read_bytes()).hexdigest()
    with pytest.raises(SystemExit, match="cell counts fail"):
        RUNNER._load_verified_spatial_partition(root, proposal)


def test_spatial_runner_wires_preloaded_partition_local_slice_and_summary_gate():
    source = (ROOT / "scripts" / "run_inference.py").read_text()
    assert "diagnose_spatial_neyman_complement(" in source
    assert "partition=spatial_partition" in source
    assert "partition_load_seconds=float(spatial_partition_load_seconds)" in source
    assert "observation_start=0" in source
    assert "object_id_offset=int(args.observation_start)" in source
    assert "spatial_neyman_summary(diagnostic)" in source
    assert 'diagnostic_summary["gates"] = spatial_summary["gates"]' in source


def test_spatial_jobs_are_pinned_clean_detached_a40_launchers():
    audit = (ROOT / "jobs" / "job_spatial_neyman_partition_audit.sh").read_text()
    science = (ROOT / "jobs" / "job_complement_spatial_neyman_center.sh").read_text()
    for job in (audit, science):
        assert "#SBATCH --gres=gpu:a40:1" in job
        assert "#SBATCH --output=slurm-%x-%j.out" in job
        assert "#SBATCH --error=slurm-%x-%j.err" in job
        assert "SBSI_COMMIT" in job
        assert "symbolic-ref -q HEAD" in job
        assert "status --porcelain --untracked-files=normal" in job
        assert "refusing to overwrite" in job
        assert 'exec "$python" "$repo/scripts/run_inference.py"' in job
    assert "--audit-spatial-neyman-partition" in audit
    assert "--mock-input" not in audit
    assert "--diagnose-complement-spatial-neyman" not in audit
    assert "OBSERVATION_START=25000" in science
    assert "OBSERVATION_STOP=25512" in science
    assert "K=16384" in science
    assert "PREFILTER=131072" in science
    assert "GLOBAL_DRAWS=16384" in science
    assert "PROPOSAL_SEED=8701" in science
    assert "--diagnose-complement-exact-oracle" in science
    assert "--diagnose-complement-spatial-neyman" in science
    assert "--spatial-neyman-partition-cache" in science
    assert "--diagnose-complement-tail-proposal" not in science
    assert "--spatial-neyman-object-chunk 8" in science
    assert "--spatial-neyman-atom-chunk 262144" in science


def test_spatial_sampling_stream_and_allocation_choices_are_not_cli_knobs():
    source = (ROOT / "scripts" / "run_inference.py").read_text()
    for forbidden in (
        "--spatial-neyman-pilot-seed",
        "--spatial-neyman-production-seed",
        "--spatial-neyman-pilot-draws",
        "--spatial-neyman-cells",
        "--spatial-neyman-allocation-min",
        "--spatial-neyman-allocation-max",
        "--spatial-neyman-allocation-total",
    ):
        assert forbidden not in source
