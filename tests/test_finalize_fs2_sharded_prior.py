import importlib.util
import json
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "finalize_fs2_sharded_prior.py"
)
SPEC = importlib.util.spec_from_file_location("finalize_fs2_sharded_prior", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def test_finalize_weights_shards_by_positive_atom_count(tmp_path):
    _write_json(tmp_path / "generation_manifest.json", {"cases": [20, 21, 22, 23]})
    for lower, upper, active in ((20, 21, 3), (22, 23, 1)):
        root = tmp_path / "shards" / f"cases{lower}_{upper}"
        _write_json(
            root / "scene_catalogue_report.json",
            {
                "cases": [lower, upper],
                "n_rows": 4,
                "n_positive_prior_atoms": active,
            },
        )
        _write_json(
            root / "scene_store" / "manifest.json",
            {"report": {"n_galaxies": 4, "n_directed_edges": active * 2}},
        )
        (root / "scene_store" / "galaxies.parquet").write_bytes(b"galaxies")
        (root / "scene_store" / "neighbours.npz").write_bytes(b"neighbours")
    report = MODULE.finalize(tmp_path, cases=(20, 21, 22, 23), cases_per_shard=2)
    assert report["status"] == "default_fs2_prior_catalogue"
    assert report["n_rows"] == 8
    assert report["n_positive_prior_atoms"] == 4
    assert report["n_directed_edges"] == 8
    assert report["global_shard_mass"] == [0.75, 0.25]
