import importlib.util
import json
from hashlib import sha256
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "combine_numerical_recenter_partitions.py"
SPEC = importlib.util.spec_from_file_location("combine_recenter_partitions", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _hash(path):
    return sha256(path.read_bytes()).hexdigest()


def _write_part(root, start, score, information):
    root.mkdir()
    moments = root / "one_step_moments.npz"
    ladder_score = score[None, ...]
    ladder_information = information[None, ...]
    np.savez_compressed(
        moments,
        score=score,
        information=information,
        draw_counts=np.full(len(score), 65536),
        unique_counts=np.full(len(score), 1234),
        ladder_score=ladder_score,
        ladder_information=ladder_information,
    )
    identity = {
        "initial_center": {"center": [0.01, -0.02]},
        "injected_shear": [0.02, 0.0],
        "model_sha256": {"measurement": "m"},
        "model_cache_sha256": {"manifest.json": "c"},
        "scene_sha256": {"manifest.json": "s"},
        "proposal_cache_sha256": {"manifest.json": "p"},
        "mock_input_sha256": {"measurements.parquet": "o"},
        "implementation_sha256": {"runner": "i"},
    }
    payload = {
        **identity,
        "observation_partition": {
            "start": start,
            "stop": start + len(score),
            "n_partition": len(score),
            "n_total": 4,
        },
        "one_step_moments": {
            "path": moments.name,
            "sha256": _hash(moments),
        },
        "config": {"adaptive_draw_ladder": [65536]},
        "result": {"elapsed_seconds": 1.5},
    }
    (root / "result.json").write_text(json.dumps(payload))


def test_combiner_sums_moments_before_newton_step(tmp_path):
    information = np.repeat(np.eye(2)[None, ...], 4, axis=0)
    score = np.asarray(
        [[0.01, 0.02], [0.03, -0.01], [-0.02, 0.04], [0.00, 0.01]]
    )
    first = tmp_path / "first"
    second = tmp_path / "second"
    _write_part(first, 0, score[:2], information[:2])
    _write_part(second, 2, score[2:], information[2:])
    output = tmp_path / "combined"

    MODULE.main([
        "--part", str(second),
        "--part", str(first),
        "--output", str(output),
    ])

    result = json.loads((output / "result.json").read_text())
    expected_step = score.sum(axis=0) / 4
    np.testing.assert_allclose(result["summary"]["step"], expected_step)
    np.testing.assert_allclose(
        result["summary"]["estimate"], np.asarray([0.01, -0.02]) + expected_step
    )
    assert result["n_observations"] == 4
    assert [part["start"] for part in result["partitions"]] == [0, 2]
