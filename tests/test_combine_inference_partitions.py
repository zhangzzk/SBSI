import json
from hashlib import sha256

import numpy as np
import pytest

from _script_loader import load_script_module


MODULE = load_script_module("combine_inference_partitions.py")


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
        "pipeline_release": "v1.1-infer",
        "pipeline_base_release": "v1.1-infer",
        "pipeline_config_sha256": "pipeline-config",
        "pipeline_resolved_config_sha256": "resolved-config",
        "likelihood_release": "v3.2-like",
        "likelihood_config_sha256": "likelihood-config",
        "pipeline_config": {
            "estimator": {"draw_ladder": [65536]},
        },
        "likelihood_component_sha256": {"measurement": "m"},
        "pipeline_implementation_sha256": {"runner": "i"},
        "likelihood_implementation_sha256": {"likelihood": "i"},
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
    score = np.asarray([[0.01, 0.02], [0.03, -0.01], [-0.02, 0.04], [0.00, 0.01]])
    first = tmp_path / "first"
    second = tmp_path / "second"
    _write_part(first, 0, score[:2], information[:2])
    _write_part(second, 2, score[2:], information[2:])
    output = tmp_path / "combined"

    MODULE.main(
        [
            "--part",
            str(second),
            "--part",
            str(first),
            "--output",
            str(output),
        ]
    )

    result = json.loads((output / "result.json").read_text())
    expected_step = score.sum(axis=0) / 4
    np.testing.assert_allclose(result["summary"]["step"], expected_step)
    np.testing.assert_allclose(result["summary"]["estimate"], np.asarray([0.01, -0.02]) + expected_step)
    assert result["n_observations"] == 4
    assert result["pipeline_release"] == "v1.1-infer"
    assert result["likelihood_release"] == "v3.2-like"
    assert [part["start"] for part in result["partitions"]] == [0, 2]


def test_combiner_rejects_mixed_release_identity(tmp_path):
    information = np.repeat(np.eye(2)[None, ...], 4, axis=0)
    score = np.ones((4, 2))
    first = tmp_path / "first"
    second = tmp_path / "second"
    _write_part(first, 0, score[:2], information[:2])
    _write_part(second, 2, score[2:], information[2:])
    result_path = second / "result.json"
    payload = json.loads(result_path.read_text())
    payload["likelihood_release"] = "other-like"
    result_path.write_text(json.dumps(payload))

    with pytest.raises(RuntimeError, match="likelihood_release"):
        MODULE.main(
            [
                "--part",
                str(first),
                "--part",
                str(second),
                "--output",
                str(tmp_path / "combined"),
            ]
        )


def _two_parts(tmp_path, n_total):
    information = np.repeat(np.eye(2)[None, ...], 4, axis=0)
    score = np.ones((4, 2)) * 0.01
    first, second = tmp_path / "first", tmp_path / "second"
    _write_part(first, 0, score[:2], information[:2])
    _write_part(second, 2, score[2:], information[2:])
    for root in (first, second):
        payload = json.loads((root / "result.json").read_text())
        payload["observation_partition"]["n_total"] = n_total
        (root / "result.json").write_text(json.dumps(payload))
    return [str(first), str(second)]


def test_a_partial_window_needs_an_explicit_opt_in(tmp_path):
    """cont.345: the hybrid chain evaluates 20,000 rows of a 500,000-row mock.

    The default refusal exists so a run that silently lost a partition cannot
    be reported as complete, which is worth keeping.  A deliberate window has
    to say so, and the report records which it was either way.
    """

    parts = _two_parts(tmp_path, n_total=10)
    argv = ["--part", parts[0], "--part", parts[1], "--output", str(tmp_path / "out")]
    with pytest.raises(RuntimeError, match="allow-partial-window"):
        MODULE.main(argv)

    MODULE.main(argv + ["--allow-partial-window"])
    report = json.loads((tmp_path / "out" / "result.json").read_text())
    assert report["source_window"] == {
        "start": 0,
        "stop": 4,
        "n_total": 10,
        "complete": False,
    }


def test_a_complete_window_is_still_reported_as_complete(tmp_path):
    parts = _two_parts(tmp_path, n_total=4)
    MODULE.main(
        ["--part", parts[0], "--part", parts[1], "--output", str(tmp_path / "out")]
    )
    report = json.loads((tmp_path / "out" / "result.json").read_text())
    assert report["source_window"]["complete"] is True


def test_a_gap_between_partitions_is_still_refused(tmp_path):
    """--allow-partial-window relaxes the end of the window, not its interior."""

    information = np.repeat(np.eye(2)[None, ...], 4, axis=0)
    score = np.ones((4, 2)) * 0.01
    first, third = tmp_path / "first", tmp_path / "third"
    _write_part(first, 0, score[:2], information[:2])
    _write_part(third, 2, score[2:], information[2:])
    payload = json.loads((third / "result.json").read_text())
    payload["observation_partition"]["start"] = 3
    payload["observation_partition"]["stop"] = 5
    (third / "result.json").write_text(json.dumps(payload))
    with pytest.raises(RuntimeError, match="not contiguous"):
        MODULE.main(
            ["--part", str(first), "--part", str(third),
             "--output", str(tmp_path / "out"), "--allow-partial-window"]
        )


def test_a_refused_combine_leaves_no_directory_to_block_the_retry(tmp_path):
    """The first hybrid chain died here: the coverage check fired, an empty
    output directory was already on disk, and the corrected rerun then refused
    to overwrite it."""

    parts = _two_parts(tmp_path, n_total=10)
    output = tmp_path / "out"
    argv = ["--part", parts[0], "--part", parts[1], "--output", str(output)]
    with pytest.raises(RuntimeError):
        MODULE.main(argv)
    assert not output.exists()
    MODULE.main(argv + ["--allow-partial-window"])
    assert (output / "result.json").exists()
