import json

import numpy as np
import pandas as pd
import pytest

from _script_loader import load_script_module


MODULE = load_script_module("summarize_catalogue_bias.py")


def _result(root, injected, estimate, *, token, stream=None):
    root.mkdir()
    pd.DataFrame({"x": [token]}).to_parquet(root / "measurements.parquet")
    pd.DataFrame({"x": [token]}).to_parquet(root / "truth.parquet")
    payload = {
        "profile": [
            {
                "injected_shear": injected,
                "proposal_seed": seed,
                "rungs": [
                    {
                        "n_draws": 32,
                        "estimated_shear": estimate + offset,
                        "quadratic_information": 10_000.0,
                    }
                ],
            }
            for seed, offset in ((11, -0.001), (12, 0.001))
        ],
        "config": {
            "direction_g1": 1.0,
            "direction_g2": 0.0,
            "n_detected": 100,
            "scene_seed": token if stream is None else stream,
            "detection_seed": 100 + (token if stream is None else stream),
            "flow_seed": 200 + (token if stream is None else stream),
        },
        "selection": {"cut_key": "cut"},
        "model_sha256": {"flow": "abc"},
    }
    path = root / "result.json"
    path.write_text(json.dumps(payload))
    return path


def test_bias_summary_recovers_known_line(tmp_path):
    paths = [
        _result(tmp_path / f"r{i}", g, 0.002 + 1.05 * g, token=i) for i, g in enumerate((-0.02, 0.0, 0.02))
    ]
    summaries = [MODULE.profile_summary(path) for path in paths]
    result = MODULE.fit_bias(summaries)
    assert result["additive_bias"] == pytest.approx(0.002)
    assert result["multiplicative_bias"] == pytest.approx(0.05)
    assert result["n_mocks"] == 3
    assert all(row["proposal_mc_sd"] == pytest.approx(np.sqrt(2) * 0.001) for row in summaries)


def test_bias_summary_rejects_duplicate_mock(tmp_path):
    paths = [_result(tmp_path / f"r{i}", g, g, token=i) for i, g in enumerate((-0.02, 0.0, 0.02))]
    summaries = [MODULE.profile_summary(path) for path in paths]
    summaries[2]["mock_sha256"] = summaries[1]["mock_sha256"]
    with pytest.raises(ValueError, match="same frozen mock"):
        MODULE.fit_bias(summaries)


def test_bias_summary_uses_paired_stream_blocks(tmp_path):
    paths = []
    block_m = (0.02, 0.04, 0.06)
    for block, multiplicative in enumerate(block_m):
        for slot, injected in enumerate((-0.02, 0.0, 0.02)):
            paths.append(
                _result(
                    tmp_path / f"b{block}s{slot}",
                    injected,
                    0.001 + (1 + multiplicative) * injected,
                    token=10 * block + slot,
                    stream=block,
                )
            )
    result = MODULE.fit_bias([MODULE.profile_summary(path) for path in paths])
    paired = result["paired_blocks"]
    assert paired["n_blocks"] == 3
    assert paired["additive_bias"] == pytest.approx(0.001)
    assert paired["multiplicative_bias"] == pytest.approx(0.04)
    assert paired["multiplicative_se"] == pytest.approx(0.02 / np.sqrt(3))
