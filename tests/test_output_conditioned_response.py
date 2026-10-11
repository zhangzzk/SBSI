import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from sbsi.output_conditioned_response import (
    OUTPUT_FEATURES, TRUTH_FEATURES, DiskResponseMoment, predict_scene_draw_responses,
)


def test_artifact_preserves_feature_order_and_physical_predictions(tmp_path):
    xgb = pytest.importorskip("xgboost")
    names = TRUTH_FEATURES + OUTPUT_FEATURES
    # A balanced factorial fixture forces both output coordinates to matter.
    # Random truth features can otherwise win a split in a tiny three-tree fit.
    truths = pd.DataFrame(np.zeros((32, 7)), columns=TRUTH_FEATURES)
    radius = np.exp(np.repeat([-1., 1.], 16))
    magnitude = np.tile(np.repeat([24., 28.], 8), 2)
    flux = 10.**((30. - magnitude)/2.5)
    expected_x = np.column_stack((truths.to_numpy(np.float32), np.log(radius), magnitude)).astype(np.float32)
    labels = expected_x[:, 7]*2+(expected_x[:, 8]-26.)
    matrix = xgb.DMatrix(expected_x, label=labels, feature_names=list(names), nthread=1)
    model = xgb.train({"max_depth": 2, "nthread": 1, "tree_method": "hist", "seed": 8}, matrix, num_boost_round=3)
    path = tmp_path / "model.json"
    model.save_model(path)
    metadata = {"format": "experimental_disk_response_moment_v1", "arm": "output_conditioned",
                "features": names, "response_shear": .2, "y_mean": .2, "y_std": 3,
                "velocity_convention": "radial_tanh(pair_sum_b_times_external_shear)",
                "moment_objective": "E[(s*b^2/2-z*b)/label_std^2]",
                "model_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    (tmp_path / "metadata.json").write_text(json.dumps(metadata))
    loaded = DiskResponseMoment(tmp_path)
    np.testing.assert_allclose(loaded.transformed_outputs([1, 3], [100, 10]), [[0, 25], [np.log(3), 27.5]])
    with pytest.raises(ValueError, match="aligned"):
        loaded.transformed_outputs([1], [100, 10])
    np.testing.assert_allclose(loaded.predict_pairs(truths, radius, flux),
                               model.predict(matrix).astype(float)*3+.2)
    pairs = truths.copy()
    pairs["index_input_p"] = np.repeat([10, 20], 16)
    pairs["index_input_s"] = np.arange(32) + 100
    draws = np.zeros((3, 8, 4))
    draws[..., 2] = np.tile([1, 1, 2, 2, 3, 3, 4, 4], (3, 1))
    draws[..., 3] = [20, 20, 100, 100, 300, 300, 800, 800]
    pooled, counts = predict_scene_draw_responses(loaded, pairs, [10, 20, 30], draws, pair_batch_size=9)
    expected = np.zeros((3, 8))
    for primary in range(2):
        part = pairs.iloc[primary*16:(primary+1)*16]
        for draw in range(8):
            expected[primary, draw] = loaded.predict_pairs(part, np.full(16, draws[primary, draw, 2]),
                                                          np.full(16, draws[primary, draw, 3])).sum()
    np.testing.assert_allclose(pooled, expected, rtol=1e-14, atol=1e-14)
    assert counts["predicted_pair_cells"] < counts["unpooled_pair_draws"]
    # Draws whose radius or flux underflowed to zero take the trees' exact one-sided limit.
    assert all(len(edges) for edges in loaded.output_thresholds)
    limits = draws.copy()
    limits[:, :2, 2:] = 0
    limited, stats = predict_scene_draw_responses(loaded, pairs, [10, 20, 30], limits)
    positive = limits.copy()
    positive[positive == 0] = 1e-200
    positive_values, _ = predict_scene_draw_responses(loaded, pairs, [10, 20, 30], positive)
    np.testing.assert_array_equal(limited, positive_values)
    assert stats["zero_radius_draws"] == 6 and stats["zero_flux_draws"] == 6
    (tmp_path / "metadata.json").write_text(json.dumps(metadata | {"arm": "truth_only"}))
    with pytest.raises(ValueError, match="unsupported"):
        DiskResponseMoment(tmp_path)
    (tmp_path / "metadata.json").write_text(json.dumps(metadata))
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="hash"):
        DiskResponseMoment(tmp_path)
