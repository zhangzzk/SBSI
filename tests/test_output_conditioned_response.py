import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from sbsi.output_conditioned_response import (
    OUTPUT_FEATURES, TRUTH_FEATURES, DiskResponseMoment, OutputConditionedResponse, output_features, predict_scene_draw_responses,
)


def test_physical_flux_and_radius_feature_units_and_invalid_rows():
    values = output_features([1, 3], [100, 10])
    np.testing.assert_allclose(values, [[0, 25], [np.log(3), 27.5]])
    with pytest.raises(ValueError, match="positive"):
        output_features([1, 0], [100, 10])
    with pytest.raises(ValueError, match="aligned"):
        output_features([1], [100, 10])


@pytest.mark.parametrize("conditioned", [False, True])
def test_experimental_artifact_preserves_feature_order_and_physical_predictions(tmp_path, conditioned):
    xgb = pytest.importorskip("xgboost")
    names = TRUTH_FEATURES + (OUTPUT_FEATURES if conditioned else ())
    rng = np.random.default_rng(13)
    truths = pd.DataFrame(rng.normal(size=(32, 7)), columns=TRUTH_FEATURES)
    radius, flux = np.exp(rng.normal(size=32)), np.exp(rng.normal(size=32)+3)
    if conditioned:
        # A balanced factorial fixture forces both output coordinates to matter.
        # Random truth features can otherwise win a split in a tiny three-tree fit.
        truths.loc[:, :] = 0.
        radius = np.exp(np.repeat([-1., 1.], 16))
        magnitude = np.tile(np.repeat([24., 28.], 8), 2)
        flux = 10.**((30. - magnitude)/2.5)
    expected_x = truths.to_numpy(np.float32)
    if conditioned:
        expected_x = np.column_stack((expected_x, output_features(radius, flux))).astype(np.float32)
    labels = expected_x[:, 7]*2+(expected_x[:, 8]-26.) if conditioned else np.arange(32)/32
    matrix = xgb.DMatrix(expected_x, label=labels, feature_names=list(names), nthread=1)
    model = xgb.train({"max_depth": 2, "nthread": 1, "tree_method": "hist", "seed": 8}, matrix, num_boost_round=3)
    path = tmp_path / "model.json"
    model.save_model(path)
    metadata = {"format": "experimental_output_conditioned_response_v1",
                "arm": "output_conditioned" if conditioned else "truth_only", "features": names,
                "response_shear": .2, "y_mean": .2, "y_std": 3,
                "model_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    (tmp_path / "metadata.json").write_text(json.dumps(metadata))
    loaded = OutputConditionedResponse(tmp_path)
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
    if conditioned:
        assert all(len(edges) for edges in loaded.output_thresholds)
        limits = draws.copy()
        limits[:, :2, 2:] = 0
        limited, stats = predict_scene_draw_responses(loaded, pairs, [10, 20, 30], limits)
        positive = limits.copy()
        positive[positive == 0] = 1e-200
        positive_values, _ = predict_scene_draw_responses(loaded, pairs, [10, 20, 30], positive)
        np.testing.assert_array_equal(limited, positive_values)
        assert stats["zero_radius_draws"] == 6 and stats["zero_flux_draws"] == 6
        disk_metadata = metadata | {"format": "experimental_disk_response_moment_v1",
                                   "velocity_convention": "radial_tanh(pair_sum_b_times_external_shear)",
                                   "moment_objective": "E[(s*b^2/2-z*b)/label_std^2]"}
        (tmp_path / "metadata.json").write_text(json.dumps(disk_metadata))
        with pytest.raises(ValueError, match="unsupported"):
            OutputConditionedResponse(tmp_path)
        transport = DiskResponseMoment(tmp_path)
        assert transport.response_coordinate == "disk_velocity"
        np.testing.assert_allclose(transport.predict_pairs(truths, radius, flux), loaded.predict_pairs(truths, radius, flux))
        (tmp_path / "metadata.json").write_text(json.dumps(metadata))
        with pytest.raises(ValueError, match="unsupported"):
            DiskResponseMoment(tmp_path)
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="hash"):
        OutputConditionedResponse(tmp_path)
