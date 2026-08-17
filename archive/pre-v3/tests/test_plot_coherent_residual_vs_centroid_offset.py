import numpy as np
import pandas as pd

from scripts.plot_coherent_residual_vs_centroid_offset import (
    OFFSET,
    PREDICTION,
    RESIDUAL,
    TRUTH,
    combine_leg_offsets,
    conditional_curve,
    paired_bin_contrast,
)


def test_combine_leg_offsets_uses_maximum_and_raw_v22_residual():
    target = pd.DataFrame({
        "case": [700, 700],
        "input_index": [10, 11],
        TRUTH: [0.3, -0.1],
        PREDICTION: [0.2, -0.3],
    })
    plus = pd.DataFrame({
        "input_index": [10, 11],
        "distance_pixels": [0.2, 1.5],
        "detection_index": [100, 101],
    })
    minus = pd.DataFrame({
        "input_index": [10, 11],
        "distance_pixels": [0.5, 1.0],
        "detection_index": [200, 201],
    })
    got = combine_leg_offsets(target, plus, minus, pixel_scale=0.2)
    np.testing.assert_allclose(got[OFFSET], [0.1, 0.3])
    np.testing.assert_allclose(got[RESIDUAL], [0.1, 0.2])


def test_conditional_curve_is_case_balanced_and_contrast_is_paired():
    rows = []
    for case in range(4):
        for index, offset in enumerate(np.linspace(0.01, 0.32, 8)):
            prediction = 0.2 + 0.01 * case
            residual = (0.5 + 0.02 * case) * offset + 0.001 * case
            rows.append({
                "case": case,
                "input_index": 100 * case + index,
                OFFSET: offset,
                TRUTH: prediction + residual,
                PREDICTION: prediction,
                RESIDUAL: residual,
            })
    frame = pd.DataFrame(rows)
    curve, work, edges = conditional_curve(frame, n_bins=4)
    assert len(curve) == 4
    assert np.isneginf(edges[0]) and np.isposinf(edges[-1])
    assert (curve.n_cases == 4).all()
    assert (curve.n_anchors == 8).all()
    assert np.all(np.diff(curve.residual_truth_minus_v22) > 0.0)
    contrast = paired_bin_contrast(work, 0, 3)
    assert contrast["n_cases"] == 4
    assert contrast["mean"] > 0.0
