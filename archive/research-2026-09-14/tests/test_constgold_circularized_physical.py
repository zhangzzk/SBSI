from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from scripts.evaluate_constgold_complete_forward_response import (
    _prediction_sufficient, _with_circularized_condition,
)
from sbsi.flow_size_condition import CIRCULARIZED_FEATURE


def test_constgold_invariant_size_uses_baseline_truth_and_preserves_sheared_frame():
    baseline = pd.DataFrame({"index": [9, 2], "Re": [1., .8],
                             "axis_ratio": [.25, 1.], "position_angle": [15., 0.]})
    plus = pd.DataFrame({"input_index": [9, 2], "Re_input_p": [1., .8],
                         "axis_ratio_input_p": [.3, .9], "e1_input_p": [.5, .02]})
    minus = plus.assign(axis_ratio_input_p=[.2, .9], e1_input_p=[.6, -.02])
    for frame in (plus, minus):
        actual = _with_circularized_condition(frame, baseline)
        np.testing.assert_allclose(actual[CIRCULARIZED_FEATURE], [.5, .8], rtol=0, atol=1e-15)
        pd.testing.assert_frame_equal(actual.drop(columns=CIRCULARIZED_FEATURE), frame)
        assert CIRCULARIZED_FEATURE not in frame
    with pytest.raises(ValueError, match="aligned"):
        _with_circularized_condition(plus, baseline.iloc[::-1])


def physical_bundle(values):
    names = ("measured_ngmix_g1", "measured_ngmix_g2",
             "measured_flux_radius", "measured_flux_from_mag_auto")
    return SimpleNamespace(target_transform=SimpleNamespace(target_names=names),
                           sample=lambda *args, **kwargs: values.copy())


def test_physical_underflow_keeps_uncut_shapes_and_obeys_each_selection_cut():
    values = np.array([[[.1, 0., 0., 100.], [.2, 0., 4., 0.], [.3, 0., 4., 100.]]])
    frame = pd.DataFrame({"input_index": [1]})
    kwargs = dict(detection_probability=np.ones(1), blend_shift=np.zeros((1, 2)),
                  draws=3, seed=1, batch_size=1, abs_shape_max=None,
                  fixed_population_masks={"matched_detected": np.ones(1, dtype=bool),
                                          "matched_detected_selected": np.ones(1, dtype=bool)})
    for mag, radius, count, shape_sum in [(None, None, 3, .6),
                                         (25.8, None, 2, .4),
                                         (None, .75, 2, .5),
                                         (25.8, .75, 1, .3)]:
        stats, _ = _prediction_sufficient(physical_bundle(values), frame,
            mag_max=mag, radius_min_arcsec=radius, **kwargs)
        assert stats["matched_detected"]["denominator"] == 1
        np.testing.assert_allclose(stats["matched_detected"]["numerator"], [.2, 0.])
        assert stats["matched_detected_selected"]["denominator"] == count / 3
        np.testing.assert_allclose(stats["matched_detected_selected"]["numerator"], [shape_sum / 3, 0.])
        assert stats["physical_sample_diagnostics"]["zero_radius_flux_draws"] == [1, 1]


@pytest.mark.parametrize("value", [-1., np.nan, np.inf])
def test_invalid_physical_scalar_is_rejected(value):
    samples = np.array([[[.1, 0., value, 100.]]])
    with pytest.raises(ValueError, match="finite and nonnegative"):
        _prediction_sufficient(physical_bundle(samples), pd.DataFrame({"input_index": [1]}),
            detection_probability=np.ones(1), blend_shift=np.zeros((1, 2)),
            draws=1, seed=1, batch_size=1, abs_shape_max=.6)
