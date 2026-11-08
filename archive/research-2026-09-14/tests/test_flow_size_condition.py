import numpy as np
import pandas as pd
import pytest

from sbsi.flow_size_condition import (
    CIRCULARIZED_FEATURE, circularized_context, circularized_radius,
    fit_circularized_preprocessor,
)
from sbsi.selection_model import TabularPreprocessor
from sbsi.shear_map import apply_shear_to_ellipticity


def test_invariant_radius_matches_general_area_preserving_shear_geometry():
    # Independent 2x2 shape-matrix calculation, including off-axis shears.
    a = np.array([1., .7, 1.3, .9])
    q = np.array([.4, .8, 1., .15])
    phi = np.array([.2, .7, 1.4, 2.])
    g1 = np.array([.05, -.03, .01, .04])
    g2 = np.array([.02, .04, -.03, .01])
    e = (1-q)/(1+q)
    e1, e2 = apply_shear_to_ellipticity(e*np.cos(2*phi), e*np.sin(2*phi), g1, g2)
    sheared_q = (1-np.hypot(e1, e2))/(1+np.hypot(e1, e2))
    for i in range(len(a)):
        rotation = np.array([[np.cos(phi[i]), -np.sin(phi[i])],
                             [np.sin(phi[i]), np.cos(phi[i])]])
        covariance = rotation @ np.diag([a[i]**2, (a[i]*q[i])**2]) @ rotation.T
        shear = np.array([[1+g1[i], g2[i]], [g2[i], 1-g1[i]]])
        shear /= np.sqrt(1-g1[i]**2-g2[i]**2)
        eigenvalues = np.linalg.eigvalsh(shear @ covariance @ shear.T)
        np.testing.assert_allclose(np.sqrt(eigenvalues[0]/eigenvalues[1]), sheared_q[i])
        np.testing.assert_allclose(np.linalg.det(covariance)**.25,
                                   circularized_radius(a[i], q[i]))
        np.testing.assert_allclose(np.sqrt(eigenvalues[1]*sheared_q[i]),
                                   circularized_radius(a[i], q[i]))
    assert not np.allclose(circularized_radius(a, sheared_q), circularized_radius(a, q))


@pytest.mark.parametrize("a,q", [(0, .5), (1, 0), (1, 1.1), (np.nan, .5), (1, np.inf)])
def test_invalid_truth_is_rejected(a, q):
    with pytest.raises(ValueError, match="pre-shear"):
        circularized_radius(a, q)


def test_train_only_preprocessor_roundtrip_and_shear_leg_invariance():
    frame = pd.DataFrame({"e1_input_p": [.1, .2, .3, .4],
                          "Re_input_p": [1., 1., 2., 2.], "other": [3., np.nan, 7., 8.]})
    pre = TabularPreprocessor.fit(frame, list(frame))
    original_means = pre.means.copy()
    radius = circularized_radius(frame.Re_input_p, [.6, .6, .4, .4])
    new = fit_circularized_preprocessor(pre, radius[:2+1])
    new = TabularPreprocessor.from_state(new.to_state())
    actual = circularized_context(pre.transform_frame(frame), radius, new)
    expected_frame = frame.rename(columns={"Re_input_p": CIRCULARIZED_FEATURE}).copy()
    expected_frame[CIRCULARIZED_FEATURE] = radius
    np.testing.assert_array_equal(actual, new.transform_frame(expected_frame))
    np.testing.assert_array_equal(actual[:, [0, 2, 3, 5]], pre.transform_frame(frame)[:, [0, 2, 3, 5]])
    np.testing.assert_array_equal(actual[[0, 2], 1], actual[[1, 3], 1])
    np.testing.assert_array_equal(pre.means, original_means)
    assert pre.feature_names[1] == "Re_input_p"
    with pytest.raises(KeyError):
        new.transform_frame(frame)
