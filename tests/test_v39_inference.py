"""Physical F8 support and shear-state cache invariants."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from sbsi.shear_map import apply_shear_to_ellipticity
from sbsi.v39_inference import FEATURES, V39CatalogueCache, physical_close_crowding


def test_physical_crowding_matches_bruteforce_without_topk_or_parent_cuts():
    scene = pd.DataFrame({'RA': np.r_[0., np.linspace(.1, 9.9, 25), 11.]/3600,
                          'DEC': 0., 'r': np.linspace(18, 29, 27), 'case': 5})
    rows = np.array([20, 0, 26])
    actual = physical_close_crowding(scene, rows, row_chunk=2)
    expected = []
    for row in rows:
        d = np.abs(scene.RA.to_numpy()-scene.RA.iloc[row])*3600
        keep = (d <= 10) & (np.arange(len(scene)) != row)
        ratio = 10**(-.4*(scene.r.to_numpy()[keep]-scene.r.iloc[row]))
        expected.append([np.log1p(np.sum(ratio*np.exp(-.5*(d[keep]/s)**2))) for s in (.5, 1., 2.)])
    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)
    scene.loc[0, 'case'] = 6
    with pytest.raises(ValueError, match='one independent scene'):
        physical_close_crowding(scene)


def test_v39_cache_changes_only_shape_and_rejects_legacy_features():
    zero = pd.DataFrame([[.2, -.1, 2., 24., .6, .1, .2, .3]], columns=FEATURES)
    probability = {(0., 0.): np.array([.8]), (.02, -.01): np.array([.7])}
    cache = V39CatalogueCache(zero, np.array([1.]), {}, probabilities=probability)
    view = cache.get(.02, -.01)
    e = apply_shear_to_ellipticity(np.array([.2]), np.array([-.1]), .02, -.01)
    np.testing.assert_allclose(view.flow.iloc[0, :2], [e[0][0], e[1][0]])
    np.testing.assert_array_equal(view.flow.iloc[:, 2:], zero.iloc[:, 2:])
    np.testing.assert_array_equal(zero.iloc[0, :2], [.2, -.1])
    np.testing.assert_array_equal(view.detection_probability, [.7])
    with pytest.raises(ValueError, match='unprepared'):
        cache.get(.03, 0.)
    with pytest.raises(ValueError, match='F8'):
        cache.validate_model_features(SimpleNamespace(condition_preprocessor=SimpleNamespace(feature_names=['old'])))
