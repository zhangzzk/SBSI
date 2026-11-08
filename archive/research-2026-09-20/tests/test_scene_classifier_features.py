import numpy as np
import pandas as pd
import pytest

from scripts.scene_classifier_features import geometry_features


def pairs():
    return pd.DataFrame({'index_input_p': [7, 7, 3, 3, 3], 'index_input_s': [10, 11, 12, 13, 14],
        'r_input_p_scaled': [25.]*5, 'r_input_s_scaled': [25., 22.5, 25., 25., 25.],
        'distance_scaled': [.49, .5, 1., 2., 4.], 'response': [999.]*5})


def test_flux_ratios_boundaries_order_and_missing_neighbours():
    x, report = geometry_features(np.array([7, 8, 3]), pairs())
    np.testing.assert_allclose(np.expm1(x), [[1, 10, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 1, 1, 1]], rtol=2e-7)
    assert report['parents_without_pairs'] == 1
    assert report['pairs_per_annulus'] == [1]*5
    changed = pairs().iloc[::-1].copy()
    changed['response'] = -123.
    changed[['r_input_p_scaled', 'r_input_s_scaled']] += 17.
    y, _ = geometry_features(np.array([7, 8, 3]), changed)
    np.testing.assert_array_equal(x, y)


def test_empty_pairs_preserve_all_parents():
    x, report = geometry_features(np.array([7, 3]), pairs().iloc[:0])
    np.testing.assert_array_equal(x, np.zeros((2, 5)))
    assert report['parents_without_pairs'] == 2


@pytest.mark.parametrize('failure', ['duplicate', 'self', 'missing', 'negative', 'nonfinite'])
def test_reject_invalid_pairs(failure):
    frame = pairs()
    if failure == 'duplicate':
        frame = pd.concat([frame, frame.iloc[:1]], ignore_index=True)
    elif failure == 'self':
        frame.loc[0, 'index_input_s'] = 7
    elif failure == 'missing':
        frame.loc[0, 'index_input_p'] = 99
    elif failure == 'negative':
        frame.loc[0, 'distance_scaled'] = -.1
    else:
        frame.loc[0, 'r_input_s_scaled'] = np.nan
    with pytest.raises(ValueError):
        geometry_features(np.array([7, 3]), frame)
