import numpy as np
import pandas as pd

from scripts.prepare_independent_truth_parent import intrinsic_context, select_truth_parent


def test_generated_role_and_strict_cut_precede_rendered_join():
    generated = pd.DataFrame({'index': range(6), 'r': [20., 20., 20., 25., np.nextafter(26., 0.), 26.]})
    rendered = pd.DataFrame({'index': [0, 1, 2, 3, 5], 'r': [20., 20., 20., 25., 26.]})
    parent, report = select_truth_parent(generated, rendered)
    assert parent['index'].tolist() == [3]
    assert report['generated_truth_parent_rows'] == 2
    assert report['unrendered_parent_ids'] == [4]


def test_intrinsic_context_does_not_fold_observed_shear_or_require_g0():
    scene = pd.DataFrame({'index': [0, 1], 'RA': [180., 180.0001], 'DEC': [0., 0.],
                          'redshift': [1., 1.], 'Re': [1., 1.], 'axis_ratio': [.5, .7],
                          'position_angle': [0., 0.], 'sersic_n': [1., 1.], 'r': [24., 25.],
                          'gamma1': [.05, .05], 'gamma2': [0., 0.]})
    plus = intrinsic_context(scene.iloc[:1].copy(), scene)
    minus = intrinsic_context(scene.iloc[:1].assign(gamma1=-.05), scene.assign(gamma1=-.05))
    np.testing.assert_array_equal(plus, minus)
    np.testing.assert_allclose(plus[0, :2], [1/3, 0.], atol=1e-7)
    np.testing.assert_allclose(plus[0, 4], np.sqrt(.5), atol=1e-7)
