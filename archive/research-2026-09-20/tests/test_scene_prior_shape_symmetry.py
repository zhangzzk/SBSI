import numpy as np

from scripts.audit_scene_prior_shape_symmetry import accumulate, angular_summary


def test_complete_shape_orbit_is_symmetric_with_scalar_weights():
    sums = {}
    # Physical angle actions induce (e1,e2), (e1,-e2), (-e1,e2), (-e1,-e2).
    accumulate(np.full(4, .5), [13., -13., 77., 103.], np.full(4, 24.), np.full(4, .6), sums)
    for entry in sums.values():
        result = angular_summary(entry['hist'].sum(0), entry['shape'].sum(0), entry['mass'].sum())
        assert result['orbit_tv'] == 0.
        np.testing.assert_allclose(result['mean_shape'], 0., atol=1e-15)


def test_coherent_prior_is_not_declared_symmetric():
    sums = {}
    accumulate([.5]*3, [0.]*3, [24.]*3, [.6]*3, sums)
    entry = sums['number']
    result = angular_summary(entry['hist'].sum(0), entry['shape'].sum(0), entry['mass'].sum())
    np.testing.assert_allclose(result['mean_shape'], [1/3, 0.])
    assert result['orbit_tv'] == .5
