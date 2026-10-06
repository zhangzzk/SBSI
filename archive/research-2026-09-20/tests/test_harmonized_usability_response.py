import numpy as np
import pytest

from sbsi.flow_guard_response import fit_guard_response
from scripts.audit_harmonized_usability_response import CUTS, aggregate, response, sufficient


def sample():
    rng = np.random.default_rng(41)
    gamma = .05*np.column_stack((np.cos(np.arange(200)), np.sin(np.arange(200))))
    zero = np.column_stack((rng.normal(0, .1, (200, 2)), rng.uniform(2.8, 3.3, 200), rng.uniform(35, 70, 200)))
    shear = zero.copy()
    shear[:, :2] += gamma @ np.array([[.6, .01], [.02, .7]]).T
    shear[:, 2] += gamma[:, 0]*2
    return zero, shear, gamma


def test_paired_reproduces_existing_influence_estimator():
    zero, shear, gamma = sample()
    usable = np.ones(len(zero), dtype=bool)
    raw = sufficient(zero, shear, gamma, usable, usable, hard=False, paired=True)
    actual = response(aggregate([raw], np.ones((1, 1))))[0]
    expected = fit_guard_response(zero, shear, gamma, CUTS, radius_softness=.05, magnitude_softness=.05)['target'].reshape(2, 2, 2)
    np.testing.assert_allclose(actual, expected, atol=2e-15)


def test_per_leg_keeps_one_leg_only_mass_and_additivity():
    zero, shear, gamma = sample()
    u0 = np.ones(len(zero), dtype=bool)
    ug = u0.copy()
    ug[:20] = False
    shear[~ug] = np.nan
    full = sufficient(zero, shear, gamma, u0, ug, hard=True, paired=False)
    pieces = [sufficient(zero[s], shear[s], gamma[s], u0[s], ug[s], hard=True, paired=False) for s in [slice(0, 100), slice(100, None)]]
    combined = aggregate(pieces, np.ones((1, 2)))
    for key in full:
        np.testing.assert_allclose(combined[key][0], full[key], atol=1e-13)
    paired = sufficient(zero, shear, gamma, u0, ug, hard=True, paired=True)
    assert full['n'] == 200 and paired['n'] == 180
    assert np.isfinite(response(combined)).all()
    assert not np.allclose(response(combined), response(aggregate([paired], np.ones((1, 1)))))


def test_unusable_nan_allowed_but_usable_nan_rejected():
    zero, shear, gamma = sample()
    usable = np.ones(len(zero), dtype=bool)
    shear[0, 0] = np.nan
    with pytest.raises(ValueError, match='usable measurements'):
        sufficient(zero, shear, gamma, usable, usable, hard=True, paired=False)
