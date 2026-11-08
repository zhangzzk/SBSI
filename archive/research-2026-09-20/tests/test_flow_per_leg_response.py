import numpy as np
import torch

from scripts.audit_flow_per_leg_response import moment_sufficient, response_conventions, row_moments
from scripts.audit_harmonized_usability_response import aggregate, response, sufficient


def test_exact_draws_reproduce_label_estimator_with_per_leg_missingness():
    rng = np.random.default_rng(101)
    gamma = rng.normal(0, .03, (30, 2))
    zero = np.column_stack((rng.normal(0, .1, (30, 2)), rng.uniform(2.8, 3.3, 30), rng.uniform(35, 70, 30)))
    shear = zero.copy()
    shear[:, :2] += .7*gamma
    shear[:, 2] += 2*gamma[:, 0]
    u0, ug = np.arange(30) != 3, np.arange(30) != 7
    for hard in (False, True):
        moments = [row_moments(torch.tensor(value[:, None, :]), hard) for value in (zero, shear)]
        for paired in (False, True):
            raw = moment_sufficient(*moments, gamma, u0, ug, paired)
            ref = sufficient(zero, shear, gamma, u0, ug, hard=hard, paired=paired)
            for key in ref:
                np.testing.assert_allclose(raw[key], ref[key], atol=3e-15)
            agg = aggregate([raw], np.ones((1, 1)))
            modes = response_conventions(agg, agg)
            np.testing.assert_array_equal(modes['own_normalization'], modes['measured_normalization'])


def test_normalization_conventions_do_not_silently_rescale_predictions():
    raw = {'n': np.array([10]), 'normal': np.eye(2)[None], 'mass_zero': np.array([[5., 5.]]),
           'shape_zero': np.zeros((1, 2, 2)), 'delta_mass_gamma': np.zeros((1, 2, 2)),
           'delta_shape_gamma': np.ones((1, 2, 2, 2))}
    observed = dict(raw, mass_zero=raw['mass_zero']/2)
    modes = response_conventions(raw, observed)
    np.testing.assert_array_equal(modes['own_normalization'], response(raw))
    np.testing.assert_array_equal(modes['measured_normalization'], 2*modes['own_normalization'])


def test_zero_photometry_retained_uncut_and_excluded_by_joint_cut():
    values = torch.tensor([[[.1, .2, 0., 0.], [.1, .2, 4., 100.]]], dtype=torch.float64)
    mass, shape = row_moments(values, True)
    np.testing.assert_array_equal(mass, [[1., .5]])
    np.testing.assert_allclose(shape, [[[.1, .2], [.05, .1]]])
    soft, _ = row_moments(values, False)
    assert np.isfinite(soft).all() and soft[0, 0] == 1


def test_exact_per_leg_density_can_fail_both_usable_response_guard():
    """Other-leg usability can select noise that the marginal density cannot know."""
    from itertools import product

    h, amplitude, true_response = .05, .01, .5
    zero, shear, gammas, usable_zero, usable_shear = [], [], [], [], []
    for axis, sign, zx, zy, w in product((0, 1), (-1, 1), (-1, 1), (-1, 1), (-1, 1)):
        gamma = np.zeros(2)
        gamma[axis] = sign*h
        shape = amplitude*np.array([zx, zy])
        zero.append([*shape, 4., 100.])
        shear.append([*(-shape+true_response*gamma), 4., 100.])
        gammas.append(gamma)
        usable_zero.append(w == 1)
        usable_shear.append(np.sign(shape[axis])*sign == -w)
    zero, shear, gamma = map(np.asarray, (zero, shear, gammas))
    u0, ug = map(np.asarray, (usable_zero, usable_shear))
    # U0 is independent of the assigned shear. Conditional on each gamma,
    # every one of the four output atoms has equal selected frequency in
    # both legs, so the oracle's *entire* per-leg distribution is exact.
    atoms = amplitude*np.array(list(product((-1, 1), repeat=2)))
    for direction in np.unique(gamma, axis=0):
        assigned = (gamma == direction).all(1)
        for values, usable, offset in ((zero, u0, np.zeros(2)), (shear, ug, true_response*direction)):
            for atom in atoms:
                count = (np.isclose(values[:, :2], atom+offset).all(1) & usable & assigned).sum()
                assert count == 1
    mass = np.ones((len(gamma), 2))
    # Both hard cuts pass deterministically. The oracle marginal means are
    # zero and true_response*gamma, with no information about the other U.
    oracle_zero = (mass, np.zeros((len(gamma), 2, 2)))
    oracle_shear = (mass, np.repeat((true_response*gamma)[:, None, :], 2, axis=1))
    values = {}
    for paired in (False, True):
        measured = sufficient(zero, shear, gamma, u0, ug, hard=True, paired=paired)
        model = moment_sufficient(oracle_zero, oracle_shear, gamma, u0, ug, paired)
        values[paired] = {name: response(aggregate([raw], np.ones((1, 1))))[0]
                          for name, raw in [('measured', measured), ('model', model)]}
    np.testing.assert_allclose(values[False]['measured'], values[False]['model'], atol=1e-15)
    np.testing.assert_allclose(values[False]['model'], np.broadcast_to(true_response*np.eye(2), (2, 2, 2)), atol=1e-15)
    np.testing.assert_allclose(values[True]['model'], values[False]['model'], atol=1e-15)
    np.testing.assert_allclose(values[True]['measured']-values[True]['model'],
                               np.broadcast_to(2*amplitude/h*np.eye(2), (2, 2, 2)), atol=1e-15)
