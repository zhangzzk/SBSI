from itertools import product

import numpy as np
import pytest

from scripts.matched_response_probe import (
    baseline_reference, coherent_response, influence_difference, probe_response, summarize_probes,
)


def test_exhaustive_rectangular_probe_recovers_selected_self_and_neighbor_response():
    # Four sheared sources, only sources 1 and 3 belong to the truth parent.
    ids = np.array([1, 3])
    mass = np.array([.7, .6])
    numerator = np.array([[.12, -.08], [-.03, .06]])
    mass_jac = np.array([[.1, .2, -.1, .15], [-.2, .05, .1, -.15]])
    jac = np.array([[[.1, .6, .2, .3], [-.2, .1, .3, -.1]],
                    [[-.1, .2, .3, .5], [.4, -.2, .1, .7]]])
    mu, total = baseline_reference(numerator/mass[:, None], mass)
    effective = jac-mu[None, :, None]*mass_jac[:, None, :]
    expected_self = sum(effective[i, :, source] for i, source in enumerate(ids))/total
    expected_total = effective.sum((0, 2))/total
    h = .02

    def leg(g):
        weight = mass+mass_jac@g
        value = numerator+np.einsum('ioj,j->io', jac, g)
        return value/weight[:, None], weight

    probes = []
    for signs in np.array(list(product([-1, 1], repeat=4))):
        plus, wp = leg(h*signs)
        minus, wm = leg(-h*signs)
        probes.append(probe_response(plus, minus, wp, wm, parent_signs=signs[ids],
                                     mean_zero=mu, mass_zero=total, h=h))
    plus, wp = leg(np.full(4, h))
    minus, wm = leg(np.full(4, -h))
    coherent = coherent_response(plus, minus, wp, wm, mean_zero=mu, mass_zero=total, h=h)
    np.testing.assert_allclose(coherent['baseline_influence'], expected_total, atol=1e-14)
    np.testing.assert_allclose(np.mean(probes, axis=0), expected_self, atol=1e-14)
    summary = summarize_probes(probes, coherent['baseline_influence'])
    np.testing.assert_allclose(summary['neighbor_response'], expected_total-expected_self, atol=1e-14)
    # The finite-shear endpoint ratio is a separate quantity, not silently relabeled.
    assert not np.allclose(coherent['endpoint_ratio'], coherent['baseline_influence'], atol=1e-10, rtol=0)


def test_per_leg_selection_retains_entering_and_leaving_rows():
    plus = np.array([[.3, .1], [np.nan, np.nan]])
    minus = np.array([[np.nan, np.nan], [.2, -.1]])
    mu = np.array([.05, .02])
    got = influence_difference(plus, minus, [1, 0], [0, 1], mean_zero=mu, mass_zero=1, h=.1)
    np.testing.assert_allclose(got, np.array([[.25, .08], [-.15, .12]])/.2)
    with pytest.raises(ValueError, match='selected shapes'):
        influence_difference(plus, minus, [1, 1], [0, 1], mean_zero=mu, mass_zero=1, h=.1)


def test_exactly_balanced_signs_do_not_have_required_cross_moment():
    probes = np.array([s for s in product([-1, 1], repeat=4) if sum(s) == 0])
    moment = probes.T@probes/len(probes)
    np.testing.assert_allclose(moment[0, 1:], -1/3)
    assert not np.allclose(moment, np.eye(4))


def test_single_probe_cannot_report_probe_uncertainty():
    with pytest.raises(ValueError, match='at least two'):
        summarize_probes([[.5, .01]], [.7, .02])
