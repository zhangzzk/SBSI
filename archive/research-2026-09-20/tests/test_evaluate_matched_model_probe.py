import numpy as np
import pytest

from scripts.evaluate_matched_model_probe import baseline, paired_difference, response_parts, sufficient_rows, summarize, validate_scope
from scripts.matched_response_probe import influence_difference


def test_sufficient_response_matches_draw_expansion_with_changing_selection():
    # Distinct joint draws enter and leave; primary 1 has no selected minus draw.
    plus = np.array([[[.2, .1], [.4, -.2]], [[.1, .3], [.5, .2]]])
    minus = np.array([[[.1, .2], [.3, -.1]], [[.2, .1], [.1, -.2]]])
    wp, wm = np.array([[1, 0], [1, 1]]), np.array([[1, 1], [0, 0]])
    shiftp, shiftm = np.full_like(plus, .03), np.full_like(minus, -.02)
    mean, mass = np.array([.1, -.03]), 1.4
    signs = np.array([-1, 1])
    def rows(shape, shift, weight):
        return sufficient_rows(weight.mean(1), (weight[..., None]*shape).mean(1),
                               (weight[..., None]*shift).mean(1))
    got = response_parts(rows(plus, shiftp, wp), rows(minus, shiftm, wm), mean, mass, signs=signs)
    expanded = influence_difference((plus+shiftp).reshape(-1, 2), (minus+shiftm).reshape(-1, 2),
                                     wp.ravel(), wm.ravel(), mean_zero=mean, mass_zero=mass*2, h=.02)
    np.testing.assert_allclose(got['total'], (expanded*np.repeat(signs, 2)[:, None]).sum(0), atol=1e-14)
    np.testing.assert_allclose(got['total'], got['flow_component']+got['blend_component'], atol=1e-14)


def test_own_reference_and_paired_probe_uncertainty_have_explicit_meaning():
    zero = sufficient_rows([.6, .8], [[.12, .03], [-.04, .02]], np.zeros((2, 2)))
    rows = {'zero': zero}
    delta = np.array([[.02, .003, .001, .001, 0], [-.01, .004, -.001, .002, .001]])
    for axis in (1, 2):
        rows[f'g{axis}+'], rows[f'g{axis}-'] = zero+delta, zero-delta
        for p in (0, 1):
            rows[f'p{p}g{axis}+'], rows[f'p{p}g{axis}-'] = zero+(p+1)*delta, zero-(p+1)*delta
    signs = np.array([[1, -1], [-1, -1]])
    own = summarize(rows, signs)
    reference = summarize(rows, signs, (np.array([-.02, .07]), 1.1))
    assert not np.allclose(own['axes']['g1']['self_response'], reference['axes']['g1']['self_response'])
    difference = paired_difference(own, own)
    for result in difference.values():
        np.testing.assert_array_equal(result['conditional_probe_se'], [0, 0])
        np.testing.assert_array_equal(result['self_response'], [0, 0])
    mu, mass = baseline(zero)
    assert mass == 1.4
    np.testing.assert_allclose(mu, zero[:, 1:3].sum(0)/1.4)


def test_zero_mass_numerator_and_empty_baseline_are_rejected():
    with pytest.raises(ValueError, match='zero selected mass'):
        sufficient_rows([0], [[.1, 0]], [[0, 0]])
    with pytest.raises(ValueError, match='positive zero-leg'):
        baseline(np.zeros((2, 5)))


def test_historical_small_scope_cannot_bypass_full_prior_requirement():
    manifest = {'format': 'random_sign_response_operational_v1'}
    source = {'area_deg2': .01}
    validate_scope(manifest, source, {}, True)
    with pytest.raises(ValueError, match='full-area'):
        validate_scope(manifest, source, {}, False)
    with pytest.raises(ValueError, match='small operational'):
        validate_scope(manifest, {'area_deg2': 1.}, {}, True)
