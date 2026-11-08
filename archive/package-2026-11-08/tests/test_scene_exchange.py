"""Independent measure and reversibility checks for native scene exchanges."""
import numpy as np
from scipy.special import logsumexp

from sbsi.galsbi_scene_prior import GalSBISceneBatch
from sbsi.galsbi_target_coordinates import GalSBITargetCoordinates
from sbsi.scene_exchange import numerical_over_analytic, propose_native_exchange
from test_galsbi_prior import prior_fixture


class Choices:
    def __init__(self, values):
        self.values = iter(values)

    def choice(self, n, p):
        selected = next(self.values)
        assert 0 <= selected < n and p[selected] > 0
        return selected


def setup_scene():
    prior = prior_fixture()
    data, _ = prior.sample(32, 71)
    target = {k: v[:1].copy() for k, v in data.items() if k != 'proposal_index'}
    neighbour = {k: v[1:2].copy() for k, v in data.items() if k != 'proposal_index'}
    target['type'][:] = 0
    neighbour['type'][:] = 1
    return prior, GalSBISceneBatch(target, neighbour, [0, 1], [[.2, .4]])


def test_swap_marginal_type_and_coordinate_measure_detailed_balance():
    prior, scene = setup_scene()
    coords = GalSBITargetCoordinates(prior, lambda z: np.full_like(z, 1000.))
    x = coords.encode(scene.targets)
    lp, data, probability = coords.log_mixture_prior(x)
    scene.targets = data
    new, ratio, active = propose_native_exchange(prior, scene, probability, Choices([0, 0]))
    y = coords.encode(new.targets)
    lp_new, _, probability_new = coords.log_mixture_prior(y)
    old_j = coords.decode(x, np.array([0]))[1]
    new_j = coords.decode(y, np.array([1]))[1]
    p_env_old = prior.log_intensity(scene.neighbours)+numerical_over_analytic(prior, scene.neighbours)
    p_env_new = prior.log_intensity(new.neighbours)+numerical_over_analytic(prior, new.neighbours)
    # Deliberately nonconstant artificial likelihood. The chosen reverse type
    # is the old neighbour's type, not the marginalized placeholder type.
    likelihood_old, likelihood_new = -.3*(x[:, 1]-19)**2, -.3*(y[:, 1]-19)**2
    log_accept = ratio+likelihood_new-likelihood_old
    forward = lp+p_env_old+np.log(probability[:, 0])+likelihood_old+np.minimum(0, log_accept)
    reverse = (lp_new+p_env_new+np.log(probability_new[:, 1])+likelihood_new
               +np.minimum(0, -log_accept)+old_j-new_j)
    assert active.all()
    np.testing.assert_allclose(forward, reverse, atol=1e-11, rtol=0)
    restored, backratio, _ = propose_native_exchange(prior, new, probability_new, Choices([0, 1]))
    for key in scene.targets:
        np.testing.assert_array_equal(restored.targets[key], scene.targets[key])
        np.testing.assert_array_equal(restored.neighbours[key], scene.neighbours[key])
    np.testing.assert_allclose(ratio+backratio, 0, atol=1e-12)
    assert abs(float(old_j-new_j)) > .1  # This test needs a nontrivial Jacobian.


def test_numerical_density_matches_independent_inverse_map_derivative():
    prior = prior_fixture()
    data, _ = prior.sample(1000, 27)
    lp, factors = prior.log_intensity(data, return_factors=True)
    ratio = numerical_over_analytic(prior, data)
    assert np.isfinite(ratio).all()
    for kind, name in enumerate(prior.types):
        ix = np.flatnonzero(data['type'] == kind)[10:20]
        z, M = data['z'][ix], data['M_B'][ix]
        tab = prior.tables[name]
        ms, _, alpha = prior.luminosity_parameters(z, name)
        from sbsi.galsbi_prior import upper_gamma
        cap = np.minimum(prior.magnitude_cap(z), prior.p['lum_fct_m_max'])
        logmax = np.log(upper_gamma(alpha+1, 10**(.4*(ms-cap))))
        logu = np.interp(ms-M, tab['inverse_y'][::-1], tab['log_g'][::-1])-logmax
        u = np.exp(logu)
        uz = np.interp(z, tab['z'], tab['cdf'])
        h = 1e-8
        # Differentiate the actual forward draw maps, not the evaluator's formula.
        dzdu = (np.interp(uz+h, tab['cdf'], tab['z'])-np.interp(uz-h, tab['cdf'], tab['z']))/(2*h)
        dmdu = -(np.interp(logmax+np.log(u+h), tab['log_g'], tab['inverse_y'])
                 -np.interp(logmax+np.log(u-h), tab['log_g'], tab['inverse_y']))/(2*h)
        expected = np.log(tab['rate'])-np.log(dzdu)-np.log(np.abs(dmdu))
        np.testing.assert_allclose(factors['luminosity'][ix]+ratio[ix], expected, atol=2e-6, rtol=0)


def test_empty_environment_is_inactive_and_support_violations_are_rejected():
    prior, scene = setup_scene()
    empty = GalSBISceneBatch(scene.targets, {k: v[:0] for k, v in scene.neighbours.items()}, [0, 0], np.empty((0, 2)))
    _, ratio, active = propose_native_exchange(prior, empty, [[.5, .5]], Choices([]))
    assert not active.any() and ratio[0] == 0
    for key, value in [('M_B', 0.), ('n', 20.), ('z', 4.), ('a', 1.)]:
        bad = {k: v.copy() for k, v in scene.targets.items()}
        bad[key][:] = value
        assert numerical_over_analytic(prior, bad)[0] == -np.inf


def test_position_weighted_choice_cancels_with_multiple_neighbours():
    prior, scene = setup_scene()
    neighbours = {k: np.repeat(v, 2, axis=0) for k, v in scene.neighbours.items()}
    scene = GalSBISceneBatch(scene.targets, neighbours, [0, 2], [[.1, .1], [3., 4.]])
    _, _, active = propose_native_exchange(prior, scene, [[.5, .5]], Choices([1, 0]))
    assert active.all()
    # Selection is independent of changing marks: positions are preserved exactly.
    proposed, _, _ = propose_native_exchange(prior, scene, [[.5, .5]], Choices([0, 0]))
    np.testing.assert_array_equal(proposed.offset_arcsec, scene.offset_arcsec)
    np.testing.assert_array_equal(proposed.indptr, scene.indptr)
