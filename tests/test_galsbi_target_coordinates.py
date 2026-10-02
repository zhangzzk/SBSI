"""Independent finite-difference Jacobian and physical-coordinate replay."""
import numpy as np
import pytest
from scipy.stats import dirichlet

from sbsi.galsbi_target_coordinates import GalSBITargetCoordinates
from scipy.special import logsumexp
from test_galsbi_prior import prior_fixture
from sbsi.galsbi_summary import native_shapes
from sbsi.pseudo_marginal import metropolis_accept, retained


def test_coordinate_roundtrip_and_full_native_jacobian():
    prior = prior_fixture()
    coord = GalSBITargetCoordinates(prior, lambda z: 1000*z/(1+z))
    data = dict(type=np.array([0]), z=np.array([.4]), M_B=np.array([-20.]),
                c=np.array([[.1, .2, .25, .3, .15]]), R_kpc=np.array([1.2]),
                n=np.array([1.5]), a=np.array([.3]), psi=np.array([1.1]))
    x = coord.encode(data)
    restored, logj = coord.decode(x, data['type'])
    for name in data:
        np.testing.assert_allclose(restored[name], data[name], rtol=1e-12, atol=1e-12)

    def native(values):
        d, _ = coord.decode(values, data['type'])
        return np.r_[d['z'], d['M_B'], d['c'][0, :4], d['R_kpc'], d['n'], d['a'], d['psi']]

    derivatives = []
    for col in range(10):
        h = np.zeros_like(x)
        h[0, col] = 1e-5
        derivatives.append((native(x+h)-native(x-h))/(2e-5))
    _, numeric = np.linalg.slogdet(np.column_stack(derivatives))
    np.testing.assert_allclose(logj[0], numeric, atol=1e-8)
    logp, _ = coord.log_prior(x, data['type'])
    np.testing.assert_allclose(logp, prior.log_intensity(data)+numeric, atol=1e-8)


def test_changing_redshift_preserves_apparent_magnitude_and_angular_size():
    prior = prior_fixture()
    coord = GalSBITargetCoordinates(prior, lambda z: 1000*z/(1+z))
    sample, _ = prior.sample(8, 64)
    x = coord.encode(sample)
    x[:, 0] += .1
    moved, _ = coord.decode(x, sample['type'])
    np.testing.assert_allclose(coord.encode(moved), x, atol=1e-12)


def test_marginal_type_prior_matches_explicit_sum_without_type_reweighting():
    prior = prior_fixture()
    coord = GalSBITargetCoordinates(prior, lambda z: 1000*z/(1+z))
    sample, _ = prior.sample(20, 45)
    x = coord.encode(sample)
    terms = np.column_stack([coord.log_prior(x, np.full(len(x), k))[0] for k in range(2)])
    total, data, probabilities = coord.log_mixture_prior(x)
    np.testing.assert_allclose(total, logsumexp(terms, axis=1), atol=1e-12)
    np.testing.assert_allclose(probabilities.sum(1), 1, atol=1e-12)
    for kind in range(2):
        native, _ = coord.decode(x, np.full(len(x), kind))
        for name in ('m_r', 'R_kpc', 'n', 'a', 'psi'):
            np.testing.assert_array_equal(data[name], native[name])


class SeparatedSED:
    types = ('red', 'blue')
    alphas = np.array([[25., 2., 2., 2., 2.], [2., 25., 2., 2., 2.]])

    def sed_alpha(self, z, kind):
        return np.broadcast_to(self.alphas[self.types.index(kind)], (len(z), 5))


def test_sed_proposal_density_matches_scipy_with_alr_jacobian():
    prior = SeparatedSED()
    coords = GalSBITargetCoordinates(prior, None)
    rng = np.random.default_rng(97)
    c = rng.dirichlet(np.ones(5), 100)
    x = np.zeros((len(c), 10))
    x[:, 2:6] = np.log(c[:, :4]/c[:, 4:])
    expected = logsumexp(np.column_stack([dirichlet.logpdf(c.T, a) for a in prior.alphas]), axis=1)
    expected += -np.log(2)+np.log(c).sum(1)
    np.testing.assert_allclose(coords.sed_log_proposal(x), expected, atol=1e-12)
    with pytest.raises(ValueError, match='ten-coordinate'):
        coords.propose_sed(x[:, :9], rng)


def test_sed_jump_recovers_unequal_mixture_and_likelihood_tilt():
    # Proposal weights are equal, but target weights are 80:20 and L(c)=c0.
    # Analytic posterior: mixture weights proportional to w*alpha0/sum(alpha),
    # each Dirichlet component updated by +1 in alpha0. Missing the Hastings
    # correction or its Jacobian gives the wrong mean and second moment.
    prior = SeparatedSED()
    coords = GalSBITargetCoordinates(prior, None)
    rng = np.random.default_rng(125)
    x = np.zeros((64, 10))
    weights = np.array([.8, .2])

    def fractions(x):
        logits = np.column_stack((x[:, 2:6], np.zeros(len(x))))
        return np.exp(logits-logsumexp(logits, axis=1, keepdims=True))

    def logtarget(x):
        c = fractions(x)
        terms = np.column_stack([dirichlet.logpdf(c.T, a) for a in prior.alphas])
        return logsumexp(terms+np.log(weights), axis=1)+np.log(c).sum(1)+np.log(c[:, 0])

    samples = []
    for step in range(800):
        proposal, correction = coords.propose_sed(x, rng)
        accept = metropolis_accept(logtarget(x), logtarget(proposal)+correction,
                                   rng.uniform(1e-15, 1, len(x)))
        x = retained(x, proposal, accept)
        if step >= 100:
            samples.append(fractions(x)[:, 0])
    draws = np.array(samples)
    posterior_weights = weights*prior.alphas[:, 0]/prior.alphas.sum(1)
    posterior_weights /= posterior_weights.sum()
    alpha = prior.alphas.copy()
    alpha[:, 0] += 1
    for power in (1, 2):
        moment = alpha[:, 0]/alpha.sum(1)
        if power == 2:
            moment *= (alpha[:, 0]+1)/(alpha.sum(1)+1)
        expected = posterior_weights@moment
        means = (draws**power).mean(0)
        assert abs(means.mean()-expected) < 6*means.std(ddof=1)/np.sqrt(len(means))
    assert (draws < .25).any() and (draws > .5).any()


def test_sed_jump_updates_native_luminosity_but_preserves_physical_features():
    prior = prior_fixture()

    def photo(z, M, c):
        return dict(m_r=M+40+3*c[:, 0], m_i=M+40+2*c[:, 1], M_i=M+2*c[:, 1])

    prior.photometry = photo
    coords = GalSBITargetCoordinates(prior, lambda z: 1000*z/(1+z))
    native, _ = prior.sample(100, 236)
    x = coords.encode(native)
    proposal, correction = coords.propose_sed(x, np.random.default_rng(237))
    moved, _ = coords.decode(proposal, native['type'])
    np.testing.assert_array_equal(x[:, [0, 1, 6, 7, 8, 9]], proposal[:, [0, 1, 6, 7, 8, 9]])
    assert np.isfinite(correction).all()
    assert not np.allclose(native['M_B'], moved['M_B'])
    np.testing.assert_allclose(native['m_r'], moved['m_r'], atol=1e-12)
    before = native_shapes(native, coords.angular_distance)
    after = native_shapes(moved, coords.angular_distance)
    for name in before:
        np.testing.assert_allclose(before[name], after[name], atol=1e-12)
    # A large change of the i-band colour must still trigger native cuts,
    # even though the likelihood-visible r magnitude stays fixed.
    prior.photometry = lambda z, M, c: dict(m_r=M+40+3*c[:, 0], m_i=M+100, M_i=M)
    logprior, _, _ = coords.log_mixture_prior(proposal)
    assert np.isneginf(logprior).all()
