"""Poisson measure, thinning, support and production crowding replay checks."""
import numpy as np
import pandas as pd
import pytest
from scipy.stats import poisson

from sbsi.galsbi_scene_prior import ARCSEC2_PER_SR, GalSBISceneBatch, GalSBIScenePrior, light_transfer_map
from sbsi.v39_inference import physical_close_crowding
from sbsi.pseudo_marginal import metropolis_accept


class UniformMarks:
    """Analytically soluble prior: pre-cut rate 2/arcsec², retain z < 1/2."""
    parent_rate = 2*ARCSEC2_PER_SR

    def sample(self, n, seed):
        z = np.random.default_rng(seed).random(n)
        keep = z < .5
        return dict(z=z[keep], m_r=20+z[keep], proposal_index=np.flatnonzero(keep)), {}

    def log_intensity(self, data):
        z = data['z']
        return np.where((z >= 0) & (z < .5), np.log(self.parent_rate), -np.inf)


def test_poisson_density_measure_and_count_changing_ratio():
    prior = GalSBIScenePrior(UniformMarks(), aperture_arcsec=1.)
    scenes = GalSBISceneBatch(dict(z=np.array([.1, .2])), dict(z=np.array([.1, .3])),
                             np.array([0, 0, 2]), np.array([[.1, 0], [-.2, .3]]))
    # Uniform mark density=2, sky density=1/arcsec². This independently
    # factors into normalized target, Poisson count, positions and marks.
    expected = np.log(2)+poisson.logpmf(scenes.counts, np.pi)+scenes.counts*np.log(2/np.pi)
    np.testing.assert_allclose(prior.log_prob(scenes, surface_density=ARCSEC2_PER_SR), expected)
    swapped = GalSBISceneBatch(scenes.targets, dict(z=scenes.neighbours['z'][::-1]),
                               scenes.indptr, scenes.offset_arcsec[::-1])
    np.testing.assert_array_equal(prior.log_kernel(swapped), prior.log_kernel(scenes))
    one = GalSBISceneBatch(dict(z=np.array([.2])), dict(z=np.array([.1])),
                          np.array([0, 1]), np.array([[.1, 0]]))
    # Adding the second point has ratio intensity/(N+1) = 2/2 = 1.
    assert prior.log_kernel(scenes)[1]-prior.log_kernel(one)[0] == pytest.approx(0)
    scenes.offset_arcsec[0] = [2, 0]
    assert np.isneginf(prior.log_kernel(scenes)[1])
    assert np.isfinite(prior.log_kernel(scenes)[0])


def test_poisson_thinning_has_correct_counts_positions_and_empty_scenes():
    prior = GalSBIScenePrior(UniformMarks(), aperture_arcsec=.5)
    scenes, report = prior.sample(6000, 39)
    expected = np.pi*.5**2  # post-cut intensity, not envelope intensity
    assert abs(scenes.counts.mean()-expected) < 5*np.sqrt(expected/6000)
    assert abs(scenes.counts.var()/scenes.counts.mean()-1) < 5*np.sqrt(2/5999)
    u = np.square(scenes.offset_arcsec).sum(1)/.5**2
    assert abs(u.mean()-.5) < 5/np.sqrt(12*len(u))
    assert np.all((scenes.neighbours['z'] >= 0) & (scenes.neighbours['z'] < .5))
    assert (scenes.counts == 0).any()
    assert report['neighbour_proposals'] > report['n_neighbours']
    assert np.isfinite(prior.log_prob(scenes, surface_density=ARCSEC2_PER_SR)).all()
    with pytest.raises(ValueError, match='at least 10'):
        prior.v39_crowding(scenes)
    empty = GalSBISceneBatch(dict(z=np.array([.2])), dict(z=np.array([])),
                            np.array([0, 0]), np.empty((0, 2)))
    assert np.isfinite(prior.log_kernel(empty)).all()
    with pytest.raises(ValueError, match='aligned'):
        GalSBISceneBatch(empty.targets, empty.neighbours, np.array([0, 1]), empty.offset_arcsec)


def test_supplied_targets_remain_fixed_while_environment_is_marginalized():
    prior = GalSBIScenePrior(UniformMarks(), aperture_arcsec=.5)
    targets = dict(z=np.full(4000, .12), m_r=np.full(4000, 19.))
    scenes, report = prior.sample_given_targets(targets, 37)
    for name in targets:
        np.testing.assert_array_equal(scenes.targets[name], targets[name])
    assert report['target_proposals'] == 0
    assert abs(scenes.counts.mean()-np.pi*.25) < 5*np.sqrt(np.pi*.25/4000)


def test_crowding_matches_existing_v39_with_all_neighbours_and_empty_patch():
    prior = GalSBIScenePrior(UniformMarks(), aperture_arcsec=12.)
    scene = GalSBISceneBatch(dict(z=np.zeros(2), m_r=np.array([20., 21.])),
                            dict(z=np.zeros(3), m_r=np.array([21., 19., 15.])),
                            np.array([0, 3, 3]), np.array([[.5, 0], [2., 0], [11., 0]]))
    frame = pd.DataFrame(dict(RA=np.r_[0, scene.offset_arcsec[:, 0]]/3600,
                              DEC=np.zeros(4), r=np.r_[20, scene.neighbours['m_r']]))
    actual = prior.v39_crowding(scene)
    np.testing.assert_allclose(actual[:1], physical_close_crowding(frame, np.array([0])), atol=1e-14)
    np.testing.assert_array_equal(actual[1], np.zeros(3))


def test_ragged_selection_and_rejections_preserve_entire_scenes_without_aliases():
    prior = GalSBIScenePrior(UniformMarks(), aperture_arcsec=.5)
    old, _ = prior.sample(20, 11)
    new, _ = prior.sample(20, 12)
    accept = np.arange(20) % 3 == 0
    result = old.retained(new, accept)
    for i in range(20):
        expected = (new if accept[i] else old).take(np.array([i]))
        actual = result.take(np.array([i]))
        for k in expected.targets:
            np.testing.assert_array_equal(actual.targets[k], expected.targets[k])
        for k in expected.neighbours:
            np.testing.assert_array_equal(actual.neighbours[k], expected.neighbours[k])
        np.testing.assert_array_equal(actual.offset_arcsec, expected.offset_arcsec)
    duplicated = old.take(np.array([3, 1, 3]))
    np.testing.assert_array_equal(duplicated.counts, old.counts[[3, 1, 3]])
    duplicated.targets['z'][0] = -1
    assert old.targets['z'][3] >= 0
    with pytest.raises(ValueError, match='integer'):
        old.take(np.array([True, False]))


@pytest.mark.parametrize('fraction', [0., .1, 1.])
def test_partial_refresh_preserves_poisson_prior_and_exchangeable_count_moments(fraction):
    prior = GalSBIScenePrior(UniformMarks(), aperture_arcsec=.5)
    old, _ = prior.sample(10000, 51)
    new = prior.refresh(old, fraction, 52)
    rate = np.pi*.25
    assert abs(new.counts.mean()-rate) < 5*np.sqrt(rate/len(new.counts))
    assert abs(new.counts.var()-rate) < .08
    assert abs(np.cov(old.counts, new.counts)[0, 1]-(1-fraction)*rate) < .08
    assert abs(np.mean(new.neighbours['z'])-.25) < .01
    assert abs(np.mean(np.square(new.offset_arcsec).sum(1)/.25)-.5) < .02
    np.testing.assert_array_equal(old.targets['z'], new.targets['z'])
    if fraction == 0:
        np.testing.assert_array_equal(old.offset_arcsec, new.offset_arcsec)
        assert not np.shares_memory(old.offset_arcsec, new.offset_arcsec)


def test_likelihood_only_acceptance_recovers_analytic_tilted_poisson_and_marks():
    # L(scene)=exp(gamma*N + beta*sum(z)) tilts a Poisson process to another
    # Poisson process with known rate and exponentially tilted marks. This
    # exercises variable counts, partial refresh, MH and retained rejections.
    prior = GalSBIScenePrior(UniformMarks(), aperture_arcsec=.5)
    scenes, _ = prior.sample(2048, 73)
    beta, gamma = 2., .3

    def loglike(scene):
        return gamma*scene.counts + beta*np.bincount(scene.owner,
            weights=scene.neighbours['z'], minlength=len(scene.counts))

    rng = np.random.default_rng(74)
    for _ in range(200):
        proposed = prior.refresh(scenes, .4, int(rng.integers(0, 2**63)))
        accept = metropolis_accept(loglike(scenes), loglike(proposed), rng.uniform(1e-15, 1, 2048))
        scenes = scenes.retained(proposed, accept)
    rate = np.pi*.25*np.exp(gamma)*np.expm1(.5*beta)/(.5*beta)
    expected_z = .5*np.exp(.5*beta)/np.expm1(.5*beta)-1/beta
    assert abs(scenes.counts.mean()-rate) < 5*np.sqrt(rate/2048)
    assert abs(scenes.neighbours['z'].mean()-expected_z) < .012
    assert abs(np.mean(np.square(scenes.offset_arcsec).sum(1)/.25)-.5) < .025


def test_joint_target_and_full_environment_move_recovers_coupled_posterior():
    # Independent uniform target and Poisson environment, coupled by a
    # likelihood. Quadrature supplies an oracle, including target/count
    # cross-moments. The independent environment proposal cancels its prior;
    # the target support still belongs in the MH ratio.
    from scipy.integrate import quad
    prior = GalSBIScenePrior(UniformMarks(), aperture_arcsec=.5)
    rng = np.random.default_rng(845)
    nchain = 256
    scenes, _ = prior.sample(nchain, 846)
    rate = np.pi*.25
    sigma = .08
    def log_target_likelihood(scene):
        z = scene.targets['z']
        value = -.5*((z-.08-.1*scene.counts)/sigma)**2
        return np.where((z >= 0) & (z < .5), value, -np.inf)
    expected = np.zeros(5)
    for count in range(20):
        def integrand(z, power):
            return z**power*np.exp(-.5*((z-.08-.1*count)/sigma)**2)
        terms = np.array([quad(integrand, 0, .5, args=(k,))[0] for k in range(3)])
        expected += poisson.pmf(count, rate)*np.array([terms[0],terms[1],terms[2],count*terms[0],count*terms[1]])
    expected = expected[1:]/expected[0]
    moments = []
    for step in range(1800):
        z = scenes.targets['z']+rng.normal(0, .18, nchain)
        targets = dict(z=z, m_r=20+z)
        proposed = GalSBISceneBatch(targets, scenes.neighbours, scenes.indptr, scenes.offset_arcsec)
        proposed = prior.refresh(proposed, 1., int(rng.integers(0, 2**63)))
        accepted = metropolis_accept(log_target_likelihood(scenes), log_target_likelihood(proposed),
                                     rng.uniform(1e-15, 1, nchain))
        scenes = scenes.retained(proposed, accepted)
        if step >= 300:
            z = scenes.targets['z']
            moments.append(np.column_stack((z,z*z,scenes.counts,z*scenes.counts)))
    means = np.array(moments).mean(0)
    assert np.all(np.abs(means.mean(0)-expected) < 6*means.std(0,ddof=1)/np.sqrt(nchain))


def test_empty_environment_uses_generator_fields_not_supplied_target_fields():
    prior = GalSBIScenePrior(UniformMarks(), aperture_arcsec=1e-8)
    scenes, _ = prior.sample_given_targets(dict(z=np.array([.1]), extra=np.array([4.])), 31)
    assert scenes.indptr[-1] == 0
    assert set(scenes.neighbours) == {'z', 'm_r'}
    refreshed = prior.refresh(scenes, .1, 32)
    assert set(refreshed.targets) == {'z', 'extra'}
    with pytest.raises(ValueError, match='fraction'):
        prior.refresh(scenes, 1.1, 1)


def test_local_positions_preserve_uniform_disk_and_retain_marks_counts():
    prior = GalSBIScenePrior(UniformMarks(), aperture_arcsec=1.)
    scenes, _ = prior.sample(6000, 811)
    moved = prior.move_positions(scenes, 1.5, 812)
    np.testing.assert_array_equal(scenes.counts, moved.counts)
    np.testing.assert_array_equal(scenes.neighbours['z'], moved.neighbours['z'])
    u = np.square(moved.offset_arcsec).sum(1)
    assert (u <= 1).all()
    assert abs(u.mean()-.5) < 5/np.sqrt(12*len(u))
    assert np.mean(np.all(moved.offset_arcsec == scenes.offset_arcsec, axis=1)) > .4


def test_local_position_mh_recovers_nonuniform_disk_likelihood():
    from scipy.integrate import quad
    rng = np.random.default_rng(819)
    n, kappa = 2048, 2.
    radius, phase = np.sqrt(rng.random(n)), rng.uniform(0, 2*np.pi, n)
    scenes = GalSBISceneBatch(dict(z=np.full(n, .1)), dict(z=np.full(n, .2)),
        np.arange(n+1), radius[:, None]*np.column_stack((np.cos(phase), np.sin(phase))))
    prior = GalSBIScenePrior(UniformMarks(), aperture_arcsec=1.)
    normalizer = quad(lambda x: 2*np.sqrt(1-x*x)*np.exp(kappa*x), -1, 1)[0]
    expected_x = quad(lambda x: x*2*np.sqrt(1-x*x)*np.exp(kappa*x), -1, 1)[0]/normalizer
    for _ in range(350):
        proposed = prior.move_positions(scenes, .3, int(rng.integers(0, 2**63)))
        accepted = metropolis_accept(kappa*scenes.offset_arcsec[:, 0], kappa*proposed.offset_arcsec[:, 0],
                                     rng.uniform(1e-15, 1, n))
        scenes = scenes.retained(proposed, accepted)
    values = scenes.offset_arcsec
    assert abs(values[:, 0].mean()-expected_x) < 5*values[:, 0].std(ddof=1)/np.sqrt(n)
    assert abs(values[:, 1].mean()) < 5*values[:, 1].std(ddof=1)/np.sqrt(n)


class MagnitudeMarks:
    """Truncated-normal M_B marks with a fixed z-dependent m_r offset."""

    def photometry(self, z, M_B, c):
        m_r = M_B+20+z
        return dict(m_r=m_r, m_i=m_r-.3, M_i=M_B-.3)

    def log_intensity(self, data):
        M = np.asarray(data['M_B'])
        return np.where(M < 3.5, -.5*(M-1)**2, -np.inf)


def light_scenes(n, rng, offsets=((1.5, 0.), (0., 6.))):
    from scipy.stats import truncnorm
    k = len(offsets)
    z = rng.random(n*k)
    M = truncnorm.rvs(-np.inf, 2.5, loc=1, scale=1, size=n*k, random_state=rng)
    neighbours = dict(z=z, M_B=M, c=np.full((n*k, 5), .2)) | MagnitudeMarks().photometry(z, M, None)
    scenes = GalSBISceneBatch(dict(z=np.full(n, .1)), neighbours, np.arange(n+1)*k,
                              np.tile(np.asarray(offsets, dtype=float), (n, 1)))
    return scenes, rng.normal(20, .5, n)


def test_light_transfer_map_is_an_involution_with_flux_jacobian():
    m_t, m_t_new, m_n, w = np.array([20., 18.3]), np.array([19.9, 18.5]), np.array([21., 18.6]), np.array([.4, .3])
    forward, log_jacobian = light_transfer_map(m_t, m_t_new, m_n, w)
    back, back_log_jacobian = light_transfer_map(m_t_new, m_t, forward, w)
    np.testing.assert_allclose(back, m_n, atol=1e-12)
    np.testing.assert_allclose(log_jacobian+back_log_jacobian, 0, atol=1e-12)
    h = 1e-6
    derivative = (light_transfer_map(m_t, m_t_new, m_n+h, w)[0]-light_transfer_map(m_t, m_t_new, m_n-h, w)[0])/(2*h)
    np.testing.assert_allclose(np.log(derivative), log_jacobian, atol=1e-7)
    # Brightening the target by more than the neighbour can give is invalid.
    assert np.isnan(light_transfer_map([20.], [19.], [25.], [1.])[0]).all()


def test_light_transfer_moves_one_near_neighbour_and_leaves_others_alone():
    prior = GalSBIScenePrior(MagnitudeMarks(), aperture_arcsec=10.)
    scenes, m_t = light_scenes(2000, np.random.default_rng(5), offsets=((1.5, 0.), (3., 0.), (0., 6.)))
    empty = GalSBISceneBatch(dict(z=np.array([.1])), {k: v[:0] for k, v in scenes.neighbours.items()},
                             np.array([0, 0]), np.empty((0, 2)))
    new, m_new, correction, active = prior.transfer_light(scenes, m_t, 6)
    assert .5 < active.mean() < 1
    changed = new.neighbours['M_B'] != scenes.neighbours['M_B']
    np.testing.assert_array_equal(np.bincount(scenes.owner[changed], minlength=2000), active)
    assert not changed[2::3].any() and changed[0::3].any() and changed[1::3].any()
    np.testing.assert_array_equal(new.neighbours['m_r'], new.neighbours['M_B']+20+new.neighbours['z'])
    np.testing.assert_array_equal(m_new[~active], m_t[~active])
    assert (correction[~active] == 0).all() and (m_new[active] != m_t[active]).all()
    assert new.targets is scenes.targets
    empty_new, empty_m, _, empty_active = prior.transfer_light(empty, np.array([20.]), 7)
    assert not empty_active.any() and empty_m[0] == 20 and empty_new.indptr[-1] == 0


@pytest.mark.parametrize('jacobian', [True, False])
def test_light_transfer_mh_preserves_joint_target_and_neighbour_prior(jacobian):
    prior = GalSBIScenePrior(MagnitudeMarks(), aperture_arcsec=10.)
    rng = np.random.default_rng(91)
    n = 100000
    scenes, m_t = light_scenes(n, rng)
    accepted_total = 0
    for _ in range(3):
        new, m_new, correction, active = prior.transfer_light(scenes, m_t, int(rng.integers(0, 2**63)))
        if not jacobian:
            shift = np.bincount(scenes.owner, weights=new.neighbours['m_r']-scenes.neighbours['m_r'], minlength=n)
            correction = correction-.4*np.log(10)*shift
        accepted = metropolis_accept(-.5*((m_t-20)/.5)**2,
            np.where(active, -.5*((m_new-20)/.5)**2+correction, -np.inf), rng.uniform(1e-15, 1, n))
        accepted_total += accepted.mean()
        scenes = scenes.retained(new, accepted)
        m_t = np.where(accepted, m_new, m_t)
    near, far = scenes.neighbours['M_B'][0::2], scenes.neighbours['M_B'][1::2]
    from scipy.stats import truncnorm
    mean, var = truncnorm.stats(-np.inf, 2.5, loc=1, scale=1, moments='mv')
    z = np.array([(m_t.mean()-20)/(.5/np.sqrt(n)), (near.mean()-mean)/np.sqrt(var/n),
                  ((m_t-20)**2).mean()/.25-1, np.corrcoef(m_t, near)[0, 1]*np.sqrt(n)])
    z[2] /= np.sqrt(2/n)
    assert accepted_total/3 > .2
    np.testing.assert_array_equal(far, light_scenes(n, np.random.default_rng(91))[0].neighbours['M_B'][1::2])
    if jacobian:
        assert np.all(np.abs(z) < 5), z
    else:
        assert np.max(np.abs(z)) > 10, z
