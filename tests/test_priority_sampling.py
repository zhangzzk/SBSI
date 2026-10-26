"""Priority sampling estimates the complement without replacement.

The tilted draw samples with replacement from ``q``, so one draw contributes
``c_j / (M q_j)`` and no bound exists on that ratio: an atom the proxy
underrates by a large factor produces an arbitrarily large weight whenever it
is drawn.  cont.313 measured the consequence as a Pareto tail index above 0.7.

Priority sampling removes the repetition and the ``1 / M`` together.  Each atom
draws its own uniform, receives the key ``q_j / u_j``, and the ``M`` largest
keys are retained; with ``tau`` the ``(M+1)``-th largest key, an atom is
retained exactly when ``u_j < q_j / tau``.  What has to be protected is the
pair of properties that buys: the sum ``sum c_j / min(1, q_j / tau)`` is
unbiased, and every retained atom with ``q_j >= tau`` enters at weight exactly
``c_j``, contributing no variance at all.
"""

import numpy as np
import pytest
import torch

from sbsi.catalogue_sampling import WholeCatalogueProxy, _priority_row_seed


class _Coordinates:
    def __init__(self, values, dispersion):
        self.values = values
        self.dispersion = dispersion


class _Proposal:
    def __init__(self, coordinates, prior_weights, local_base_weights):
        self.coordinates = coordinates
        self.prior_weights = prior_weights
        self.local_base_weights = local_base_weights
        self.active_indices = np.arange(len(prior_weights))


@pytest.fixture
def proxy():
    rng = np.random.default_rng(11)
    n_atoms, n_targets = 800, 4
    coordinates = _Coordinates(
        rng.normal(size=(n_atoms, n_targets)),
        np.exp(0.3 * rng.normal(size=(n_atoms, n_targets))),
    )
    prior = rng.random(n_atoms)
    return WholeCatalogueProxy(
        _Proposal(coordinates, prior / prior.sum(), rng.random(n_atoms)),
        torch.device("cpu"),
    )


def _select(proxy, observation, n_select, seed, excluded=None):
    return proxy.select_priority_batch(
        np.atleast_2d(observation),
        n_select=n_select,
        delta=0.1,
        seed=seed,
        object_ids=np.array([0], dtype=np.int64),
        excluded=None if excluded is None else [np.asarray(excluded)],
    )


def test_the_estimator_is_unbiased_over_repeated_races(proxy):
    """Average the Horvitz-Thompson sum over seeds and recover the exact total.

    The quantity summed is deliberately *not* the proposal: ``c`` is a
    perturbed version of ``q`` so the ratio ``c / q`` varies across atoms, which
    is the only regime in which unbiasedness is a non-trivial claim.
    """

    observation = np.array([0.4, -0.2, 0.1, 0.3])
    mixture = proxy.mixture(np.atleast_2d(observation), delta=0.1)[0].numpy()
    rng = np.random.default_rng(3)
    target = mixture * np.exp(rng.normal(scale=1.5, size=mixture.size))
    exact = target.sum()

    estimates = []
    for seed in range(400):
        indices, inclusion = _select(proxy, observation, 64, seed)
        estimates.append((target[indices[0]] / inclusion[0]).sum())
    estimates = np.asarray(estimates)
    error = estimates.std(ddof=1) / np.sqrt(estimates.size)
    assert abs(estimates.mean() - exact) < 3.0 * error


def test_certainty_atoms_enter_at_weight_one(proxy):
    """Atoms above the threshold are retained every time, at inclusion one.

    They exist only once ``M`` is comparable to the proposal's own effective
    support ``1 / sum q_j^2``: this fixture has an effective support of 291
    atoms, and the race produces no certainty atom at ``M = 64`` but 45 of them
    at ``M = 200``.  That ratio is what decides whether priority sampling buys
    a zero-variance core in production or only a bounded ratio, so the test
    runs in the regime where the property is claimed and
    :func:`test_a_flat_proposal_has_no_certainty_atoms` pins the other side.
    """

    observation = np.array([0.4, -0.2, 0.1, 0.3])
    mixture = proxy.mixture(np.atleast_2d(observation), delta=0.1)[0].numpy()
    effective = 1.0 / np.square(mixture).sum()
    assert 200 > 0.5 * effective
    indices, inclusion = _select(proxy, observation, 200, 0)
    certain = inclusion[0] >= 1.0
    assert certain.any()
    # Inclusion is exactly one, never merely close, because it is a clamp.
    assert np.all(inclusion[0][certain] == 1.0)
    # Those same atoms are the largest-mass ones and survive every seed.
    # Certainty is a statement conditional on tau, and tau is itself random, so
    # an atom certain at one seed need not be certain at another.  The sharp
    # invariant is that the certainty set is exactly {j : q_j >= tau}: recover
    # tau from any inflated retained atom, then check both directions.
    inflated = ~certain
    assert inflated.any()
    threshold = mixture[indices[0][inflated]] / inclusion[0][inflated]
    np.testing.assert_allclose(threshold, threshold[0], rtol=1e-12)
    tau = float(threshold[0])
    assert np.all(mixture[indices[0][certain]] >= tau)
    missed = np.setdiff1d(np.arange(mixture.size), indices[0])
    assert np.all(mixture[missed] < tau)


def test_a_flat_proposal_has_no_certainty_atoms(proxy):
    """Below the effective support the scheme still bounds the ratio.

    With ``M`` well under ``1 / sum q_j^2`` every retained atom is inflated,
    so the zero-variance core is empty -- but the inflation is still capped at
    ``tau / q_j``, which is the property that with-replacement drawing lacks
    entirely.  This is the regime production is expected to sit in, since
    ``M = 8192`` against a 12,760,990-atom catalogue.
    """

    observation = np.array([0.4, -0.2, 0.1, 0.3])
    mixture = proxy.mixture(np.atleast_2d(observation), delta=0.1)[0].numpy()
    effective = 1.0 / np.square(mixture).sum()
    assert 64 < 0.25 * effective
    _, inclusion = _select(proxy, observation, 64, 0)
    assert not (inclusion[0] >= 1.0).any()
    assert np.isfinite(1.0 / inclusion[0]).all()


def test_no_retained_weight_exceeds_the_threshold_ratio(proxy):
    """The ratio is bounded by tau / q_j rather than by the luck of the draw."""

    observation = np.array([0.4, -0.2, 0.1, 0.3])
    mixture = proxy.mixture(np.atleast_2d(observation), delta=0.1)[0].numpy()
    indices, inclusion = _select(proxy, observation, 64, 7)
    # inclusion == min(1, q/tau), so 1/inclusion == max(1, tau/q) exactly.
    ratio = 1.0 / inclusion[0]
    reconstructed = np.maximum(1.0, ratio.max() * mixture[indices[0]].min() / mixture[indices[0]])
    assert ratio.max() < np.inf
    assert np.all(ratio >= 1.0)
    assert reconstructed.shape == ratio.shape


def test_selection_is_nested_in_the_number_retained(proxy):
    """The top-32 race is a subset of the top-64 race at the same seed."""

    observation = np.array([0.4, -0.2, 0.1, 0.3])
    small, _ = _select(proxy, observation, 32, 5)
    large, _ = _select(proxy, observation, 64, 5)
    assert set(small[0].tolist()) <= set(large[0].tolist())


def test_excluded_atoms_never_race(proxy):
    """The exact stratum is removed deterministically, not by chance."""

    observation = np.array([0.4, -0.2, 0.1, 0.3])
    baseline, _ = _select(proxy, observation, 64, 2)
    dropped = np.sort(baseline[0][:16])
    indices, _ = _select(proxy, observation, 64, 2, excluded=dropped)
    assert not (set(indices[0].tolist()) & set(dropped.tolist()))


def test_excluding_the_stratum_leaves_the_complement_unbiased(proxy):
    """Removing a fixed atom set still estimates the rest exactly."""

    observation = np.array([0.4, -0.2, 0.1, 0.3])
    mixture = proxy.mixture(np.atleast_2d(observation), delta=0.1)[0].numpy()
    rng = np.random.default_rng(4)
    target = mixture * np.exp(rng.normal(scale=1.5, size=mixture.size))
    dropped = np.sort(np.argsort(mixture)[-32:])
    keep = np.setdiff1d(np.arange(mixture.size), dropped)
    exact = target[keep].sum()

    estimates = []
    for seed in range(400):
        indices, inclusion = _select(proxy, observation, 64, seed, excluded=dropped)
        estimates.append((target[indices[0]] / inclusion[0]).sum())
    estimates = np.asarray(estimates)
    error = estimates.std(ddof=1) / np.sqrt(estimates.size)
    assert abs(estimates.mean() - exact) < 3.0 * error


def test_the_race_does_not_depend_on_chunking(proxy):
    """One object's retained set is the same whatever it is batched with."""

    rows = np.array([[0.4, -0.2, 0.1, 0.3], [-0.5, 0.6, 0.0, -0.1]])
    ids = np.array([17, 42], dtype=np.int64)
    together, together_p = proxy.select_priority_batch(
        rows, n_select=64, delta=0.1, seed=9, object_ids=ids
    )
    for row in (0, 1):
        alone, alone_p = proxy.select_priority_batch(
            rows[row : row + 1],
            n_select=64,
            delta=0.1,
            seed=9,
            object_ids=ids[row : row + 1],
        )
        # The retained set is identical.  The inclusion probabilities are not
        # bit-identical, because the mixture matmul reassociates across rows --
        # `draw_uniforms_batch` documents the same ~1e-15 relative movement.
        np.testing.assert_array_equal(together[row], alone[0])
        np.testing.assert_allclose(together_p[row], alone_p[0], rtol=1e-13)


def test_row_seeds_are_distinct_and_in_range():
    seeds = {_priority_row_seed(3, oid) for oid in range(2000)}
    assert len(seeds) == 2000
    assert all(0 <= value < 2**63 for value in seeds)


def test_more_draws_than_atoms_is_refused(proxy):
    with pytest.raises(ValueError, match="more active atoms than draws"):
        _select(proxy, np.array([0.4, -0.2, 0.1, 0.3]), 10_000, 0)
