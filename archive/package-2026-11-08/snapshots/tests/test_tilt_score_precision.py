"""The whole-catalogue proxy may compute its inner products in float32.

That score is 78% of the estimator's wall clock and the A40 runs float64 at
1/64 of its float32 rate, so the precision of two matrix products is the
dominant cost knob in the tilted proposal.  Lowering it is only defensible
because ``q`` enters the importance weight as a ratio: any strictly positive
``q`` used consistently to draw and to weight is a different valid proposal,
not a less accurate answer.  What must therefore be protected is not that the
two dtypes agree bit for bit, but that the float32 path still returns a
normalised float64 mixture that tracks the float64 path closely enough to be
the same proposal in practice.
"""

import numpy as np
import pytest
import torch

from sbsi.catalogue_sampling import WholeCatalogueProxy


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
def proposal():
    rng = np.random.default_rng(0)
    n_atoms, n_targets = 2000, 4
    coordinates = _Coordinates(
        rng.normal(size=(n_atoms, n_targets)),
        np.exp(0.3 * rng.normal(size=(n_atoms, n_targets))),
    )
    base = rng.random(n_atoms)
    # Atoms with no detection probability carry no target mass; the proxy drops
    # them to -inf and the softmax must still be finite.
    base[:10] = 0.0
    prior = rng.random(n_atoms)
    return _Proposal(coordinates, prior / prior.sum(), base)


def _proxies(proposal):
    device = torch.device("cpu")
    return (
        WholeCatalogueProxy(proposal, device, score_dtype=torch.float64),
        WholeCatalogueProxy(proposal, device, score_dtype=torch.float32),
    )


def test_float32_keeps_the_mixture_in_float64_and_normalised(proposal):
    wide, narrow = _proxies(proposal)
    observations = np.random.default_rng(1).normal(size=(3, 4))
    for proxy in (wide, narrow):
        mixture = proxy.mixture(observations, delta=0.1)
        assert mixture.dtype is torch.float64
        assert torch.isfinite(mixture).all()
        assert (mixture >= 0).all()
        assert np.allclose(mixture.sum(dim=1).numpy(), 1.0, atol=1e-12)


def test_float32_tracks_float64_closely_enough_to_be_the_same_proposal(proposal):
    wide, narrow = _proxies(proposal)
    observations = np.random.default_rng(1).normal(size=(3, 4))
    reference = wide.mixture(observations, delta=0.1).numpy()
    lowered = narrow.mixture(observations, delta=0.1).numpy()
    # The defensive floor makes every entry strictly positive, so a relative
    # comparison is well defined everywhere.
    relative = np.abs(lowered - reference) / reference
    assert relative.max() < 1e-3


def test_float32_does_not_change_which_atoms_lead(proposal):
    wide, narrow = _proxies(proposal)
    observation = np.random.default_rng(2).normal(size=4)
    top_wide = wide.top_atoms(observation, 50, delta=0.1)
    top_narrow = narrow.top_atoms(observation, 50, delta=0.1)
    assert set(top_wide.tolist()) == set(top_narrow.tolist())


def test_score_carries_the_same_dtype_contract(proposal):
    wide, narrow = _proxies(proposal)
    observation = np.random.default_rng(3).normal(size=4)
    assert wide.score(observation).dtype is torch.float64
    assert narrow.score(observation).dtype is torch.float64
    assert narrow.a1.dtype is torch.float32
    assert narrow.constant.dtype is torch.float64


def test_an_unsupported_precision_is_refused(proposal):
    with pytest.raises(ValueError, match="score_dtype"):
        WholeCatalogueProxy(proposal, torch.device("cpu"), score_dtype=torch.float16)


def test_float32_error_grows_with_the_cancellation_magnitude():
    """cont.340: the expanded score is a large difference of large numbers.

    ``s_j = c_j - 0.5 (x^2 . a1_j - 2 x . a2_j)`` is algebraically
    ``-0.5 sum((x-mu)^2/sigma^2)``, O(1) for an atom near the observation, but
    the expansion reaches it through terms of size ``0.5 sum(mu^2/sigma^2)``.
    On the production cache that term has median 6.1e3 and 99th percentile
    3.9e5, because ``measured_mag_auto`` runs 17-27 while dispersions reach
    0.022.  float32's ~1.2e-7 relative error on it lands in the *exponent* of
    the softmax.

    The error is therefore proportional to that magnitude, which scales as
    1/sigma^2.  Shrinking sigma tenfold must worsen the float32 deviation by
    about a hundredfold; anything close to a flat ratio would mean the
    mechanism is not cancellation and this rejection needs revisiting.
    Measured consequence on the production cache: the g1 error bar degraded 32%
    while the phase got only 11.5% faster, so float32 is rejected.
    """

    rng = np.random.default_rng(7)
    n_atoms = 4000
    values = np.column_stack([
        rng.normal(scale=0.3, size=n_atoms),
        rng.normal(scale=0.3, size=n_atoms),
        rng.uniform(17.0, 27.0, size=n_atoms),
        rng.normal(loc=1.6, scale=0.4, size=n_atoms),
    ])
    observation = np.array([0.0, 0.0, 22.0, 1.6])
    device = torch.device("cpu")
    base = rng.random(n_atoms)
    prior = np.full(n_atoms, 1.0 / n_atoms)

    def deviation(sigma):
        proposal = _Proposal(
            _Coordinates(values, np.full_like(values, sigma)), prior, base
        )
        wide = WholeCatalogueProxy(proposal, device, score_dtype=torch.float64)
        narrow = WholeCatalogueProxy(proposal, device, score_dtype=torch.float32)
        reference = wide.mixture(observation, delta=0.1).numpy()
        lowered = narrow.mixture(observation, delta=0.1).numpy()
        magnitude = 0.5 * (np.square(values) / sigma**2).sum(axis=1)
        return np.abs(lowered - reference).max() / reference.max(), magnitude.max()

    benign, benign_magnitude = deviation(0.5)
    severe, severe_magnitude = deviation(0.05)

    assert severe_magnitude > 1e5, "fixture must reach the production regime"
    assert severe_magnitude / benign_magnitude == pytest.approx(100.0, rel=1e-6)
    # Linear in the magnitude, so a hundredfold magnitude means a hundredfold
    # error.  Assert an order of magnitude, well inside that prediction.
    assert severe / benign > 10.0
