"""The complement diagnostic must score and sample exactly what it claims.

Its whole conclusion is a comparison of importance ratios ``c_j / q_j`` between
two proposals, so two properties decide whether the numbers mean anything: the
whole-catalogue proxy has to be the same score the production reranker uses,
and the tilted sampler has to draw atoms with the probability it reports.  A
mismatch in either would not raise -- it would quietly bias the diagnostic.
"""

import numpy as np
import pytest
import torch

from sbsi.complement_diagnostic import (
    ComplementDiagnostic,
    _WholeCatalogueProxy,
    _membership,
    tail_of,
)


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
    n_atoms, n_targets = 1000, 4
    coordinates = _Coordinates(
        rng.normal(size=(n_atoms, n_targets)),
        np.exp(0.3 * rng.normal(size=(n_atoms, n_targets))),
    )
    base = rng.random(n_atoms)
    # Atoms with no detection probability carry no target mass; the proxy must
    # drop them without producing NaN.
    base[:10] = 0.0
    prior = rng.random(n_atoms)
    return _Proposal(coordinates, prior / prior.sum(), base)


def test_proxy_reproduces_the_reranker_score(proposal):
    proxy = _WholeCatalogueProxy(proposal, torch.device("cpu"))
    observation = np.random.default_rng(1).normal(size=4)
    scored = proxy.score(observation).numpy()

    values = proposal.coordinates.values
    dispersion = proposal.coordinates.dispersion
    with np.errstate(divide="ignore"):
        expected = (
            np.log(proposal.local_base_weights)
            - np.log(dispersion).sum(axis=1)
            - 0.5 * np.square((observation - values) / dispersion).sum(axis=1)
        )
    finite = np.isfinite(expected)
    assert np.allclose(scored[finite], expected[finite], atol=1e-9)
    assert np.isneginf(scored[~finite]).all()


def test_tilted_sampler_reports_its_own_probability(proposal):
    proxy = _WholeCatalogueProxy(proposal, torch.device("cpu"))
    observation = np.random.default_rng(1).normal(size=4)
    delta = 0.1
    indices, probability = proxy.draw(
        observation,
        n_draws=200_000,
        delta=delta,
        rng=np.random.default_rng(2),
    )

    scored = proxy.score(observation).numpy()
    finite = np.isfinite(scored)
    tilted = np.zeros(len(scored))
    tilted[finite] = np.exp(scored[finite] - scored[finite].max())
    tilted /= tilted.sum()
    mixture = (1.0 - delta) * tilted + delta * proposal.prior_weights

    # The reported probability is what divides the target, so it must be the
    # true mixture probability, not the tilted component alone.
    assert np.allclose(probability, mixture[indices])
    frequency = np.bincount(indices, minlength=len(mixture)) / len(indices)
    assert np.abs(frequency - mixture).max() / mixture.max() < 0.05


def test_membership_restores_its_lookup_table():
    lookup = np.zeros(10, dtype=bool)
    member = _membership(lookup, np.array([1, 3, 5]), np.array([0, 1, 2, 3, 9]))
    assert member.tolist() == [False, True, False, True, False]
    assert not lookup.any()


def test_tail_index_is_undefined_rather_than_guessed():
    assert np.isnan(tail_of(np.arange(5.0))[0])
    index, maximum = tail_of(np.random.default_rng(3).random(1000))
    assert index < 0.0
    assert maximum <= 1.0


def test_summary_reduces_the_mass_shares():
    import pandas as pd

    frame = pd.DataFrame(
        [
            {
                "mass_stratum": 1.0,
                "mass_prefilter_complement": 0.5,
                "mass_outside_prefilter": 0.5,
                "mass_outside_prefilter_standard_error": 0.05,
                "pareto_k_complement": 0.6,
                "pareto_k_prefilter_complement": 0.9,
                "pareto_k_outside_prefilter": 0.3,
            }
        ]
    )
    summary = ComplementDiagnostic(frame, {"flow_evaluations": 10}).summary()
    assert summary["share_stratum_percentiles"][1] == pytest.approx(0.5)
    assert summary["share_outside_prefilter_percentiles"][1] == pytest.approx(0.25)
    assert summary["pareto_k_tilted_complement_percentiles"] is None


def test_summary_refuses_to_report_a_requested_measurement_as_absent():
    """A silent null is how a dead code path hides.

    The tilted and exact-top columns are optional, so their absence reads as
    "not requested".  When they WERE requested, absence means the branch never
    ran, and three GPU jobs once returned nulls for exactly that reason.
    """

    import pandas as pd

    frame = pd.DataFrame(
        [
            {
                "mass_stratum": 1.0,
                "mass_prefilter_complement": 0.5,
                "mass_outside_prefilter": 0.5,
                "mass_outside_prefilter_standard_error": 0.05,
                "pareto_k_complement": 0.6,
                "pareto_k_prefilter_complement": 0.9,
                "pareto_k_outside_prefilter": 0.3,
            }
        ]
    )
    # Nothing requested: the optional columns are legitimately absent.
    ComplementDiagnostic(frame, {"flow_evaluations": 10}).summary()

    with pytest.raises(ValueError, match="pareto_k_tilted_complement"):
        ComplementDiagnostic(
            frame, {"flow_evaluations": 10, "tilt_delta": 0.1}
        ).summary()

    # The tilted column present but the exact-top one missing: only the second
    # measurement failed to run, and only it must be objected to.
    with_tilt = frame.assign(pareto_k_tilted_complement=0.2)
    with pytest.raises(ValueError, match="pareto_k_tilted_exact_top"):
        ComplementDiagnostic(
            with_tilt,
            {"flow_evaluations": 10, "tilt_delta": 0.1, "exact_top": 4096},
        ).summary()
