"""The complement diagnostic must score and sample exactly what it claims.

Its whole conclusion is a comparison of importance ratios ``c_j / q_j`` between
two proposals, so two properties decide whether the numbers mean anything: the
whole-catalogue proxy has to be the same score the production reranker uses,
and the tilted sampler has to draw atoms with the probability it reports.  A
mismatch in either would not raise -- it would quietly bias the diagnostic.
"""

from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from sbsi.catalogue_sampling import (
    TAIL_PROPOSAL_RECIPE,
    ProposalCandidates,
    ProposalDraw,
    WholeCatalogueProxy,
)
from sbsi.complement_diagnostic import (
    ComplementDiagnostic,
    _fixed_component_standard_error,
    _membership,
    diagnose_complement,
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
    proxy = WholeCatalogueProxy(proposal, torch.device("cpu"))
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
    proxy = WholeCatalogueProxy(proposal, torch.device("cpu"))
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


def test_tilted_sampler_conditions_on_an_exact_stratum(proposal):
    proxy = WholeCatalogueProxy(proposal, torch.device("cpu"))
    observation = np.random.default_rng(1).normal(size=4)
    delta = 0.1
    excluded = np.array([3, 17, 101, 402, 777])
    indices, probability = proxy.draw(
        observation,
        n_draws=200_000,
        delta=delta,
        rng=np.random.default_rng(2),
        exclude_indices=excluded,
    )

    scored = proxy.score(observation).numpy()
    finite = np.isfinite(scored)
    tilted = np.zeros(len(scored))
    tilted[finite] = np.exp(scored[finite] - scored[finite].max())
    tilted /= tilted.sum()
    conditional = (1.0 - delta) * tilted + delta * proposal.prior_weights
    conditional[excluded] = 0.0
    conditional /= conditional.sum()

    assert not np.isin(indices, excluded).any()
    assert np.allclose(probability, conditional[indices])
    frequency = np.bincount(indices, minlength=len(conditional)) / len(indices)
    assert np.abs(frequency - conditional).max() / conditional.max() < 0.05


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


def test_fixed_component_standard_error_preserves_paired_groups():
    counts = 2 * np.asarray(TAIL_PROPOSAL_RECIPE.component_counts)
    component_id = np.repeat(np.arange(len(counts)), counts)
    values = np.arange(1.0, len(component_id) + 1.0) ** 1.5
    expected_variance = sum(
        len(selected) * selected.var(ddof=1)
        for component in range(len(counts))
        for selected in [values[component_id == component]]
    ) / len(values) ** 2
    assert _fixed_component_standard_error(values, component_id) == pytest.approx(
        np.sqrt(expected_variance)
    )
    assert np.isnan(
        _fixed_component_standard_error(values[:-2], component_id[:-2])
    )


def test_summary_reduces_the_mass_shares():
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
    assert summary["pareto_k_tail_proposal_complement_percentiles"] is None
    assert summary["tail_proposal_component_complement_mass_percentiles"] is None


class _DiagnosticProposal:
    def __init__(self):
        self.prior_weights = np.array([0.1, 0.2, 0.3, 0.4])
        self.stratified_calls = []
        self.tail_calls = []

    def candidates(
        self,
        observed,
        *,
        n_candidates,
        prefilter_candidates=None,
        torch_device=None,
    ):
        del prefilter_candidates, torch_device
        indices = np.broadcast_to(
            np.arange(n_candidates, dtype=np.int64),
            (len(observed), n_candidates),
        ).copy()
        return ProposalCandidates(
            indices=indices,
            distances=np.zeros_like(indices, dtype=np.float64),
            radius=np.zeros(len(observed), dtype=np.float64),
        )

    def draw_stratified(self, candidates, *, n_draws, seed, object_ids):
        self.stratified_calls.append((int(seed), object_ids.copy()))
        pattern = np.resize(np.array([0, 1, 2, 3], dtype=np.int64), n_draws)
        indices = np.broadcast_to(pattern, (len(object_ids), n_draws)).copy()
        local_member = indices == 0
        local_position = np.where(local_member, 0, -1).astype(np.int32)
        return ProposalDraw(
            indices=indices,
            probability=self.prior_weights[indices],
            local_member=local_member,
            global_component=np.ones_like(local_member),
            candidate_radius=candidates.radius,
            seed=seed,
            local_position=local_position,
        )

    def draw_tilted_mis(
        self,
        candidates,
        observed,
        *,
        n_draws,
        seed,
        object_ids,
        device,
        atom_chunk=262144,
        object_chunk=32,
    ):
        del observed, device, atom_chunk, object_chunk
        self.tail_calls.append((int(seed), object_ids.copy()))
        schedule = np.resize(
            np.asarray(TAIL_PROPOSAL_RECIPE.component_cycle, dtype=np.int16),
            n_draws,
        )
        indices = np.broadcast_to(
            np.resize(np.array([1, 2, 3], dtype=np.int64), n_draws),
            (len(object_ids), n_draws),
        ).copy()
        probability = np.full(indices.shape, 1.0 / 3.0)
        component_id = np.broadcast_to(schedule, indices.shape).copy()
        n_components = len(TAIL_PROPOSAL_RECIPE.component_names)
        complement_mass = np.broadcast_to(
            np.linspace(0.20, 0.80, n_components),
            (len(object_ids), n_components),
        ).copy()
        complement_mass[:, :-1] += object_ids[:, None] / 1_000_000.0
        complement_mass[:, -1] = 0.90
        prior_component = n_components - 1
        return ProposalDraw(
            indices=indices,
            probability=probability,
            local_member=np.zeros_like(indices, dtype=bool),
            global_component=component_id == prior_component,
            candidate_radius=candidates.radius,
            seed=seed,
            local_position=np.full(indices.shape, -1, dtype=np.int32),
            component_id=component_id,
            component_complement_mass=complement_mass,
        )


class _DiagnosticLikelihood:
    flow_model = SimpleNamespace(device=torch.device("cpu"))
    atom_mass = np.array([0.4, 0.3, 0.2, 0.1])

    def observed_target_tensor(self, observed):
        return torch.zeros((len(observed), 1), dtype=torch.float64)

    def log_importance_weights_tensor(
        self,
        observed,
        g1,
        g2,
        *,
        atom_indices,
        proposal_probability,
        **kwargs,
    ):
        del observed, g1, g2, kwargs
        ratio = self.atom_mass[np.asarray(atom_indices)] / np.asarray(
            proposal_probability
        )
        return torch.as_tensor(np.log(ratio), dtype=torch.float64)


def test_partition_uses_absolute_ids_once_and_reports_production_components():
    proposal = _DiagnosticProposal()
    mock = SimpleNamespace(
        measurements=pd.DataFrame({"observed": [10.0, 11.0]})
    )
    diagnostic = diagnose_complement(
        _DiagnosticLikelihood(),
        mock,
        proposal,
        center=(0.0, 0.0),
        n_candidates=1,
        prefilter_candidates=3,
        n_global_draws=16,
        seed=73,
        observation_start=0,
        observation_stop=2,
        object_id_offset=25_000,
        object_chunk=1,
        candidate_backend="scipy",
        include_tail_proposal=True,
    )

    assert diagnostic.table["object_index"].tolist() == [25_000, 25_001]
    assert [ids.tolist() for _, ids in proposal.stratified_calls] == [
        [25_000],
        [25_001],
    ]
    assert [ids.tolist() for _, ids in proposal.tail_calls] == [
        [25_000],
        [25_001],
    ]
    assert {seed for seed, _ in proposal.stratified_calls} == {73}
    assert {seed for seed, _ in proposal.tail_calls} == {73}
    assert diagnostic.metadata["observation_start"] == 25_000
    assert diagnostic.metadata["observation_stop"] == 25_002
    assert diagnostic.metadata["tail_proposal"] == asdict(TAIL_PROPOSAL_RECIPE)
    assert diagnostic.metadata["candidate_backend"] == "scipy"
    assert diagnostic.metadata["candidate_device"] is None
    assert diagnostic.metadata["flow_device"] == "cpu"
    assert diagnostic.metadata["object_chunk"] == 1
    assert diagnostic.metadata["flow_atom_chunk"] == 4096
    assert diagnostic.metadata["tail_proposal_device"] == "cpu"
    assert diagnostic.metadata["tail_proposal_atom_chunk"] == 262144
    assert diagnostic.metadata["tail_proposal_object_chunk"] == 32
    assert diagnostic.metadata["elapsed_seconds"] >= 0.0
    assert set(diagnostic.metadata["phase_seconds"]) == {
        "candidate_query",
        "prefilter_flow",
        "prior_draw",
        "prior_flow",
        "tail_proposal_draw",
        "tail_proposal_flow",
        "reduction",
    }
    for name in TAIL_PROPOSAL_RECIPE.component_names:
        assert f"tail_proposal_component_complement_mass_{name}" in diagnostic.table
    assert np.isfinite(
        diagnostic.table[
            "mass_complement_tail_minus_prior_standard_error"
        ]
    ).all()
    assert "mass_complement_tail_minus_prior_z" in diagnostic.table
    for _, row in diagnostic.table.iterrows():
        target = sum(
            row[f"tail_proposal_component_target_contribution_fraction_{name}"]
            for name in TAIL_PROPOSAL_RECIPE.component_names
        )
        variance = sum(
            row[f"tail_proposal_component_variance_fraction_{name}"]
            for name in TAIL_PROPOSAL_RECIPE.component_names
        )
        top = sum(
            row[f"tail_proposal_top_1_percent_source_fraction_{name}"]
            for name in TAIL_PROPOSAL_RECIPE.component_names
        )
        assert target == pytest.approx(1.0)
        assert variance == pytest.approx(1.0)
        assert top == pytest.approx(1.0)
    summary = diagnostic.summary()
    assert set(summary["tail_proposal_component_complement_mass_percentiles"]) == set(
        TAIL_PROPOSAL_RECIPE.component_names
    )
    assert summary["tail_proposal_relative_error_percentiles"] is not None
    assert summary["tail_proposal_minus_prior_z_mean"] is not None
    assert summary["tail_proposal_minus_prior_z_rms"] is not None
    assert summary["pareto_k_tail_proposal_complement_undefined"] == 2
    assert set(summary["tail_proposal_component_variance_fraction_percentiles"]) == set(
        TAIL_PROPOSAL_RECIPE.component_names
    )
