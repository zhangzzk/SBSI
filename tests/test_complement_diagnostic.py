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
    _candidate_exclusion_manifest_sha256,
)
from sbsi.complement_diagnostic import (
    EXACT_ORACLE_ATOM_CHUNK,
    EXACT_ORACLE_CONFIRMATION,
    EXACT_ORACLE_LABEL,
    EXACT_ORACLE_OBJECT_CHUNK,
    EXACT_ORACLE_SELECTION_COUNT,
    EXACT_ORACLE_SELECTION_KEY_SHA256,
    ComplementDiagnostic,
    _fixed_component_standard_error,
    _membership,
    _stream_exact_oracle_mass,
    diagnose_complement,
    exact_oracle_object_ids,
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
        self.active_indices = np.arange(len(self.prior_weights), dtype=np.int64)
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
        del observed, device
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
            component_recipe=TAIL_PROPOSAL_RECIPE,
            atom_chunk=atom_chunk,
            object_chunk=object_chunk,
            object_excluded_candidate_indices_sha256=(
                _candidate_exclusion_manifest_sha256(candidates.indices)
            ),
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
        del g1, g2, kwargs
        indices = np.asarray(atom_indices)
        probability = np.asarray(proposal_probability)
        if indices.ndim == 1:
            indices = np.broadcast_to(indices, (len(observed), len(indices)))
            probability = np.broadcast_to(probability, indices.shape)
        ratio = self.atom_mass[indices] / probability
        return torch.as_tensor(np.log(ratio), dtype=torch.float64)


def test_exact_oracle_selection_has_frozen_golden_manifests():
    assert EXACT_ORACLE_LABEL == "SBSI/complement-exact-oracle/v1"
    assert EXACT_ORACLE_SELECTION_KEY_SHA256 == (
        "f02f535eab874dd6f9f6184d4040283493266074903fc6c20160b4ae88b5993e"
    )
    development, development_manifest = exact_oracle_object_ids(25_000, 25_512)
    confirmation, confirmation_manifest = exact_oracle_object_ids(25_512, 26_024)
    assert len(development) == EXACT_ORACLE_SELECTION_COUNT
    assert len(confirmation) == EXACT_ORACLE_SELECTION_COUNT
    assert np.all(np.diff(development) > 0)
    assert np.all(np.diff(confirmation) > 0)
    assert development_manifest == (
        "6abcc361aec9ada6010f2a91a68aa70440904873518182445c2a712ae69bad6a"
    )
    assert confirmation_manifest == (
        "fc44a632ed4758440a3e241ffeab9fba087f58397f114cfbc38aecc7feedcb8b"
    )
    assert EXACT_ORACLE_CONFIRMATION["center"] == [0.0, 0.0]
    assert EXACT_ORACLE_CONFIRMATION["candidate_backend"] == "torch"
    assert EXACT_ORACLE_CONFIRMATION["diagnostic_object_chunk"] == 64
    assert EXACT_ORACLE_CONFIRMATION["flow_atom_chunk"] == 4096
    assert EXACT_ORACLE_CONFIRMATION["tail_proposal_atom_chunk"] == 262144
    assert EXACT_ORACLE_CONFIRMATION["tail_proposal_object_chunk"] == 32
    assert EXACT_ORACLE_CONFIRMATION["exact_atom_chunk"] == EXACT_ORACLE_ATOM_CHUNK
    assert (
        EXACT_ORACLE_CONFIRMATION["exact_object_chunk"]
        == EXACT_ORACLE_OBJECT_CHUNK
    )
    assert EXACT_ORACLE_CONFIRMATION["selection_key_sha256"] == (
        EXACT_ORACLE_SELECTION_KEY_SHA256
    )
    assert EXACT_ORACLE_CONFIRMATION["selection_manifest_sha256"] == (
        confirmation_manifest
    )
    assert EXACT_ORACLE_CONFIRMATION["tail_proposal_recipe"] == asdict(
        TAIL_PROPOSAL_RECIPE
    )
    assert EXACT_ORACLE_CONFIRMATION["tail_proposal_recipe"]["name"] == (
        "conditioned_dm_mis_t1_t2_prior_4_3_1_v2"
    )
    with pytest.raises(ValueError, match="512-object"):
        exact_oracle_object_ids(25_000, 25_511)


def test_streamed_exact_oracle_matches_toy_finite_sum_and_chunking():
    likelihood = _DiagnosticLikelihood()
    proposal = _DiagnosticProposal()
    observed = pd.DataFrame({"observed": [10.0, 11.0]})
    targets = likelihood.observed_target_tensor(observed)
    strata = np.array([[0], [1]], dtype=np.int64)

    whole = _stream_exact_oracle_mass(
        likelihood,
        observed,
        proposal,
        strata,
        center=(0.0, 0.0),
        observed_targets=targets,
        atom_chunk=4,
        object_chunk=2,
    )
    fragmented = _stream_exact_oracle_mass(
        likelihood,
        observed,
        proposal,
        strata,
        center=(0.0, 0.0),
        observed_targets=targets,
        atom_chunk=1,
        object_chunk=1,
    )

    np.testing.assert_allclose(np.exp(whole.log_stratum), [0.4, 0.3])
    np.testing.assert_allclose(np.exp(whole.log_complement), [0.6, 0.7])
    np.testing.assert_allclose(fragmented.log_stratum, whole.log_stratum, atol=1e-15)
    np.testing.assert_allclose(
        fragmented.log_complement, whole.log_complement, atol=1e-15
    )
    assert whole.flow_evaluations == 8
    assert whole.active_atoms == 4
    assert whole.atoms_visited == 4
    assert whole.atom_blocks == 1
    assert fragmented.atom_blocks == 4


def test_exact_oracle_is_sparse_and_counts_every_active_flow_evaluation():
    proposal = _DiagnosticProposal()
    mock = SimpleNamespace(
        measurements=pd.DataFrame({"observed": np.arange(512, dtype=float)})
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
        object_id_offset=25_000,
        object_chunk=128,
        candidate_backend="scipy",
        include_tail_proposal=True,
        include_exact_oracle=True,
    )

    table = diagnostic.table
    selected = table["exact_oracle_selected"]
    assert selected.sum() == EXACT_ORACLE_SELECTION_COUNT
    np.testing.assert_allclose(
        table.loc[selected, "mass_stratum_exact_oracle"], 0.4
    )
    np.testing.assert_allclose(
        table.loc[selected, "mass_complement_exact_oracle"], 0.6
    )
    np.testing.assert_allclose(table.loc[selected, "mass_total_exact_oracle"], 1.0)
    assert table.loc[~selected, "mass_total_exact_oracle"].isna().all()
    assert (
        table.loc[selected, "mass_stratum_exact_oracle_relative_error"].max()
        < 1e-14
    )

    exact = diagnostic.metadata["exact_oracle"]
    expected_exact_flow = EXACT_ORACLE_SELECTION_COUNT * 4
    expected_base_flow = 512 * (3 + 16 + 16)
    assert exact["atom_chunk"] == EXACT_ORACLE_ATOM_CHUNK
    assert exact["object_chunk"] == EXACT_ORACLE_OBJECT_CHUNK
    assert exact["flow_evaluations"] == expected_exact_flow
    assert diagnostic.metadata["flow_evaluations"] == (
        expected_base_flow + expected_exact_flow
    )
    assert exact["active_atoms_visited_per_selected_object"] == 4
    summary = diagnostic.summary()
    assert summary["exact_oracle"]["selected_objects"] == 64
    assert summary["gates"]["predeclared"] is False
    assert summary["gates"]["thresholds_predeclared"] is True
    assert exact["confirmation_protocol"]["expected"] == EXACT_ORACLE_CONFIRMATION
    assert exact["confirmation_protocol"]["matched"] is False
    observed_protocol = exact["confirmation_protocol"]["observed"]
    assert observed_protocol["center"] == [0.0, 0.0]
    assert observed_protocol["candidate_backend"] == "scipy"
    assert observed_protocol["diagnostic_object_chunk"] == 128
    assert observed_protocol["flow_atom_chunk"] == 4096
    assert observed_protocol["tail_proposal_atom_chunk"] == 262144
    assert observed_protocol["tail_proposal_object_chunk"] == 32
    assert observed_protocol["exact_atom_chunk"] == EXACT_ORACLE_ATOM_CHUNK
    assert observed_protocol["exact_object_chunk"] == EXACT_ORACLE_OBJECT_CHUNK
    assert observed_protocol["selection_key_sha256"] == (
        EXACT_ORACLE_SELECTION_KEY_SHA256
    )
    assert observed_protocol["selection_manifest_sha256"] == (
        "6abcc361aec9ada6010f2a91a68aa70440904873518182445c2a712ae69bad6a"
    )
    assert observed_protocol["tail_proposal_recipe"] == asdict(TAIL_PROPOSAL_RECIPE)
    assert not summary["gates"]["exact_oracle_64_object_subset"][
        "confirmation_protocol_matched"
    ]["passed"]
    assert (
        summary["gates"]["exact_oracle_64_object_subset"][
            "active_support_visited_once_per_selected_object"
        ]["passed"]
    )


def test_exact_oracle_off_preserves_existing_table_and_flow_identity():
    mock = SimpleNamespace(
        measurements=pd.DataFrame({"observed": [10.0, 11.0]})
    )

    def run(include_exact_oracle=None):
        kwargs = {}
        if include_exact_oracle is not None:
            kwargs["include_exact_oracle"] = include_exact_oracle
        return diagnose_complement(
            _DiagnosticLikelihood(),
            mock,
            _DiagnosticProposal(),
            center=(0.0, 0.0),
            n_candidates=1,
            prefilter_candidates=3,
            n_global_draws=16,
            seed=73,
            object_id_offset=25_000,
            object_chunk=1,
            candidate_backend="scipy",
            include_tail_proposal=True,
            **kwargs,
        )

    default = run()
    explicit_off = run(False)
    pd.testing.assert_frame_equal(default.table, explicit_off.table)
    assert not any("exact_oracle" in name for name in default.table.columns)
    assert "exact_oracle" not in default.metadata
    assert "exact_oracle_flow" not in default.metadata["phase_seconds"]
    assert default.metadata["flow_evaluations"] == explicit_off.metadata[
        "flow_evaluations"
    ]
    assert "exact_oracle" not in default.summary()
    with pytest.raises(ValueError, match="tail-proposal"):
        diagnose_complement(
            _DiagnosticLikelihood(),
            mock,
            _DiagnosticProposal(),
            center=(0.0, 0.0),
            n_candidates=1,
            prefilter_candidates=3,
            n_global_draws=16,
            seed=73,
            candidate_backend="scipy",
            include_exact_oracle=True,
        )
    selected_likelihood = _DiagnosticLikelihood()
    selected_likelihood.selection = object()
    with pytest.raises(ValueError, match="measured selection.*disabled"):
        diagnose_complement(
            selected_likelihood,
            mock,
            _DiagnosticProposal(),
            center=(0.0, 0.0),
            n_candidates=1,
            prefilter_candidates=3,
            n_global_draws=16,
            seed=73,
            candidate_backend="scipy",
            include_tail_proposal=True,
            include_exact_oracle=True,
        )


def test_exact_oracle_summary_pins_gate_boundaries_and_protocol_identity():
    selected = np.arange(512) < EXACT_ORACLE_SELECTION_COUNT
    signs = np.where(np.arange(512) % 2, 1.0, -1.0)
    tail_se = np.full(512, 0.008)
    tail_mass = 1.0 + signs * 0.8 * tail_se
    frame = pd.DataFrame(
        {
            "exact_oracle_selected": selected,
            "pareto_k_tail_proposal_complement": np.full(512, 0.49),
            "mass_complement_tail_proposal": tail_mass,
            "mass_complement_tail_proposal_standard_error": tail_se,
            "mass_complement_exact_oracle": np.where(selected, 1.0, np.nan),
            "mass_stratum_exact_oracle": np.where(selected, 1.0, np.nan),
            "mass_total_exact_oracle": np.where(selected, 2.0, np.nan),
            "mass_stratum_exact_oracle_relative_error": np.where(
                selected, 0.0, np.nan
            ),
            "delta_log_total_evidence_tail_minus_exact": np.where(
                selected, np.log((1.0 + tail_mass) / 2.0), np.nan
            ),
            "mass_complement_tail_minus_exact_z": np.where(
                selected, signs * 0.8, np.nan
            ),
        }
    )
    protocol = {
        "active_atoms": 4,
        "active_atoms_visited_per_selected_object": 4,
        "selected_objects_evaluated": 64,
        "confirmation_protocol": {
            "expected": dict(EXACT_ORACLE_CONFIRMATION),
            "observed": dict(EXACT_ORACLE_CONFIRMATION),
            "matched": True,
        },
    }
    summary = ComplementDiagnostic(frame, {"exact_oracle": protocol})._exact_oracle_summary()
    gates = summary["gates"]

    assert gates["predeclared"] is True
    assert gates["passed"] is True
    assert gates["proposal_all_512_objects"][
        "complement_relative_error_median"
    ]["threshold"] == 0.00882
    assert gates["proposal_all_512_objects"][
        "complement_relative_error_p90_higher"
    ]["threshold"] == 0.0268
    exact_gates = gates["exact_oracle_64_object_subset"]
    assert exact_gates["stratum_relative_crosscheck_max"]["threshold"] == 1.0e-5
    assert exact_gates["absolute_delta_log_total_evidence_median"][
        "threshold"
    ] == 0.01
    assert exact_gates["absolute_delta_log_total_evidence_p90_higher"][
        "threshold"
    ] == 0.05
    assert exact_gates["absolute_signed_aggregate_z"]["threshold"] == 2.5
    assert exact_gates["exact_z_rms"]["threshold"] == [0.8, 1.2]
    assert exact_gates["exact_z_within_1_96_count"]["threshold"] == 58


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
