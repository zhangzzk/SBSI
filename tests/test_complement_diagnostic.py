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
    PILOT_STRATUM_RECIPE,
    TAIL_PROPOSAL_RECIPE,
    ProposalCandidates,
    ProposalDraw,
    WholeCatalogueProxy,
    _array_manifest_sha256,
    _candidate_exclusion_manifest_sha256,
    _object_array_manifest_sha256,
)
from sbsi.complement_diagnostic import (
    COMPLEMENT_DIAGNOSTIC_SEED_OFFSET,
    EXACT_ORACLE_ATOM_CHUNK,
    EXACT_ORACLE_CONFIRMATION,
    EXACT_ORACLE_LABEL,
    EXACT_ORACLE_OBJECT_CHUNK,
    EXACT_ORACLE_SELECTION_COUNT,
    EXACT_ORACLE_SELECTION_KEY_SHA256,
    PILOT_STRATUM_DEVELOPMENT_BASELINE,
    PILOT_STRATUM_DEVELOPMENT_GATE_THRESHOLDS,
    PILOT_STRATUM_DEVELOPMENT_LABEL,
    PILOT_STRATUM_DEVELOPMENT_PROTOCOL,
    PILOT_STRATUM_DIAGNOSTIC_SEED_OFFSET,
    PILOT_STRATUM_METHOD_TIME_COMPONENTS,
    PILOT_STRATUM_REPLICATION_SEED_OFFSET,
    ComplementDiagnostic,
    _fixed_component_standard_error,
    _membership,
    _pilot_stratum_run_manifest_sha256,
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
        self.tail_call_details = []
        self.promotion_calls = []

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
        local_member = np.zeros_like(indices, dtype=bool)
        local_position = np.full_like(indices, -1, dtype=np.int32)
        for row in range(len(object_ids)):
            for position, atom in enumerate(candidates.indices[row]):
                member = indices[row] == atom
                local_member[row, member] = True
                local_position[row, member] = position
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
        recipe=TAIL_PROPOSAL_RECIPE,
        atom_chunk=262144,
        object_chunk=32,
    ):
        del observed, device
        self.tail_calls.append((int(seed), object_ids.copy()))
        self.tail_call_details.append(
            {
                "seed": int(seed),
                "object_ids": object_ids.copy(),
                "n_draws": int(n_draws),
                "candidates": candidates.indices.copy(),
                "recipe": recipe,
                "atom_chunk": int(atom_chunk),
                "object_chunk": int(object_chunk),
            }
        )
        schedule = np.resize(
            np.asarray(recipe.component_cycle, dtype=np.int16),
            n_draws,
        )
        indices = np.empty((len(object_ids), n_draws), dtype=np.int64)
        probability = np.empty_like(indices, dtype=np.float64)
        for row in range(len(object_ids)):
            complement = self.active_indices[
                ~np.isin(self.active_indices, candidates.indices[row])
            ]
            if len(complement):
                indices[row] = np.resize(complement, n_draws)
                probability[row].fill(1.0 / len(complement))
            else:
                indices[row].fill(candidates.indices[row, 0])
                probability[row].fill(1.0)
        component_id = np.broadcast_to(schedule, indices.shape).copy()
        n_components = len(recipe.component_names)
        complement_mass = np.broadcast_to(
            np.linspace(0.20, 0.80, n_components),
            (len(object_ids), n_components),
        ).copy()
        complement_mass[:, :-1] += object_ids[:, None] / 1_000_000.0
        complement_mass[:, -1] = 0.90
        prior_component = n_components - 1
        local_member = np.zeros_like(indices, dtype=bool)
        local_position = np.full(indices.shape, -1, dtype=np.int32)
        for row in range(len(object_ids)):
            for position, atom in enumerate(candidates.indices[row]):
                member = indices[row] == atom
                local_member[row, member] = True
                local_position[row, member] = position
        return ProposalDraw(
            indices=indices,
            probability=probability,
            local_member=local_member,
            global_component=component_id == prior_component,
            candidate_radius=candidates.radius,
            seed=seed,
            local_position=local_position,
            component_id=component_id,
            component_complement_mass=complement_mass,
            component_recipe=recipe,
            atom_chunk=atom_chunk,
            object_chunk=object_chunk,
            object_excluded_candidate_indices_sha256=(
                _candidate_exclusion_manifest_sha256(candidates.indices)
            ),
        )

    def promote_exact_stratum(
        self, candidates, observed, pilot_draw, pilot_scores, *, recipe
    ):
        del observed
        assert recipe == PILOT_STRATUM_RECIPE
        self.promotion_calls.append(
            {
                "base": candidates.indices.copy(),
                "pilot_draw": pilot_draw,
                "pilot_scores": pilot_scores.copy(),
            }
        )
        promoted = np.full((len(candidates.indices), 1), 3, dtype=np.int64)
        final = np.concatenate((candidates.indices, promoted), axis=1)
        anchors = promoted.copy()

        def call_hash(label, values, dtype, *, sort_rows=False):
            return _array_manifest_sha256(
                label, values, dtype=dtype, sort_rows=sort_rows
            )

        def object_hashes(label, values, dtype, *, sort_rows=False):
            return _object_array_manifest_sha256(
                label, values, dtype=dtype, sort_rows=sort_rows
            )

        return SimpleNamespace(
            candidates=ProposalCandidates(
                final,
                np.zeros_like(final, dtype=np.float64),
                np.zeros(len(final), dtype=np.float64),
            ),
            recipe=recipe,
            base_size=candidates.indices.shape[1],
            anchor_indices=anchors,
            anchor_valid=np.ones_like(promoted, dtype=bool),
            promoted_indices=promoted,
            pilot_unique_counts=np.full(len(final), 3, dtype=np.int64),
            pilot_finite_score_counts=np.full(len(final), 3, dtype=np.int64),
            fallback_fill_counts=np.zeros(len(final), dtype=np.int64),
            max_neighbor_query_width=np.zeros(len(final), dtype=np.int64),
            provenance=SimpleNamespace(
                pilot_seed=pilot_draw.seed,
                pilot_atom_chunk=pilot_draw.atom_chunk,
                pilot_object_chunk=pilot_draw.object_chunk,
                recipe_sha256=recipe.sha256,
                call_pilot_draw_indices_sha256=call_hash(
                    "call_pilot_draw_indices", pilot_draw.indices, ">i8"
                ),
                call_pilot_probability_sha256=call_hash(
                    "call_pilot_probability", pilot_draw.probability, ">f8"
                ),
                call_pilot_scores_sha256=call_hash(
                    "call_pilot_scores", pilot_scores, ">f8"
                ),
                call_base_stratum_indices_sha256=call_hash(
                    "call_base_stratum_indices",
                    candidates.indices,
                    ">i8",
                    sort_rows=True,
                ),
                call_anchor_indices_sha256=call_hash(
                    "call_anchor_indices", anchors, ">i8"
                ),
                call_promoted_indices_sha256=call_hash(
                    "call_promoted_indices", promoted, ">i8", sort_rows=True
                ),
                call_final_stratum_indices_sha256=call_hash(
                    "call_final_stratum_indices", final, ">i8", sort_rows=True
                ),
                object_pilot_draw_indices_sha256=object_hashes(
                    "object_pilot_draw_indices", pilot_draw.indices, ">i8"
                ),
                object_pilot_probability_sha256=object_hashes(
                    "object_pilot_probability", pilot_draw.probability, ">f8"
                ),
                object_pilot_scores_sha256=object_hashes(
                    "object_pilot_scores", pilot_scores, ">f8"
                ),
                object_base_stratum_indices_sha256=object_hashes(
                    "object_base_stratum_indices",
                    candidates.indices,
                    ">i8",
                    sort_rows=True,
                ),
                object_anchor_indices_sha256=object_hashes(
                    "object_anchor_indices", anchors, ">i8"
                ),
                object_promoted_indices_sha256=object_hashes(
                    "object_promoted_indices",
                    promoted,
                    ">i8",
                    sort_rows=True,
                ),
                object_final_stratum_indices_sha256=object_hashes(
                    "object_final_stratum_indices",
                    final,
                    ">i8",
                    sort_rows=True,
                ),
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

    def run(include_exact_oracle=None, include_pilot_stratum=None):
        kwargs = {}
        if include_exact_oracle is not None:
            kwargs["include_exact_oracle"] = include_exact_oracle
        if include_pilot_stratum is not None:
            kwargs["include_pilot_stratum"] = include_pilot_stratum
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
    explicit_pilot_off = run(include_pilot_stratum=False)
    pd.testing.assert_frame_equal(default.table, explicit_off.table)
    pd.testing.assert_frame_equal(default.table, explicit_pilot_off.table)
    assert not any("exact_oracle" in name for name in default.table.columns)
    assert "exact_oracle" not in default.metadata
    assert "exact_oracle_flow" not in default.metadata["phase_seconds"]
    assert "pilot_stratum" not in default.metadata
    assert not any("pilot_stratum" in name for name in default.table.columns)
    assert not any(
        name.startswith("pilot_") for name in default.metadata["phase_seconds"]
    )
    assert default.metadata["flow_evaluations"] == explicit_off.metadata[
        "flow_evaluations"
    ]
    assert default.metadata["flow_evaluations"] == explicit_pilot_off.metadata[
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


def _run_toy_pilot_stratum(*, object_chunk=2, object_id_offset=25_000):
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
        pilot_seed=37,
        object_id_offset=object_id_offset,
        object_chunk=object_chunk,
        candidate_backend="scipy",
        include_tail_proposal=True,
        include_pilot_stratum=True,
    )
    return diagnostic, proposal


def test_pilot_stratum_is_discarded_then_production_uses_final_stratum():
    diagnostic, proposal = _run_toy_pilot_stratum()

    assert PILOT_STRATUM_DIAGNOSTIC_SEED_OFFSET == 1_000_003
    assert PILOT_STRATUM_REPLICATION_SEED_OFFSET == 2_000_003
    assert COMPLEMENT_DIAGNOSTIC_SEED_OFFSET == 7_000_003
    assert len(proposal.promotion_calls) == 1
    promotion_call = proposal.promotion_calls[0]
    pilot_draw = promotion_call["pilot_draw"]
    expected_log_ratio = np.log(
        _DiagnosticLikelihood.atom_mass[pilot_draw.indices]
        / pilot_draw.probability
    )
    np.testing.assert_allclose(
        promotion_call["pilot_scores"],
        2.0 * expected_log_ratio + np.log(pilot_draw.probability),
    )

    assert [call["seed"] for call in proposal.tail_call_details] == [37, 73]
    assert proposal.tail_call_details[0]["n_draws"] == 8192
    assert proposal.tail_call_details[0]["candidates"].shape == (2, 1)
    assert proposal.tail_call_details[0]["recipe"] == PILOT_STRATUM_RECIPE.pilot_proposal
    assert proposal.tail_call_details[0]["atom_chunk"] == PILOT_STRATUM_RECIPE.atom_chunk
    assert (
        proposal.tail_call_details[0]["object_chunk"]
        == PILOT_STRATUM_RECIPE.object_chunk
    )
    assert proposal.tail_call_details[1]["n_draws"] == 16
    assert proposal.tail_call_details[1]["candidates"].shape == (2, 2)
    assert {seed for seed, _ in proposal.stratified_calls} == {73}

    table = diagnostic.table
    np.testing.assert_allclose(table["mass_stratum"], 0.5)
    np.testing.assert_allclose(table["mass_stratum_base_exact"], 0.4)
    np.testing.assert_allclose(table["mass_stratum_promoted_exact"], 0.1)
    np.testing.assert_allclose(table["mass_stratum_pilot_anchor_exact"], 0.1)
    np.testing.assert_allclose(table["mass_stratum_pilot_neighbor_exact"], 0.0)
    np.testing.assert_allclose(table["mass_prefilter_complement"], 0.5)
    np.testing.assert_allclose(table["mass_outside_prefilter"], 0.0)
    assert table["pilot_stratum_promoted_outside_prefilter_count"].tolist() == [
        1,
        1,
    ]
    assert table["pilot_stratum_anchor_count"].tolist() == [1, 1]
    assert table["pilot_stratum_neighbor_count"].tolist() == [0, 0]
    assert table["pilot_stratum_fallback_fill_count"].tolist() == [0, 0]
    assert table["pilot_stratum_max_neighbor_query_width"].tolist() == [0, 0]
    assert table["draws_outside_prefilter"].tolist() == [0, 0]

    pilot = diagnostic.metadata["pilot_stratum"]
    assert pilot["recipe"] == asdict(PILOT_STRATUM_RECIPE)
    assert pilot["recipe_sha256"] == PILOT_STRATUM_RECIPE.sha256
    assert pilot["pilot_seed"] == 37
    assert pilot["production_seed"] == 73
    assert pilot["pilot_seed_offset"] == 1_000_003
    assert pilot["production_seed_offset"] == 7_000_003
    assert pilot["pilot_atom_chunk"] == PILOT_STRATUM_RECIPE.atom_chunk
    assert pilot["pilot_object_chunk"] == PILOT_STRATUM_RECIPE.object_chunk
    assert pilot["streams_independent"]
    assert pilot["pilot_draws_enter_estimate"] is False
    assert pilot["score_formula"] == "2 * log_target_over_qbar + log(qbar)"
    assert pilot["flow_evaluations"] == {
        "pilot_proposal": 2 * 8192,
        "final_stratum_exact": 2 * 2,
        "total": 2 * 8192 + 2 * 2,
    }
    assert diagnostic.metadata["flow_evaluations"] == (
        2 * (3 + 8192 + 2 + 16 + 16)
    )
    assert len(pilot["object_provenance"]) == 2
    assert len(pilot["execution_call_provenance"]) == 1
    assert pilot["execution_call_provenance"][0]["object_start"] == 25_000
    assert pilot["execution_call_provenance"][0]["object_stop"] == 25_002
    for record in pilot["object_provenance"]:
        assert set(record) == {
            "object_index",
            "pilot_draw_indices_sha256",
            "pilot_probability_sha256",
            "pilot_scores_sha256",
            "base_stratum_indices_sha256",
            "anchor_indices_sha256",
            "promoted_indices_sha256",
            "final_stratum_indices_sha256",
            "conditioned_exclusion_indices_sha256",
        }
        assert all(
            len(value) == 64
            for name, value in record.items()
            if name != "object_index"
        )
    assert set(diagnostic.metadata["phase_seconds"]) == {
        "candidate_query",
        "prefilter_flow",
        "prior_draw",
        "prior_flow",
        "tail_proposal_draw",
        "tail_proposal_flow",
        "pilot_proposal_draw",
        "pilot_proposal_flow",
        "pilot_promotion",
        "pilot_stratum_exact_flow",
        "reduction",
    }
    summary = diagnostic.summary()["pilot_stratum"]
    assert summary["promoted_count_percentiles"] == [1.0, 1.0, 1.0]
    assert summary["mass_promoted_exact_percentiles"] == pytest.approx(
        [0.1, 0.1, 0.1]
    )
    assert summary["promoted_share_of_exact_stratum_percentiles"] == pytest.approx(
        [0.2, 0.2, 0.2]
    )
    assert pilot["development_protocol"]["label"] == (
        PILOT_STRATUM_DEVELOPMENT_LABEL
    )
    assert pilot["development_protocol"]["expected"] == (
        PILOT_STRATUM_DEVELOPMENT_PROTOCOL
    )
    assert pilot["development_protocol"]["matched"] is False
    assert summary["gates"]["predeclared"] is False
    assert summary["gates"]["passed"] is False


def _pilot_gate_diagnostic(
    *,
    complement_relative_error=None,
    total_relative_error=None,
    pareto_k=None,
    method_seconds=1.0,
    identity_matched=True,
):
    objects = PILOT_STRATUM_DEVELOPMENT_GATE_THRESHOLDS["objects"]
    object_ids = np.arange(25_000, 25_000 + objects, dtype=np.int64)
    if complement_relative_error is None:
        complement_relative_error = np.full(objects, 0.008, dtype=np.float64)
    else:
        complement_relative_error = np.broadcast_to(
            np.asarray(complement_relative_error, dtype=np.float64), (objects,)
        ).copy()
    if total_relative_error is None:
        total_relative_error = complement_relative_error / 2.0
    else:
        total_relative_error = np.broadcast_to(
            np.asarray(total_relative_error, dtype=np.float64), (objects,)
        ).copy()
    if pareto_k is None:
        pareto_k = np.full(objects, 0.49, dtype=np.float64)
    else:
        pareto_k = np.broadcast_to(
            np.asarray(pareto_k, dtype=np.float64), (objects,)
        ).copy()

    tail_mass = np.ones(objects, dtype=np.float64)
    tail_standard_error = complement_relative_error * tail_mass
    total_mass = np.divide(
        tail_standard_error,
        total_relative_error,
        out=np.full(objects, np.nan, dtype=np.float64),
        where=total_relative_error != 0.0,
    )
    stratum_mass = total_mass - tail_mass
    object_provenance = []
    for object_id in object_ids:
        value = f"{int(object_id):064x}"
        object_provenance.append(
            {
                "object_index": int(object_id),
                "pilot_draw_indices_sha256": value,
                "pilot_probability_sha256": value,
                "pilot_scores_sha256": value,
                "base_stratum_indices_sha256": value,
                "anchor_indices_sha256": value,
                "promoted_indices_sha256": value,
                "final_stratum_indices_sha256": value,
                "conditioned_exclusion_indices_sha256": value,
            }
        )
    call_hash = "f" * 64
    execution_calls = [
        {
            "object_start": int(start),
            "object_stop": int(start + 64),
            "objects": 64,
            "pilot_draw_indices_sha256": call_hash,
            "pilot_probability_sha256": call_hash,
            "pilot_scores_sha256": call_hash,
            "base_stratum_indices_sha256": call_hash,
            "anchor_indices_sha256": call_hash,
            "promoted_indices_sha256": call_hash,
            "final_stratum_indices_sha256": call_hash,
        }
        for start in range(25_000, 25_512, 64)
    ]
    final_hashes = [
        (record["object_index"], record["final_stratum_indices_sha256"])
        for record in object_provenance
    ]
    observed_identity = dict(PILOT_STRATUM_DEVELOPMENT_PROTOCOL)
    if not identity_matched:
        observed_identity["pilot_seed"] += 1
    protocol = {
        "recipe": asdict(PILOT_STRATUM_RECIPE),
        "recipe_sha256": PILOT_STRATUM_RECIPE.sha256,
        "hash_contract": PILOT_STRATUM_DEVELOPMENT_PROTOCOL["hash_contract"],
        "pilot_seed": PILOT_STRATUM_DEVELOPMENT_PROTOCOL["pilot_seed"],
        "production_seed": PILOT_STRATUM_DEVELOPMENT_PROTOCOL[
            "production_seed"
        ],
        "mock_input_sha256": PILOT_STRATUM_DEVELOPMENT_PROTOCOL[
            "mock_input_sha256"
        ],
        "pilot_atom_chunk": PILOT_STRATUM_DEVELOPMENT_PROTOCOL[
            "pilot_atom_chunk"
        ],
        "pilot_object_chunk": PILOT_STRATUM_DEVELOPMENT_PROTOCOL[
            "pilot_object_chunk"
        ],
        "base_stratum_size": 16_384,
        "final_stratum_size": 20_480,
        "flow_evaluations": {
            "pilot_proposal": PILOT_STRATUM_DEVELOPMENT_PROTOCOL[
                "pilot_flow_evaluations"
            ],
            "final_stratum_exact": PILOT_STRATUM_DEVELOPMENT_PROTOCOL[
                "final_stratum_flow_evaluations"
            ],
            "total": (
                PILOT_STRATUM_DEVELOPMENT_PROTOCOL["pilot_flow_evaluations"]
                + PILOT_STRATUM_DEVELOPMENT_PROTOCOL[
                    "final_stratum_flow_evaluations"
                ]
            ),
        },
        "object_provenance": object_provenance,
        "execution_call_provenance": execution_calls,
        "run_manifest_sha256": _pilot_stratum_run_manifest_sha256(final_hashes),
        "development_protocol": {
            "label": PILOT_STRATUM_DEVELOPMENT_LABEL,
            "expected": dict(PILOT_STRATUM_DEVELOPMENT_PROTOCOL),
            "observed": observed_identity,
            "matched": bool(identity_matched),
        },
    }
    frame = pd.DataFrame(
        {
            "object_index": object_ids,
            "mass_stratum": stratum_mass,
            "mass_prefilter_complement": np.zeros(objects),
            "mass_outside_prefilter": np.zeros(objects),
            "mass_outside_prefilter_standard_error": np.zeros(objects),
            "mass_complement_tail_proposal": tail_mass,
            "mass_complement_tail_proposal_standard_error": (
                tail_standard_error
            ),
            "mass_complement_prior": tail_mass,
            "mass_complement_prior_standard_error": tail_standard_error,
            "pareto_k_complement": pareto_k,
            "pareto_k_prefilter_complement": pareto_k,
            "pareto_k_outside_prefilter": pareto_k,
            "pareto_k_tail_proposal_complement": pareto_k,
            "pilot_stratum_base_count": np.full(objects, 16_384),
            "pilot_stratum_promoted_count": np.full(objects, 4_096),
            "pilot_stratum_anchor_count": np.full(objects, 256),
            "pilot_stratum_neighbor_count": np.zeros(objects, dtype=np.int64),
            "pilot_stratum_fallback_fill_count": np.full(objects, 3_840),
            "pilot_stratum_max_neighbor_query_width": np.zeros(
                objects, dtype=np.int64
            ),
            "pilot_stratum_finite_score_atom_count": np.full(objects, 256),
            "pilot_stratum_promoted_outside_prefilter_count": np.zeros(objects),
            "mass_stratum_promoted_exact": np.full(objects, 0.1),
            "draws_tail_proposal_complement": np.full(objects, 16_384),
            "pilot_stratum_final_indices_sha256": [
                record["final_stratum_indices_sha256"]
                for record in object_provenance
            ],
        }
    )
    for name in TAIL_PROPOSAL_RECIPE.component_names:
        frame[f"tail_proposal_component_complement_mass_{name}"] = 1.0
        frame[
            f"tail_proposal_component_target_contribution_fraction_{name}"
        ] = 1.0 / len(TAIL_PROPOSAL_RECIPE.component_names)
        frame[f"tail_proposal_component_variance_fraction_{name}"] = (
            1.0 / len(TAIL_PROPOSAL_RECIPE.component_names)
        )
        frame[f"pareto_k_tail_proposal_component_{name}"] = 0.49
        frame[
            f"tail_proposal_top_1_percent_source_fraction_{name}"
        ] = 1.0 / len(TAIL_PROPOSAL_RECIPE.component_names)
    phase_seconds = {
        name: 0.0 for name in PILOT_STRATUM_METHOD_TIME_COMPONENTS
    }
    phase_seconds[PILOT_STRATUM_METHOD_TIME_COMPONENTS[0]] = method_seconds
    phase_seconds.update({"prior_draw": 17.0, "prior_flow": 23.0})
    metadata = {
        "observation_start": 25_000,
        "observation_stop": 25_512,
        "center": PILOT_STRATUM_DEVELOPMENT_PROTOCOL["center"],
        "prefilter_candidates": 131_072,
        "n_global_draws": 16_384,
        "candidate_backend": "torch",
        "object_chunk": 64,
        "flow_atom_chunk": 4_096,
        "tail_proposal_atom_chunk": 262_144,
        "tail_proposal_object_chunk": 32,
        "tail_proposal": asdict(TAIL_PROPOSAL_RECIPE),
        "flow_evaluations": PILOT_STRATUM_DEVELOPMENT_PROTOCOL[
            "run_flow_evaluations"
        ],
        "phase_seconds": phase_seconds,
        "elapsed_seconds": method_seconds + 40.0,
        "pilot_stratum": protocol,
    }
    return ComplementDiagnostic(frame, metadata)


def test_pilot_development_protocol_and_baseline_are_frozen():
    protocol = PILOT_STRATUM_DEVELOPMENT_PROTOCOL
    assert PILOT_STRATUM_DEVELOPMENT_LABEL == (
        "SBSI/complement-pilot-stratum-development/v2-direct"
    )
    assert protocol["universe_start"] == 25_000
    assert protocol["universe_stop"] == 25_512
    assert protocol["center"] == [
        0.013610377120038044,
        0.002740148300964851,
    ]
    assert protocol["base_stratum_size"] == 16_384
    assert protocol["promotions"] == 4_096
    assert protocol["final_stratum_size"] == 20_480
    assert protocol["prefilter_candidates"] == 131_072
    assert protocol["n_global_draws"] == 16_384
    assert protocol["proposal_seed"] == 8_701
    assert protocol["production_seed"] == 7_008_704
    assert protocol["pilot_seed"] == 1_008_704
    assert protocol["diagnostic_object_chunk"] == 64
    assert protocol["flow_atom_chunk"] == 4_096
    assert protocol["tail_proposal_atom_chunk"] == 262_144
    assert protocol["tail_proposal_object_chunk"] == 32
    assert protocol["pilot_atom_chunk"] == 262_144
    assert protocol["pilot_object_chunk"] == 32
    assert protocol["tail_proposal_recipe"] == asdict(TAIL_PROPOSAL_RECIPE)
    assert protocol["pilot_stratum_recipe"] == asdict(PILOT_STRATUM_RECIPE)
    assert protocol["pilot_stratum_recipe_sha256"] == (
        "7571ab61f7ee16e10199abdfc47c68113f89ca121451057526f94f2bcfac9154"
    )
    assert protocol["pilot_flow_evaluations"] == 4_194_304
    assert protocol["final_stratum_flow_evaluations"] == 10_485_760
    assert protocol["run_flow_evaluations"] == 98_566_144
    assert protocol["mock_input_sha256"] == {
        "measurements.parquet": (
            "fdfecacc13adabd6005e21150ab0dc74767cad34da3d3ea95e362515ce38365b"
        ),
        "truth.parquet": (
            "78d1c8e02f43e1ac4673b2ca309b484d3565de3b9400bf736ca98bf584e5dc30"
        ),
        "likelihood_mock_manifest.json": (
            "5c097c98fa820177738156f6dd3290daaf68594dd90b2e80733f083b019d9a48"
        ),
    }
    assert protocol["mock_input_sha256"] == EXACT_ORACLE_CONFIRMATION[
        "mock_input_sha256"
    ]
    assert PILOT_STRATUM_DEVELOPMENT_BASELINE == {
        "total_relative_error_median": 0.013214266498501442,
        "total_relative_error_p90_higher": 0.021110866788308195,
        "mean_total_relative_error_squared": 4.6256941067948046e-4,
        "total_relative_error_squared_p90_higher": 4.45668696553694e-4,
        "method_seconds": 91.04022970038932,
        "total_elapsed_seconds": 103.91162065404933,
    }
    assert PILOT_STRATUM_DEVELOPMENT_GATE_THRESHOLDS == {
        "objects": 512,
        "pareto_k_below_0_5_count": 461,
        "pareto_k_below_0_7_count": 507,
        "complement_relative_error_median": 0.00882,
        "complement_relative_error_p90_higher": 0.0268,
        "efficiency_mean": 0.90,
        "efficiency_p90_higher": 0.90,
    }


def test_pilot_development_summary_passes_only_all_three_gate_families():
    diagnostic = _pilot_gate_diagnostic()
    summary = diagnostic.summary()
    pilot = summary["pilot_stratum"]

    assert summary["gates"] == pilot["gates"]
    assert pilot["development_protocol_matched"] is True
    assert pilot["gates"]["predeclared"] is True
    assert pilot["gates"]["integrity_passed"] is True
    assert pilot["gates"]["tail_reliability_passed"] is True
    assert pilot["gates"]["efficiency_passed"] is True
    assert pilot["gates"]["passed"] is True
    assert pilot["method_seconds"] == 1.0
    assert pilot["total_elapsed_seconds"] == 41.0
    assert pilot["total_elapsed_ratio"] == pytest.approx(
        41.0 / PILOT_STRATUM_DEVELOPMENT_BASELINE["total_elapsed_seconds"]
    )
    assert pilot["method_time_components"] == list(
        PILOT_STRATUM_METHOD_TIME_COMPONENTS
    )

    diagnostic.metadata["pilot_stratum"]["object_provenance"][0][
        "pilot_scores_sha256"
    ] = "not-a-hash"
    failed = diagnostic._pilot_stratum_summary()["gates"]
    assert failed["integrity_passed"] is False
    assert failed["tail_reliability_passed"] is True
    assert failed["efficiency_passed"] is True
    assert failed["passed"] is False


@pytest.mark.parametrize(
    ("column", "value", "gate_name"),
    (
        (
            "pilot_stratum_anchor_count",
            257,
            "direct_anchor_count_matches_finite_scores_all_objects",
        ),
        (
            "pilot_stratum_fallback_fill_count",
            3_839,
            "direct_fallback_fills_uniqueness_deficit_all_objects",
        ),
        (
            "pilot_stratum_neighbor_count",
            1,
            "zero_coordinate_neighbors_all_objects",
        ),
        (
            "pilot_stratum_max_neighbor_query_width",
            1,
            "zero_neighbor_query_width_all_objects",
        ),
    ),
)
def test_direct_pilot_integrity_rejects_wrong_promotion_counts(
    column, value, gate_name
):
    diagnostic = _pilot_gate_diagnostic()
    diagnostic.table.loc[0, column] = value
    gates = diagnostic._pilot_stratum_summary()["gates"]

    assert gates["integrity"][gate_name]["passed"] is False
    assert gates["integrity_passed"] is False
    assert gates["passed"] is False


def test_direct_pilot_integrity_rejects_fractional_count_columns():
    diagnostic = _pilot_gate_diagnostic()
    diagnostic.table["pilot_stratum_finite_score_atom_count"] = (
        diagnostic.table["pilot_stratum_finite_score_atom_count"].astype(float)
        + 0.5
    )
    gates = diagnostic._pilot_stratum_summary()["gates"]

    assert gates["integrity"][
        "count_columns_are_nonnegative_int64_compatible_integers"
    ]["passed"] is False
    assert gates["integrity_passed"] is False
    assert gates["passed"] is False


def test_pilot_development_tail_gates_use_higher_and_include_boundaries():
    relative_error = np.concatenate(
        (
            np.full(460, 0.00882),
            np.array([0.0268]),
            np.full(51, 0.03),
        )
    )
    pareto_k = np.concatenate(
        (
            np.full(461, 0.49),
            np.full(46, 0.69),
            np.full(5, 0.70),
        )
    )
    pilot = _pilot_gate_diagnostic(
        complement_relative_error=relative_error,
        total_relative_error=relative_error / 2.0,
        pareto_k=pareto_k,
    )._pilot_stratum_summary()
    gates = pilot["gates"]["tail_reliability"]

    assert pilot["quantile_method"] == "higher"
    assert pilot["complement_relative_error_median"] == pytest.approx(0.00882)
    assert pilot["complement_relative_error_p90_higher"] == pytest.approx(0.0268)
    assert gates["pareto_k_below_0_5_count"]["observed"] == 461
    assert gates["pareto_k_below_0_7_count"]["observed"] == 507
    assert all(item["passed"] for item in gates.values())

    failing_k = pareto_k.copy()
    failing_k[460] = 0.70
    failed = _pilot_gate_diagnostic(
        complement_relative_error=relative_error,
        total_relative_error=relative_error / 2.0,
        pareto_k=failing_k,
    )._pilot_stratum_summary()["gates"]
    assert failed["tail_reliability_passed"] is False
    assert failed["passed"] is False


def test_pilot_development_efficiency_gate_boundaries_and_failures():
    baseline = PILOT_STRATUM_DEVELOPMENT_BASELINE
    threshold = PILOT_STRATUM_DEVELOPMENT_GATE_THRESHOLDS["efficiency_mean"]

    mean_boundary = _pilot_gate_diagnostic(
        total_relative_error=np.sqrt(
            baseline["mean_total_relative_error_squared"]
        ),
        method_seconds=threshold * baseline["method_seconds"],
    )._pilot_stratum_summary()
    mean_gate = mean_boundary["gates"]["efficiency"][
        "mean_relative_variance_time_efficiency"
    ]
    assert mean_gate["observed"] == pytest.approx(threshold)
    assert mean_gate["passed"] is True

    p90_boundary = _pilot_gate_diagnostic(
        total_relative_error=np.sqrt(
            baseline["total_relative_error_squared_p90_higher"]
        ),
        method_seconds=threshold * baseline["method_seconds"],
    )._pilot_stratum_summary()
    p90_gate = p90_boundary["gates"]["efficiency"][
        "p90_higher_relative_variance_time_efficiency"
    ]
    assert p90_gate["observed"] == pytest.approx(threshold)
    assert p90_gate["passed"] is True

    failed = _pilot_gate_diagnostic(
        total_relative_error=np.sqrt(
            baseline["total_relative_error_squared_p90_higher"]
        ),
        method_seconds=np.nextafter(
            threshold * baseline["method_seconds"], np.inf
        ),
        identity_matched=False,
    )._pilot_stratum_summary()
    assert failed["gates"]["predeclared"] is False
    assert failed["gates"]["integrity_passed"] is False
    assert failed["gates"]["efficiency_passed"] is False
    assert failed["gates"]["passed"] is False

    zero_time = _pilot_gate_diagnostic(method_seconds=0.0)._pilot_stratum_summary()
    assert np.isnan(zero_time["method_seconds"])
    assert zero_time["gates"]["efficiency_passed"] is False
    assert zero_time["gates"]["passed"] is False


def test_pilot_development_integrity_rejects_nonintegral_object_ids():
    table_ids = _pilot_gate_diagnostic()
    table_ids.table["object_index"] = (
        table_ids.table["object_index"].astype(np.float64) + 0.5
    )
    table_summary = table_ids._pilot_stratum_summary()
    assert table_summary["gates"]["integrity"][
        "absolute_object_ids_exact"
    ]["passed"] is False
    assert table_summary["gates"]["passed"] is False

    provenance_ids = _pilot_gate_diagnostic()
    for record in provenance_ids.metadata["pilot_stratum"]["object_provenance"]:
        record["object_index"] += 0.5
    provenance_summary = provenance_ids._pilot_stratum_summary()
    assert provenance_summary["gates"]["integrity"][
        "hashes_and_run_manifest_internally_matched"
    ]["passed"] is False
    assert provenance_summary["gates"]["passed"] is False


def test_pilot_stratum_manifest_composes_from_identical_chunked_and_sliced_rows():
    together, _ = _run_toy_pilot_stratum(object_chunk=2)
    separate, _ = _run_toy_pilot_stratum(object_chunk=1)
    pd.testing.assert_frame_equal(together.table, separate.table)
    assert (
        together.metadata["pilot_stratum"]["run_manifest_sha256"]
        == separate.metadata["pilot_stratum"]["run_manifest_sha256"]
    )

    sliced_proposal = _DiagnosticProposal()
    sliced = diagnose_complement(
        _DiagnosticLikelihood(),
        SimpleNamespace(measurements=pd.DataFrame({"observed": [11.0]})),
        sliced_proposal,
        center=(0.0, 0.0),
        n_candidates=1,
        prefilter_candidates=3,
        n_global_draws=16,
        seed=73,
        pilot_seed=37,
        object_id_offset=25_001,
        object_chunk=1,
        candidate_backend="scipy",
        include_tail_proposal=True,
        include_pilot_stratum=True,
    )
    pd.testing.assert_series_equal(
        together.table.iloc[1], sliced.table.iloc[0], check_names=False
    )
    full_identity = together.metadata["pilot_stratum"]["object_provenance"][1]
    assert sliced.metadata["pilot_stratum"]["object_provenance"] == [
        full_identity
    ]
    assert len(together.metadata["pilot_stratum"]["execution_call_provenance"]) == 1
    assert len(separate.metadata["pilot_stratum"]["execution_call_provenance"]) == 2

    object_hashes = [
        (entry["object_index"], entry["final_stratum_indices_sha256"])
        for entry in together.metadata["pilot_stratum"]["object_provenance"]
    ]
    assert _pilot_stratum_run_manifest_sha256(object_hashes) == (
        _pilot_stratum_run_manifest_sha256(list(reversed(object_hashes)))
    )


def test_pilot_stratum_fails_closed_on_invalid_mode_or_seed():
    mock = SimpleNamespace(measurements=pd.DataFrame({"observed": [10.0]}))
    common = {
        "likelihood": _DiagnosticLikelihood(),
        "mock": mock,
        "proposal": _DiagnosticProposal(),
        "center": (0.0, 0.0),
        "n_candidates": 1,
        "prefilter_candidates": 3,
        "n_global_draws": 16,
        "seed": 73,
        "candidate_backend": "scipy",
    }
    with pytest.raises(ValueError, match="requires the tail-proposal"):
        diagnose_complement(**common, include_pilot_stratum=True, pilot_seed=37)
    with pytest.raises(ValueError, match="explicit pilot seed"):
        diagnose_complement(
            **common,
            include_tail_proposal=True,
            include_pilot_stratum=True,
        )
    with pytest.raises(ValueError, match="must be different"):
        diagnose_complement(
            **common,
            include_tail_proposal=True,
            include_pilot_stratum=True,
            pilot_seed=73,
        )
    with pytest.raises(ValueError, match="exact-oracle v1"):
        diagnose_complement(
            **common,
            include_tail_proposal=True,
            include_pilot_stratum=True,
            pilot_seed=37,
            include_exact_oracle=True,
        )
    with pytest.raises(ValueError, match="pilot seed requires"):
        diagnose_complement(**common, pilot_seed=37)

    selected_likelihood = _DiagnosticLikelihood()
    selected_likelihood.selection = object()
    with pytest.raises(ValueError, match="measured selection.*disabled"):
        diagnose_complement(
            **{**common, "likelihood": selected_likelihood},
            include_tail_proposal=True,
            include_pilot_stratum=True,
            pilot_seed=37,
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
