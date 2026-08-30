from dataclasses import FrozenInstanceError, asdict, replace
import json

import numpy as np
import pandas as pd
import pytest
from scipy.special import logsumexp

from sbsi.catalogue_closure import generate_mock_catalogue
from sbsi.catalogue_likelihood import CatalogueLikelihood, CatalogueSelection
from sbsi.catalogue_sampling import (
    DefensiveLocalProposal,
    PILOT_STRATUM_HASH_CONTRACT,
    PILOT_STRATUM_PROPOSAL_RECIPE,
    PILOT_STRATUM_RECIPE,
    PilotStratumRecipe,
    ProposalCandidates,
    ProposalCoordinateTable,
    ProposalDraw,
    TAIL_PROPOSAL_RECIPE,
    TailProposalRecipe,
    _candidate_exclusion_manifest_sha256,
    assess_importance_convergence,
    importance_diagnostics,
    pareto_tail_index,
    run_importance_ladder,
    run_importance_profiles,
    select_adaptive_draw_counts,
    select_independent_pilot_draw_counts,
)
from sbsi.catalogue_likelihood import OutputCut
from test_catalogue_likelihood import AnalyticGaussianSelection, _likelihood


def _coordinates(likelihood):
    values = likelihood.cache.get(0.0, 0.0).flow[["e1_input_p"]].to_numpy(float)
    return ProposalCoordinateTable(
        values=values,
        target_names=("measured_e1",),
        center=np.array([0.0]),
        scale=np.array([0.2]),
        statistic="mean",
    )


def _mis_fixture():
    coordinates = ProposalCoordinateTable(
        values=np.array(
            [
                [-1.2, 0.3],
                [-0.8, -0.2],
                [-0.35, 0.5],
                [0.0, -0.1],
                [0.25, 0.35],
                [0.55, -0.45],
                [0.9, 0.15],
                [1.3, -0.3],
                [9.0, 9.0],
            ]
        ),
        target_names=("x", "y"),
        center=np.zeros(2),
        scale=np.ones(2),
        dispersion=np.array(
            [
                [0.32, 0.55],
                [0.45, 0.28],
                [0.25, 0.48],
                [0.60, 0.35],
                [0.22, 0.31],
                [0.38, 0.62],
                [0.52, 0.24],
                [0.29, 0.44],
                [1.00, 1.00],
            ]
        ),
        statistic="mean",
    )
    prior = np.array([0.03, 0.07, 0.11, 0.19, 0.21, 0.16, 0.14, 0.09, 0.0])
    detection = np.array([0.8, 0.3, 0.95, 0.55, 0.7, 0.4, 0.9, 0.25, 0.0])
    proposal = DefensiveLocalProposal(
        coordinates, prior, local_base_weights=detection
    )
    observed = pd.DataFrame(
        {"x": [-0.42, 0.18, 0.76], "y": [0.08, 0.22, -0.16]}
    )
    candidates = ProposalCandidates(
        indices=np.array([[1, 4], [0, 5], [2, 7]]),
        distances=np.zeros((3, 2)),
        radius=np.ones(3),
    )
    return proposal, observed, candidates


def _pilot_promotion_fixture(n_objects=1):
    coordinates = ProposalCoordinateTable(
        values=np.arange(12, dtype=np.float64)[:, None],
        target_names=("x",),
        center=np.zeros(1),
        scale=np.ones(1),
        dispersion=np.ones((12, 1)),
        statistic="mean",
    )
    proposal = DefensiveLocalProposal(coordinates, np.full(12, 1.0 / 12.0))
    observed = pd.DataFrame({"x": np.linspace(5.0, 6.0, n_objects)})
    candidates = ProposalCandidates(
        indices=np.broadcast_to(
            np.array([0, 11], dtype=np.int64), (n_objects, 2)
        ).copy(),
        distances=np.zeros((n_objects, 2), dtype=np.float64),
        radius=np.zeros(n_objects, dtype=np.float64),
    )
    recipe = replace(
        PILOT_STRATUM_RECIPE,
        name="toy_explicit_score_pilot",
        pilot_draws=8,
        max_anchors=2,
        n_promotions=6,
        neighbors_per_anchor=3,
        score_name="toy_explicit_score",
    )
    patterns = np.array(
        [
            [5, 5, 7, 6, 7, 8, 9, 4],
            [8, 8, 3, 7, 3, 2, 1, 6],
        ],
        dtype=np.int64,
    )[:n_objects]
    scores = np.array(
        [
            [10.0, 1.0, 9.0, 9.0, 2.0, -np.inf, 0.0, 3.0],
            [7.0, 2.0, 8.0, 6.0, 1.0, 5.0, 4.0, 3.0],
        ],
        dtype=np.float64,
    )[:n_objects]
    schedule = np.tile(
        np.asarray(recipe.pilot_proposal.component_cycle, dtype=np.int16),
        recipe.pilot_draws // recipe.pilot_proposal.cycle_length,
    )
    component_id = np.broadcast_to(schedule, patterns.shape).copy()
    n_components = len(recipe.pilot_proposal.component_names)
    draw = ProposalDraw(
        indices=patterns,
        probability=np.full(patterns.shape, 0.1),
        local_member=np.zeros(patterns.shape, dtype=bool),
        global_component=component_id == n_components - 1,
        candidate_radius=candidates.radius,
        seed=812_701,
        local_position=np.full(patterns.shape, -1, dtype=np.int32),
        component_id=component_id,
        component_complement_mass=np.full(
            (n_objects, n_components), 0.75, dtype=np.float64
        ),
        component_recipe=recipe.pilot_proposal,
        atom_chunk=recipe.atom_chunk,
        object_chunk=recipe.object_chunk,
        object_excluded_candidate_indices_sha256=(
            _candidate_exclusion_manifest_sha256(candidates.indices)
        ),
    )
    return proposal, observed, candidates, draw, scores, recipe


def _dense_conditioned_components(proposal, observation, excluded):
    recipe = TAIL_PROPOSAL_RECIPE
    proxy = proposal._tilted_proxy(None)
    excluded_position = np.searchsorted(proposal.active_indices, excluded)
    complement = np.ones(len(proposal.active_indices), dtype=bool)
    complement[excluded_position] = False
    prior = proposal.prior_weights[proposal.active_indices]
    prior_component = np.zeros_like(prior)
    prior_component[complement] = prior[complement] / prior[complement].sum()
    components = []
    masses = []
    for score_tensor in proxy.component_log_scores(observation, recipe):
        score = score_tensor.cpu().numpy()
        component = np.zeros_like(score)
        if np.isfinite(score[complement]).any():
            component[complement] = np.exp(
                score[complement] - logsumexp(score[complement])
            )
            masses.append(
                np.exp(logsumexp(score[complement]) - logsumexp(score))
            )
        else:
            component = prior_component.copy()
            masses.append(0.0)
        components.append(component)
    components.append(prior_component)
    masses.append(prior[complement].sum())
    components = np.stack(components)
    qbar = np.asarray(recipe.component_weights) @ components
    return components, qbar, np.asarray(masses), complement


def test_pilot_stratum_recipe_freezes_reviewed_second_moment_design():
    recipe = PILOT_STRATUM_RECIPE
    assert isinstance(recipe, PilotStratumRecipe)
    assert recipe.pilot_draws == 8192
    assert recipe.max_anchors == 256
    assert recipe.n_promotions == 4096
    assert recipe.neighbors_per_anchor == 16
    assert recipe.atom_chunk == 262144
    assert recipe.object_chunk == 32
    assert recipe.score_name == "log_target_squared_over_proposal"
    assert recipe.pilot_proposal is TAIL_PROPOSAL_RECIPE
    assert PILOT_STRATUM_PROPOSAL_RECIPE is TAIL_PROPOSAL_RECIPE
    assert recipe.hash_contract == PILOT_STRATUM_HASH_CONTRACT
    assert recipe.sha256 == (
        "b4349271763311c2127d785c79fc637e115f421a31fb821e6db6e0a592b8d410"
    )
    with pytest.raises(FrozenInstanceError):
        recipe.pilot_draws = 4096

    with pytest.raises(ValueError, match="positive multiple"):
        replace(recipe, pilot_draws=8191)
    with pytest.raises(ValueError, match="maximum pilot anchors"):
        replace(recipe, max_anchors=8193)
    with pytest.raises(ValueError, match="score name"):
        replace(recipe, score_name="")
    with pytest.raises(ValueError, match="chunk sizes"):
        replace(recipe, atom_chunk=0)
    with pytest.raises(ValueError, match="unsupported pilot-anchor"):
        replace(recipe, ranking="python_hash_order")


def test_pilot_promotion_ranks_unique_explicit_scores_and_fills_fixed_width():
    proposal, observed, candidates, draw, scores, recipe = (
        _pilot_promotion_fixture()
    )
    result = proposal.promote_exact_stratum(
        candidates, observed, draw, scores, recipe=recipe
    )

    np.testing.assert_array_equal(result.anchor_counts, [2])
    np.testing.assert_array_equal(result.anchor_indices[0], [5, 6])
    # Atom 5 is duplicated, but frequency does not change its maximum explicit
    # score. Atoms 6 and 7 tie, so absolute id chooses anchor 6.
    np.testing.assert_array_equal(result.promoted_indices[0], [5, 4, 6, 7, 3, 2])
    np.testing.assert_array_equal(result.candidates.indices[0, :2], [0, 11])
    assert result.candidates.indices.shape == (1, 8)
    assert len(np.unique(result.candidates.indices[0])) == 8
    np.testing.assert_array_equal(result.pilot_unique_counts, [6])
    np.testing.assert_array_equal(result.pilot_finite_score_counts, [5])
    np.testing.assert_array_equal(result.fallback_fill_counts, [0])
    # The two three-atom clusters overlap, so the same two anchors are deepened
    # to six canonical neighbors rather than consuming a third pilot anchor.
    np.testing.assert_array_equal(result.max_neighbor_query_width, [6])
    assert result.promotion_size == 6
    assert result.provenance.pilot_seed == draw.seed
    assert result.provenance.pilot_atom_chunk == recipe.atom_chunk
    assert result.provenance.pilot_object_chunk == recipe.object_chunk
    assert result.provenance.recipe_sha256 == recipe.sha256
    assert result.provenance.call_pilot_draw_indices_sha256 == (
        "06f37b0cac10263e14ed0dd9566c4bf7b3eae3b058066964bafe72fdc0e492c0"
    )
    assert result.provenance.call_pilot_probability_sha256 == (
        "003895ad51168bc3615f96d0382025831cd23d4a0e8a01113d1e05a9bf82e6a3"
    )
    assert result.provenance.call_pilot_scores_sha256 == (
        "a4aec0f60cdc444c096132876266f625a79ca8b0dcb90b3ac36f539846f7a663"
    )
    assert result.provenance.call_final_stratum_indices_sha256 == (
        "e6c7778e1e2da1d851436145e448c0a4513b6b5fb55ad7527b4bbdbae5db9c7f"
    )
    assert result.provenance.object_final_stratum_indices_sha256 == (
        "e9f4eda64b6611125b11776381319d237eece1cddb6293e2eefb41e1493c7fc6",
    )
    np.testing.assert_allclose(
        result.candidates.distances[0],
        [5.0, 6.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0],
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_array_equal(result.candidates.radius, [6.0])
    assert not result.anchor_indices.flags.writeable
    assert not result.promoted_indices.flags.writeable
    assert not result.candidates.indices.flags.writeable
    assert not result.candidates.distances.flags.writeable
    assert not result.candidates.radius.flags.writeable
    final_hash = result.provenance.call_final_stratum_indices_sha256
    with pytest.raises(ValueError):
        result.promoted_indices[0, 0] = 1
    with pytest.raises(ValueError):
        result.candidates.indices[0, 0] = 1
    with pytest.raises(ValueError):
        result.candidates.distances[0, 0] = -1.0
    with pytest.raises(ValueError):
        result.candidates.radius[0] = -1.0
    candidates.indices[0, 0] = 1
    assert result.candidates.indices[0, 0] == 0
    assert result.provenance.call_final_stratum_indices_sha256 == final_hash


def test_pilot_promotion_is_independent_of_kdtree_tie_return_order():
    proposal, observed, candidates, draw, scores, recipe = (
        _pilot_promotion_fixture()
    )
    expected_neighbors = proposal._canonical_coordinate_neighbors(
        np.array([5]), n_neighbors=2
    )
    np.testing.assert_array_equal(expected_neighbors, [[5, 4]])
    expected = proposal.promote_exact_stratum(
        candidates, observed, draw, scores, recipe=recipe
    )

    underlying = proposal.tree

    class ReverseTieOrder:
        def query(self, *args, **kwargs):
            distances, positions = underlying.query(*args, **kwargs)
            distances = np.asarray(distances).copy()
            positions = np.asarray(positions).copy()
            for distance in np.unique(distances):
                tied = distances == distance
                positions[tied] = positions[tied][::-1]
            return distances, positions

    proposal.tree = ReverseTieOrder()
    reordered_neighbors = proposal._canonical_coordinate_neighbors(
        np.array([5]), n_neighbors=2
    )
    actual = proposal.promote_exact_stratum(
        candidates, observed, draw, scores, recipe=recipe
    )
    np.testing.assert_array_equal(reordered_neighbors, expected_neighbors)
    np.testing.assert_array_equal(actual.promoted_indices, expected.promoted_indices)
    assert (
        actual.provenance.call_final_stratum_indices_sha256
        == expected.provenance.call_final_stratum_indices_sha256
    )


def test_pilot_neighbors_bound_memory_for_large_coordinate_ties():
    n_atoms = 20_000
    coordinates = ProposalCoordinateTable(
        values=np.zeros((n_atoms, 1), dtype=np.float64),
        target_names=("x",),
        center=np.zeros(1),
        scale=np.ones(1),
        statistic="mean",
    )
    proposal = DefensiveLocalProposal(
        coordinates, np.full(n_atoms, 1.0 / n_atoms)
    )
    underlying = proposal.tree

    class OneAnchorQueryOnly:
        max_probe = 0

        def query(self, query, *, k, workers):
            assert np.asarray(query).ndim == 1
            self.max_probe = max(self.max_probe, int(k))
            return underlying.query(query, k=k, workers=workers)

        def query_ball_point(self, *args, **kwargs):
            raise AssertionError("canonical neighbor search used an unbounded ball query")

    guarded = OneAnchorQueryOnly()
    proposal.tree = guarded
    neighbors = proposal._canonical_coordinate_neighbors(
        np.array([n_atoms - 1]), n_neighbors=16
    )
    np.testing.assert_array_equal(neighbors[0], np.arange(16))
    assert guarded.max_probe == n_atoms


def test_pilot_promotion_never_materializes_whole_standardized_table(monkeypatch):
    proposal, observed, candidates, draw, scores, recipe = (
        _pilot_promotion_fixture()
    )

    def forbidden(_self):
        raise AssertionError("promotion materialized the whole standardized table")

    # The proposal tree is already built. Promotion must standardize only the
    # indexed anchors, boundary pools, and promoted atoms from this point on.
    monkeypatch.setattr(
        ProposalCoordinateTable, "standardized", property(forbidden)
    )
    result = proposal.promote_exact_stratum(
        candidates, observed, draw, scores, recipe=recipe
    )
    assert result.promotion_size == 6


def test_pilot_promotion_is_object_chunk_invariant_with_row_manifests():
    proposal, observed, candidates, draw, scores, recipe = (
        _pilot_promotion_fixture(n_objects=2)
    )
    together = proposal.promote_exact_stratum(
        candidates, observed, draw, scores, recipe=recipe
    )
    for row in range(2):
        rows = np.array([row])
        separate = proposal.promote_exact_stratum(
            candidates.take(rows),
            observed.iloc[[row]].reset_index(drop=True),
            draw.take(rows),
            scores[[row]],
            recipe=recipe,
        )
        np.testing.assert_array_equal(
            separate.candidates.indices[0], together.candidates.indices[row]
        )
        np.testing.assert_array_equal(
            separate.promoted_indices[0], together.promoted_indices[row]
        )
        np.testing.assert_array_equal(
            separate.anchor_indices[0], together.anchor_indices[row]
        )
        for name in (
            "object_pilot_draw_indices_sha256",
            "object_pilot_probability_sha256",
            "object_pilot_scores_sha256",
            "object_base_stratum_indices_sha256",
            "object_anchor_indices_sha256",
            "object_promoted_indices_sha256",
            "object_final_stratum_indices_sha256",
        ):
            assert getattr(separate.provenance, name)[0] == getattr(
                together.provenance, name
            )[row]
        assert (
            separate.provenance.call_final_stratum_indices_sha256
            != together.provenance.call_final_stratum_indices_sha256
        )


def test_pilot_promotion_manifests_are_canonical_for_stratum_membership():
    proposal, observed, candidates, draw, scores, recipe = (
        _pilot_promotion_fixture()
    )
    forward = proposal.promote_exact_stratum(
        candidates, observed, draw, scores, recipe=recipe
    )
    reversed_base = ProposalCandidates(
        candidates.indices[:, ::-1],
        candidates.distances[:, ::-1],
        candidates.radius,
    )
    reverse = proposal.promote_exact_stratum(
        reversed_base, observed, draw, scores, recipe=recipe
    )
    assert (
        forward.provenance.call_base_stratum_indices_sha256
        == reverse.provenance.call_base_stratum_indices_sha256
    )
    assert (
        forward.provenance.call_promoted_indices_sha256
        == reverse.provenance.call_promoted_indices_sha256
    )
    assert (
        forward.provenance.call_final_stratum_indices_sha256
        == reverse.provenance.call_final_stratum_indices_sha256
    )
    assert (
        forward.provenance.object_final_stratum_indices_sha256
        == reverse.provenance.object_final_stratum_indices_sha256
    )


def test_pilot_promotion_falls_back_only_for_insufficient_finite_anchors():
    proposal, observed, candidates, draw, scores, recipe = (
        _pilot_promotion_fixture()
    )
    scores[:] = -np.inf
    result = proposal.promote_exact_stratum(
        candidates, observed, draw, scores, recipe=recipe
    )
    np.testing.assert_array_equal(result.anchor_counts, [0])
    np.testing.assert_array_equal(result.fallback_fill_counts, [6])
    np.testing.assert_array_equal(result.max_neighbor_query_width, [0])
    np.testing.assert_array_equal(result.promoted_indices[0], [1, 2, 3, 4, 5, 6])


def test_pilot_promotion_validates_scores_support_conditioning_and_schedule():
    proposal, observed, candidates, draw, scores, recipe = (
        _pilot_promotion_fixture()
    )
    invalid_scores = scores.copy()
    invalid_scores[0, 0] = np.nan
    with pytest.raises(ValueError, match=r"NaN/\+inf"):
        proposal.promote_exact_stratum(
            candidates, observed, draw, invalid_scores, recipe=recipe
        )

    overlap = draw.indices.copy()
    overlap[0, 0] = candidates.indices[0, 0]
    with pytest.raises(ValueError, match="conditioned on the base-stratum"):
        proposal.promote_exact_stratum(
            candidates,
            observed,
            replace(draw, indices=overlap),
            scores,
            recipe=recipe,
        )

    inactive = draw.indices.copy()
    inactive[0, 0] = 99
    with pytest.raises(ValueError, match="only active atom ids"):
        proposal.promote_exact_stratum(
            candidates,
            observed,
            replace(draw, indices=inactive),
            scores,
            recipe=recipe,
        )

    bad_component = draw.component_id.copy()
    bad_component[0] = np.roll(bad_component[0], 1)
    with pytest.raises(ValueError, match="component schedule"):
        proposal.promote_exact_stratum(
            candidates,
            observed,
            replace(draw, component_id=bad_component),
            scores,
            recipe=recipe,
        )

    wrong_component_recipe = replace(
        recipe.pilot_proposal, name="wrong_component_recipe"
    )
    with pytest.raises(ValueError, match="component recipe"):
        proposal.promote_exact_stratum(
            candidates,
            observed,
            replace(draw, component_recipe=wrong_component_recipe),
            scores,
            recipe=recipe,
        )

    with pytest.raises(ValueError, match="execution chunks"):
        proposal.promote_exact_stratum(
            candidates,
            observed,
            replace(draw, atom_chunk=recipe.atom_chunk // 2),
            scores,
            recipe=recipe,
        )
    with pytest.raises(ValueError, match="execution chunks"):
        proposal.promote_exact_stratum(
            candidates,
            observed,
            replace(draw, object_chunk=recipe.object_chunk // 2),
            scores,
            recipe=recipe,
        )

    other_base = ProposalCandidates(
        indices=np.array([[1, 10]], dtype=np.int64),
        distances=np.zeros((1, 2), dtype=np.float64),
        radius=np.zeros(1, dtype=np.float64),
    )
    assert not np.isin(draw.indices, other_base.indices).any()
    with pytest.raises(ValueError, match="conditioned-exclusion support"):
        proposal.promote_exact_stratum(
            other_base, observed, draw, scores, recipe=recipe
        )

    duplicate_base = ProposalCandidates(
        np.array([[0, 0]]), np.zeros((1, 2)), np.zeros(1)
    )
    with pytest.raises(ValueError, match="must be unique"):
        proposal.promote_exact_stratum(
            duplicate_base, observed, draw, scores, recipe=recipe
        )


def test_pilot_promotion_requires_duplicate_balance_probability_agreement():
    proposal, observed, candidates, draw, scores, recipe = (
        _pilot_promotion_fixture()
    )
    close = draw.probability.copy()
    close[0, 1] *= 1.0 + 5.0e-13
    proposal.promote_exact_stratum(
        candidates,
        observed,
        replace(draw, probability=close),
        scores,
        recipe=recipe,
    )

    inconsistent = draw.probability.copy()
    inconsistent[0, 1] *= 1.0 + 5.0e-10
    with pytest.raises(ValueError, match="inconsistent balance probability"):
        proposal.promote_exact_stratum(
            candidates,
            observed,
            replace(draw, probability=inconsistent),
            scores,
            recipe=recipe,
        )


def test_pilot_promotion_full_support_is_an_exact_identity():
    proposal, observed, _, draw, scores, recipe = _pilot_promotion_fixture()
    active = proposal.active_indices[None, :]
    full = ProposalCandidates(
        active, np.zeros_like(active, dtype=np.float64), np.zeros(1)
    )
    full_draw = replace(
        draw,
        indices=np.zeros_like(draw.indices),
        probability=np.ones_like(draw.probability),
        local_member=np.ones_like(draw.local_member),
        local_position=np.zeros_like(draw.local_position),
        component_complement_mass=np.zeros_like(
            draw.component_complement_mass
        ),
        object_excluded_candidate_indices_sha256=(
            _candidate_exclusion_manifest_sha256(full.indices)
        ),
    )
    result = proposal.promote_exact_stratum(
        full, observed, full_draw, scores, recipe=recipe
    )
    np.testing.assert_array_equal(result.candidates.indices, full.indices)
    expected_distances = np.abs(np.arange(12, dtype=np.float64) - 5.0)[None, :]
    np.testing.assert_array_equal(result.candidates.distances, expected_distances)
    np.testing.assert_array_equal(result.candidates.radius, [6.0])
    assert result.promotion_size == 0
    assert result.anchor_indices.shape == (1, 0)
    np.testing.assert_array_equal(result.fallback_fill_counts, [0])
    np.testing.assert_array_equal(result.max_neighbor_query_width, [0])


def test_defensive_proposal_has_global_support_and_bounded_prior_ratio():
    likelihood = _likelihood()
    proposal = DefensiveLocalProposal(
        _coordinates(likelihood),
        likelihood.cache.prior.weights,
        local_base_weights=likelihood.cache.get(0.0, 0.0).detection_probability,
    )
    epsilon = 0.15
    draw = proposal.draw(
        pd.DataFrame({"measured_e1": [-0.2, 0.2]}),
        n_draws=200,
        n_candidates=2,
        epsilon=epsilon,
        bandwidth=0.7,
        seed=7,
    )
    prior = likelihood.cache.prior.weights[draw.indices]
    assert draw.indices.shape == (2, 200)
    assert np.all(draw.probability >= epsilon * prior)
    assert np.all(prior / draw.probability <= 1.0 / epsilon + 1e-12)
    assert (~draw.local_member).any()
    assert draw.global_component.any()


def test_defensive_proposal_draws_are_exact_prefixes_across_maximum_sizes():
    likelihood = _likelihood()
    proposal = DefensiveLocalProposal(
        _coordinates(likelihood),
        likelihood.cache.prior.weights,
        local_base_weights=likelihood.cache.get(0.0, 0.0).detection_probability,
    )
    observed = pd.DataFrame({"measured_e1": [-0.2, 0.0, 0.2]})
    short = proposal.draw(
        observed,
        n_draws=37,
        n_candidates=3,
        epsilon=0.15,
        bandwidth=0.7,
        seed=20260821,
    )
    long = proposal.draw(
        observed,
        n_draws=101,
        n_candidates=3,
        epsilon=0.15,
        bandwidth=0.7,
        seed=20260821,
    )

    np.testing.assert_array_equal(short.indices, long.indices[:, :37])
    np.testing.assert_array_equal(short.probability, long.probability[:, :37])
    np.testing.assert_array_equal(short.local_member, long.local_member[:, :37])
    np.testing.assert_array_equal(short.global_component, long.global_component[:, :37])
    np.testing.assert_array_equal(short.candidate_radius, long.candidate_radius)


def test_posterior_adapted_proposal_uses_exact_mixture_probability():
    likelihood = _likelihood()
    proposal = DefensiveLocalProposal(
        _coordinates(likelihood),
        likelihood.cache.prior.weights,
        local_base_weights=likelihood.cache.get(0.0, 0.0).detection_probability,
    )
    observed = pd.DataFrame({"measured_e1": [-0.15, 0.12]})
    candidates = proposal.candidates(observed, n_candidates=3)
    target = likelihood.log_importance_weights(
        observed,
        0.0,
        0.0,
        atom_indices=candidates.indices,
        proposal_probability=np.ones_like(candidates.indices, dtype=float),
    )
    draw = proposal.draw_adapted(
        candidates,
        target,
        n_draws=1000,
        epsilon=0.1,
        seed=20260822,
    )
    prior = likelihood.cache.prior.weights[draw.indices]
    assert np.all(draw.probability >= 0.1 * prior)
    assert np.all(prior / draw.probability <= 10.0 + 1e-12)
    assert draw.global_component.any()
    assert (~draw.global_component).any()
    assert np.all(draw.local_member[~draw.global_component])
    assert np.all(draw.local_position[draw.local_member] >= 0)
    rows = np.arange(len(draw.indices))[:, None]
    np.testing.assert_array_equal(
        candidates.indices[rows, draw.local_position.clip(min=0)][draw.local_member],
        draw.indices[draw.local_member],
    )


def test_posterior_adapted_draw_is_exact_across_object_chunks():
    likelihood = _likelihood()
    proposal = DefensiveLocalProposal(
        _coordinates(likelihood),
        likelihood.cache.prior.weights,
        local_base_weights=likelihood.cache.get(0.0, 0.0).detection_probability,
    )
    observed = pd.DataFrame({"measured_e1": [-0.22, -0.08, 0.0, 0.11, 0.24]})

    def adapted_draw(frame, *, object_offset):
        candidates = proposal.candidates(frame, n_candidates=3)
        target = likelihood.log_importance_weights(
            frame,
            0.0,
            0.0,
            atom_indices=candidates.indices,
            proposal_probability=np.ones_like(candidates.indices, dtype=float),
        )
        return proposal.draw_adapted(
            candidates,
            target,
            n_draws=257,
            epsilon=0.15,
            seed=20260824,
            object_offset=object_offset,
        )

    full = adapted_draw(observed, object_offset=0)
    chunks = [
        adapted_draw(observed.iloc[start:stop].reset_index(drop=True), object_offset=start)
        for start, stop in ((0, 2), (2, 3), (3, 5))
    ]

    for name in (
        "indices",
        "probability",
        "local_member",
        "global_component",
        "candidate_radius",
        "local_position",
    ):
        np.testing.assert_array_equal(
            getattr(full, name),
            np.concatenate([getattr(chunk, name) for chunk in chunks], axis=0),
        )


def test_posterior_adapted_subset_keeps_absolute_object_streams():
    likelihood = _likelihood()
    proposal = DefensiveLocalProposal(
        _coordinates(likelihood),
        likelihood.cache.prior.weights,
        local_base_weights=likelihood.cache.get(0.0, 0.0).detection_probability,
    )
    observed = pd.DataFrame({"measured_e1": [-0.22, -0.08, 0.0, 0.11, 0.24]})
    candidates = proposal.candidates(observed, n_candidates=3)
    target = likelihood.log_importance_weights(
        observed,
        0.0,
        0.0,
        atom_indices=candidates.indices,
        proposal_probability=np.ones_like(candidates.indices, dtype=float),
    )
    full = proposal.draw_adapted(
        candidates,
        target,
        n_draws=257,
        epsilon=0.15,
        seed=20260825,
    )
    rows = np.array([1, 4])
    subset = proposal.draw_adapted(
        candidates.take(rows),
        target[rows],
        n_draws=73,
        epsilon=0.15,
        seed=20260825,
        object_ids=rows,
    )

    expected = full.take(rows).prefix(73)
    for name in (
        "indices",
        "probability",
        "local_member",
        "global_component",
        "candidate_radius",
        "local_position",
    ):
        np.testing.assert_array_equal(getattr(subset, name), getattr(expected, name))


def test_global_prior_draws_are_nested_and_mark_candidate_membership():
    likelihood = _likelihood()
    proposal = DefensiveLocalProposal(_coordinates(likelihood), likelihood.cache.prior.weights)
    observed = pd.DataFrame({"measured_e1": [-0.15, 0.12]})
    candidates = proposal.candidates(observed, n_candidates=2)
    short = proposal.draw_global(candidates, n_draws=31, seed=207)
    long = proposal.draw_global(candidates, n_draws=79, seed=207)

    np.testing.assert_array_equal(short.indices, long.indices[:, :31])
    np.testing.assert_array_equal(short.probability, long.probability[:, :31])
    assert np.all(short.global_component)
    rows = np.arange(len(short.indices))[:, None]
    recovered = candidates.indices[rows, short.local_position.clip(min=0)]
    np.testing.assert_array_equal(recovered[short.local_member], short.indices[short.local_member])


def test_coalesced_draw_preserves_every_nested_importance_sum():
    likelihood = _likelihood()
    proposal = DefensiveLocalProposal(_coordinates(likelihood), likelihood.cache.prior.weights)
    draw = proposal.draw(
        pd.DataFrame({"measured_e1": [-0.1, 0.1]}),
        n_draws=64,
        n_candidates=3,
        epsilon=0.2,
        bandwidth=0.8,
        seed=20260823,
    )
    coalesced = draw.coalesce()
    rng = np.random.default_rng(91)
    unique_log_weight = rng.normal(size=coalesced.indices.shape)
    rows = np.arange(len(draw.indices))[:, None]
    draw_log_weight = unique_log_weight[rows, coalesced.inverse]
    for n_draws in (8, 32, 64):
        counts = coalesced.counts(n_draws)
        direct = logsumexp(draw_log_weight[:, :n_draws], axis=1)
        collapsed = logsumexp(
            np.where(
                counts > 0,
                unique_log_weight + np.log(np.maximum(counts, 1)),
                -np.inf,
            ),
            axis=1,
        )
        np.testing.assert_allclose(collapsed, direct, rtol=0, atol=1e-12)


def test_zero_mass_neighbour_rows_never_enter_proposal_draws():
    coordinates = ProposalCoordinateTable(
        values=np.array([[0.0], [1.0], [2.0]]),
        target_names=("measured_e1",),
        center=np.array([0.0]),
        scale=np.array([1.0]),
        statistic="mean",
    )
    proposal = DefensiveLocalProposal(coordinates, np.array([0.5, 0.0, 0.5]))
    draw = proposal.draw(
        pd.DataFrame({"measured_e1": [1.0]}),
        n_draws=100,
        n_candidates=1,
        epsilon=0.1,
        bandwidth=1.0,
        seed=9,
    )
    assert set(np.unique(draw.indices)) <= {0, 2}
    assert np.all(proposal.prior_weights[draw.indices] > 0)


def test_proposal_coordinate_cache_round_trip(tmp_path):
    likelihood = _likelihood()
    coordinates = ProposalCoordinateTable.from_flow(
        likelihood,
        target_names=("measured_e1",),
        n_flow_samples=8,
        statistic="median",
        row_chunk=2,
        seed=9,
        metadata={"flow": "fake"},
    )
    coordinates.save(tmp_path / "proposal")
    restored = ProposalCoordinateTable.load(tmp_path / "proposal")
    np.testing.assert_allclose(restored.values, coordinates.values)
    np.testing.assert_allclose(restored.center, coordinates.center)
    np.testing.assert_allclose(restored.scale, coordinates.scale)
    assert restored.target_names == ("measured_e1",)
    assert restored.metadata == {"flow": "fake"}
    assert restored.dispersion is not None
    assert restored.dispersion_statistic == "robust_iqr"
    np.testing.assert_allclose(restored.dispersion, coordinates.dispersion)


def test_gaussian_coordinate_cache_uses_mean_and_standard_deviation():
    likelihood = _likelihood()
    draws = np.array(
        [
            [[-1.0], [0.0], [1.0], [2.0]],
            [[2.0], [2.0], [4.0], [8.0]],
            [[-2.0], [-1.0], [1.0], [2.0]],
            [[3.0], [4.0], [5.0], [6.0]],
        ]
    )

    def sample(frame, n_samples=1, batch_size=None, qmc=False):
        assert n_samples == 4
        assert qmc
        rows = frame.index.to_numpy()
        return draws[rows]

    likelihood.flow_model.sample = sample
    coordinates = ProposalCoordinateTable.from_flow(
        likelihood,
        target_names=("measured_e1",),
        n_flow_samples=4,
        statistic="mean",
        dispersion_statistic="std",
        row_chunk=2,
        seed=9,
    )

    np.testing.assert_allclose(coordinates.values[:2, 0], draws.mean(axis=1)[:2, 0])
    np.testing.assert_allclose(coordinates.dispersion[:2, 0], draws.std(axis=1)[:2, 0])
    assert coordinates.dispersion_statistic == "std"


def test_version_two_proposal_cache_remains_loadable(tmp_path):
    root = tmp_path / "proposal"
    root.mkdir()
    np.savez_compressed(
        root / "coordinates.npz",
        values=np.array([[0.0], [1.0]]),
        center=np.array([0.5]),
        scale=np.array([1.0]),
    )
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "version": 2,
                "target_names": ["measured_e1"],
                "statistic": "median",
                "n_flow_samples": 8,
                "metadata": {},
            }
        )
    )
    restored = ProposalCoordinateTable.load(root)
    assert restored.dispersion is None
    assert restored.dispersion_statistic == "robust_iqr"


def test_uncertainty_reranking_can_recover_narrow_high_mass_atom():
    coordinates = ProposalCoordinateTable(
        values=np.array([[0.0], [0.1], [0.2], [0.3]]),
        target_names=("measured_e1",),
        center=np.array([0.0]),
        scale=np.array([1.0]),
        dispersion=np.array([[0.1], [0.1], [1.0], [0.1]]),
        statistic="mean",
    )
    proposal = DefensiveLocalProposal(
        coordinates,
        np.array([0.01, 0.01, 0.97, 0.01]),
    )
    observed = pd.DataFrame({"measured_e1": [0.0]})

    nearest = proposal.candidates(observed, n_candidates=1)
    reranked = proposal.candidates(observed, n_candidates=1, prefilter_candidates=4)
    torch_reranked = proposal.candidates(
        observed,
        n_candidates=1,
        prefilter_candidates=4,
        torch_device="cpu",
    )

    assert nearest.indices[0, 0] == 0
    assert reranked.indices[0, 0] == 2
    assert reranked.distances[0, 0] == 0.0
    np.testing.assert_array_equal(torch_reranked.indices, reranked.indices)
    np.testing.assert_allclose(torch_reranked.distances, reranked.distances)


def test_direct_uncertainty_mips_matches_brute_gaussian_ranking():
    coordinates = ProposalCoordinateTable(
        values=np.array([[0.0, 0.2], [0.1, -0.2], [0.3, 0.0], [-0.4, 0.1]]),
        target_names=("x", "y"),
        center=np.zeros(2),
        scale=np.ones(2),
        dispersion=np.array([[0.1, 0.4], [0.3, 0.2], [0.2, 0.5], [0.6, 0.1]]),
        statistic="mean",
    )
    prior = np.array([0.05, 0.15, 0.5, 0.3])
    detection = np.array([0.8, 0.4, 0.9, 0.7])
    proposal = DefensiveLocalProposal(coordinates, prior, local_base_weights=detection)
    observed = pd.DataFrame({"x": [0.12, -0.25], "y": [0.03, 0.18]})
    result = proposal.uncertainty_candidates(observed, n_candidates=4)

    values = observed[["x", "y"]].to_numpy()
    score = (
        np.log(proposal.local_base_weights)[None, :]
        - np.log(coordinates.dispersion).sum(axis=1)[None, :]
        - 0.5
        * np.square(
            (values[:, None, :] - coordinates.values[None, :, :]) / coordinates.dispersion[None, :, :]
        ).sum(axis=2)
    )
    expected = np.argsort(-score, axis=1)
    np.testing.assert_array_equal(result.indices, expected)

def test_importance_ladder_uses_nested_draws_and_approaches_exact_oracle():
    likelihood = _likelihood()
    mock = generate_mock_catalogue(
        likelihood,
        n_detected=400,
        g1=0.015,
        g2=0.0,
        scene_seed=20,
        detection_seed=21,
        flow_seed=22,
    )
    proposal = DefensiveLocalProposal(
        _coordinates(likelihood),
        likelihood.cache.prior.weights,
        local_base_weights=likelihood.cache.get(0.0, 0.0).detection_probability,
    )
    result = run_importance_ladder(
        likelihood,
        mock,
        proposal,
        ladder=(32, 128, 512),
        n_candidates=3,
        epsilon=0.2,
        bandwidth=0.8,
        proposal_seed=23,
        delta=0.005,
        richardson=False,
        compare_exact=True,
        object_chunk=200,
        atom_chunk=256,
    )
    assert [rung.n_draws for rung in result.rungs] == [32, 128, 512]
    assert result.exact_estimated_shear is not None
    assert abs(result.rungs[-1].shear_error_exact) < 0.02
    assert result.rungs[-1].diagnostics.mean_ess > 1
    assert 0 <= result.rungs[-1].diagnostics.mean_outside_local_contribution <= 1

    arms = [
        result,
        replace(result, proposal_seed=24),
        replace(result, n_candidates=4, proposal_seed=23),
        replace(result, n_candidates=4, proposal_seed=24),
    ]
    assessment = assess_importance_convergence(
        arms,
        max_exact_shear_error=0.03,
        max_rung_change=0.05,
        max_seed_spread=0.01,
        max_candidate_change=0.01,
        min_ess_fraction=0.001,
        max_p90_weight_fraction=1.0,
    )
    assert assessment.passed
    assert not assessment.failures
    assert all(type(value) is bool for value in assessment.checks.values())
    json.dumps(asdict(assessment))


def test_importance_diagnostics_identify_single_dominant_draw():
    likelihood = _likelihood()
    proposal = DefensiveLocalProposal(_coordinates(likelihood), likelihood.cache.prior.weights)
    draw = proposal.draw(
        pd.DataFrame({"measured_e1": [0.0]}),
        n_draws=4,
        n_candidates=2,
        epsilon=0.2,
        bandwidth=1.0,
        seed=31,
    )
    diagnostic = importance_diagnostics(np.array([[0.0, -100.0, -100.0, -100.0]]), draw)
    assert np.isclose(diagnostic.mean_ess, 1.0)
    assert np.isclose(diagnostic.p90_max_weight_fraction, 1.0)


def test_adaptive_draw_doubling_selects_first_stable_prefix_per_object():
    log_weights = np.array(
        [
            np.zeros(8),
            [0.0, -100.0, -100.0, -100.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, -100.0, -100.0, -100.0, -100.0, -100.0, -100.0, -100.0],
        ]
    )
    selected = select_adaptive_draw_counts(
        log_weights,
        (2, 4, 8),
        min_ess=2.0,
        max_weight_fraction=0.5,
    )
    np.testing.assert_array_equal(selected, [2, 8, 8])


def test_independent_pilot_projects_concentration_before_production_draw():
    pilot = np.array(
        [
            np.zeros(8),
            [0.0, 0.0, -100.0, -100.0, -100.0, -100.0, -100.0, -100.0],
            [0.0, -100.0, -100.0, -100.0, -100.0, -100.0, -100.0, -100.0],
        ]
    )
    selected = select_independent_pilot_draw_counts(
        pilot,
        (8, 16, 32),
        min_ess=8.0,
        max_weight_fraction=0.25,
    )
    np.testing.assert_array_equal(selected, [8, 32, 32])

    conservative = select_independent_pilot_draw_counts(
        pilot,
        (8, 16, 32),
        min_ess=8.0,
        max_weight_fraction=0.25,
        safety_factor=2.0,
    )
    np.testing.assert_array_equal(conservative, [16, 32, 32])


def test_importance_profile_recovers_small_catalogue_mle_and_releases_views():
    likelihood = _likelihood()
    mock = generate_mock_catalogue(
        likelihood,
        n_detected=120,
        g1=0.02,
        g2=0.0,
        scene_seed=40,
        detection_seed=41,
        flow_seed=42,
    )
    proposal = DefensiveLocalProposal(
        _coordinates(likelihood),
        likelihood.cache.prior.weights,
        local_base_weights=likelihood.cache.get(0.0, 0.0).detection_probability,
    )
    protected = set(likelihood.cache.available_shears)
    results = run_importance_profiles(
        likelihood,
        mock,
        proposal,
        shears=(-0.02, 0.0, 0.02, 0.04),
        ladder=(256, 1024),
        n_candidates=4,
        epsilon=0.2,
        bandwidth=0.8,
        proposal_seeds=(43, 44),
        object_chunk=120,
    )
    assert [result.proposal_seed for result in results] == [43, 44]
    assert all([rung.n_draws for rung in result.rungs] == [256, 1024] for result in results)
    assert all(abs(result.rungs[-1].estimated_shear - 0.02) < 0.03 for result in results)
    assert set(likelihood.cache.available_shears) == protected

def test_importance_profile_uses_measured_selection_normalization():
    base = _likelihood()
    selected = CatalogueLikelihood(
        base.flow_model,
        base.cache,
        selection=AnalyticGaussianSelection(upper=0.05),
    )
    mock = generate_mock_catalogue(
        selected,
        n_detected=40,
        g1=0.02,
        g2=0.0,
        scene_seed=45,
        detection_seed=46,
        flow_seed=47,
    )
    proposal = DefensiveLocalProposal(
        _coordinates(base),
        base.cache.prior.weights,
        local_base_weights=base.cache.get(0.0, 0.0).detection_probability,
    )
    kwargs = dict(
        shears=(0.0, 0.02),
        ladder=(128,),
        n_candidates=4,
        epsilon=0.2,
        bandwidth=0.8,
        proposal_seeds=(48,),
        object_chunk=40,
    )
    uncut_result = run_importance_profiles(base, mock, proposal, **kwargs)[0]
    selected_result = run_importance_profiles(selected, mock, proposal, **kwargs)[0]

    uncut_points = uncut_result.rungs[0].points
    selected_points = selected_result.rungs[0].points
    for uncut_point, selected_point in zip(uncut_points, selected_points):
        view = selected.cache.get(selected_point.shear, 0.0)
        detected_mass = selected.cache.prior.weights * view.detection_probability
        selected_mass = detected_mass * selected.selection_probability(selected_point.shear, 0.0)
        expected_shift = len(mock.measurements) * np.log(detected_mass.sum() / selected_mass.sum())
        assert selected_point.log_likelihood_sum - uncut_point.log_likelihood_sum == pytest.approx(
            expected_shift
        )


def test_importance_profile_populates_selection_cache():
    base = _likelihood()
    cut = OutputCut(["measured_e1"], bounds=[("measured_e1", None, 0.05)])
    selection = CatalogueSelection(cut, n_samples=16, seed=49, row_chunk=4)
    selected = CatalogueLikelihood(base.flow_model, base.cache, selection=selection)
    mock = generate_mock_catalogue(
        selected,
        n_detected=20,
        g1=0.02,
        g2=0.0,
        scene_seed=50,
        detection_seed=51,
        flow_seed=52,
    )
    proposal = DefensiveLocalProposal(_coordinates(base), base.cache.prior.weights)
    run_importance_profiles(
        selected,
        mock,
        proposal,
        shears=(0.0, 0.02),
        ladder=(32,),
        n_candidates=4,
        epsilon=0.2,
        bandwidth=0.8,
        proposal_seeds=(53,),
        object_chunk=20,
    )
    assert selection.available_shears == ((0.0, 0.0), (0.02, 0.0))


def test_stratified_draws_are_nested_and_pair_with_the_mixture_global_component():
    likelihood = _likelihood()
    proposal = DefensiveLocalProposal(
        _coordinates(likelihood),
        likelihood.cache.prior.weights,
        local_base_weights=likelihood.cache.get(0.0, 0.0).detection_probability,
    )
    observed = pd.DataFrame({"measured_e1": [-0.15, 0.12]})
    candidates = proposal.candidates(observed, n_candidates=2)
    short = proposal.draw_stratified(candidates, n_draws=31, seed=411)
    long = proposal.draw_stratified(candidates, n_draws=79, seed=411)

    np.testing.assert_array_equal(short.indices, long.indices[:, :31])
    np.testing.assert_array_equal(short.probability, long.probability[:, :31])
    np.testing.assert_array_equal(
        short.probability, likelihood.cache.prior.weights[short.indices]
    )

    # The complement draws are the mixture's own global-component draws, so a
    # paired screen differences the two estimators under common random numbers.
    log_target = np.log(
        np.maximum(likelihood.cache.prior.weights[candidates.indices], 1e-300)
    )
    mixture = proposal.draw_adapted(
        candidates, log_target, n_draws=79, epsilon=0.3, seed=411
    )
    assert mixture.global_component.any()
    np.testing.assert_array_equal(
        mixture.indices[mixture.global_component],
        long.indices[mixture.global_component],
    )


def test_tilted_draws_are_nested_and_report_their_own_probability():
    """The tilted complement must stay unbiased and stay paired.

    Unbiasedness of the complement estimator rests entirely on the reported
    ``probability`` being the true mixture probability of the atom drawn -- a
    mismatch would not raise, it would silently bias every ``A_i``.  The nested
    prefix is what lets one run report a whole ``M`` ladder, and reusing the
    uniform stream is what keeps a tilted arm paired with a prior-drawn one.
    """

    likelihood = _likelihood()
    base = _coordinates(likelihood)
    observed = pd.DataFrame({"measured_e1": [-0.15, 0.12]})
    delta = 0.25

    # The proxy is the reranker's score, so it needs the cached per-atom flow
    # dispersion; without it the mode must refuse rather than invent one.
    without = DefensiveLocalProposal(
        base,
        likelihood.cache.prior.weights,
        local_base_weights=likelihood.cache.get(0.0, 0.0).detection_probability,
    )
    with pytest.raises(ValueError, match="version-3 proposal cache"):
        without.draw_tilted(
            without.candidates(observed, n_candidates=2),
            observed,
            n_draws=4,
            delta=delta,
            seed=411,
        )

    rng = np.random.default_rng(20260829)
    coordinates = replace(
        base,
        dispersion=0.1 + 0.2 * rng.random(base.values.shape),
    )
    proposal = DefensiveLocalProposal(
        coordinates,
        likelihood.cache.prior.weights,
        local_base_weights=likelihood.cache.get(0.0, 0.0).detection_probability,
    )
    candidates = proposal.candidates(observed, n_candidates=2)
    short = proposal.draw_tilted(
        candidates, observed, n_draws=31, delta=delta, seed=411
    )
    long = proposal.draw_tilted(
        candidates, observed, n_draws=79, delta=delta, seed=411
    )
    np.testing.assert_array_equal(short.indices, long.indices[:, :31])
    np.testing.assert_array_equal(short.probability, long.probability[:, :31])

    proxy = proposal._tilted_proxy(None)
    values = proposal._observed_values(observed)
    for row in range(len(observed)):
        score = proxy.score(values[row]).numpy()
        finite = np.isfinite(score)
        tilted = np.zeros(len(score))
        tilted[finite] = np.exp(score[finite] - score[finite].max())
        tilted /= tilted.sum()
        mixture = (1.0 - delta) * tilted + delta * proposal.prior_weights[
            proposal.active_indices
        ]
        candidate_positions = np.searchsorted(
            proposal.active_indices, candidates.indices[row]
        )
        mixture[candidate_positions] = 0.0
        mixture /= mixture.sum()
        lookup = dict(zip(proposal.active_indices.tolist(), mixture.tolist()))
        expected = np.array([lookup[int(j)] for j in long.indices[row]])
        np.testing.assert_allclose(long.probability[row], expected, rtol=1e-12)
        assert not np.isin(long.indices[row], candidates.indices[row]).any()
    assert not long.local_member.any()

    # Flattening changes only the proposal, so the reported probability must
    # move with it and stay a normalized mixture.
    flattened = proposal.draw_tilted(
        candidates, observed, n_draws=79, delta=delta, seed=411, temperature=3.0
    )
    assert not np.allclose(flattened.probability, long.probability)
    assert (flattened.probability > 0).all()
    with pytest.raises(ValueError, match="temperature must be positive"):
        proposal.draw_tilted(
            candidates, observed, n_draws=4, delta=delta, seed=411, temperature=0.0
        )

    # Same uniform stream as the prior-drawn complement, so the two estimator
    # arms remain paired even though they map those uniforms to other atoms.
    prior_drawn = proposal.draw_stratified(candidates, n_draws=79, seed=411)
    assert prior_drawn.indices.shape == long.indices.shape
    assert not np.array_equal(prior_drawn.indices, long.indices)


def test_conditioned_dm_mis_reports_qbar_counts_nesting_and_split_streams():
    proposal, observed, candidates = _mis_fixture()
    recipe = TAIL_PROPOSAL_RECIPE
    assert recipe.component_names == (
        "proxy_t1",
        "proxy_t2",
        "prior",
    )
    assert recipe.component_counts == (4, 3, 1)
    assert recipe.cycle_length == 8
    assert recipe.proxy_temperatures == (1.0, 2.0)

    long = proposal.draw_tilted_mis(
        candidates,
        observed,
        n_draws=80,
        seed=20260830,
        atom_chunk=3,
    )
    short = proposal.draw_tilted_mis(
        candidates,
        observed,
        n_draws=16,
        seed=20260830,
        atom_chunk=3,
    )
    for name in (
        "indices",
        "probability",
        "local_member",
        "global_component",
        "local_position",
        "component_id",
    ):
        np.testing.assert_array_equal(getattr(short, name), getattr(long, name)[:, :16])
    np.testing.assert_array_equal(
        short.component_complement_mass, long.component_complement_mass
    )
    prefixed = long.prefix(16)
    np.testing.assert_array_equal(prefixed.component_id, short.component_id)
    expected_exclusion_hashes = _candidate_exclusion_manifest_sha256(
        candidates.indices
    )
    assert long.component_recipe == recipe
    assert long.atom_chunk == 3
    assert long.object_chunk == 32
    assert long.object_excluded_candidate_indices_sha256 == expected_exclusion_hashes
    assert prefixed.component_recipe == recipe
    assert prefixed.atom_chunk == long.atom_chunk
    assert prefixed.object_chunk == long.object_chunk
    assert (
        prefixed.object_excluded_candidate_indices_sha256
        == expected_exclusion_hashes
    )
    with pytest.raises(FrozenInstanceError):
        long.atom_chunk = 4

    for row in range(len(observed)):
        components, qbar, masses, complement = _dense_conditioned_components(
            proposal,
            observed.loc[row, ["x", "y"]].to_numpy(float),
            candidates.indices[row],
        )
        active_lookup = np.searchsorted(proposal.active_indices, long.indices[row])
        np.testing.assert_allclose(
            long.probability[row], qbar[active_lookup], rtol=2e-14, atol=1e-15
        )
        np.testing.assert_allclose(
            long.component_complement_mass[row], masses, rtol=2e-14, atol=1e-15
        )
        assert complement[active_lookup].all()
        assert not np.isin(long.indices[row], candidates.indices[row]).any()
        prior_c = components[-1]
        assert np.all(
            qbar[complement]
            >= recipe.component_weights[-1] * prior_c[complement]
        )
        for prefix in range(recipe.cycle_length, 81, recipe.cycle_length):
            counts = np.bincount(
                long.component_id[row, :prefix],
                minlength=len(recipe.component_names),
            )
            np.testing.assert_array_equal(
                counts,
                np.asarray(recipe.component_counts) * (prefix // recipe.cycle_length),
            )
    assert not long.local_member.any()
    np.testing.assert_array_equal(
        long.global_component,
        long.component_id == len(recipe.component_names) - 1,
    )

    rows = np.array([2, 0])
    subset = proposal.draw_tilted_mis(
        candidates.take(rows),
        observed.iloc[rows].reset_index(drop=True),
        n_draws=16,
        seed=20260830,
        object_ids=rows,
        atom_chunk=3,
    )
    expected = long.take(rows).prefix(16)
    for name in (
        "indices",
        "probability",
        "local_member",
        "global_component",
        "candidate_radius",
        "local_position",
        "component_id",
        "component_complement_mass",
    ):
        np.testing.assert_array_equal(getattr(subset, name), getattr(expected, name))
    assert subset.component_recipe == expected.component_recipe == recipe
    assert subset.atom_chunk == expected.atom_chunk == 3
    assert subset.object_chunk == expected.object_chunk == 32
    assert (
        subset.object_excluded_candidate_indices_sha256
        == expected.object_excluded_candidate_indices_sha256
        == tuple(expected_exclusion_hashes[int(row)] for row in rows)
    )


def test_conditioned_dm_mis_batched_matches_scalar_constructor_and_object_batches():
    proposal, observed, candidates = _mis_fixture()
    recipe = TAIL_PROPOSAL_RECIPE
    n_draws = 80
    schedule = np.tile(
        np.asarray(recipe.component_cycle, dtype=np.int16),
        n_draws // recipe.cycle_length,
    )
    uniforms = []
    scalar = []
    proxy = proposal._tilted_proxy(None)
    values = proposal._observed_values(observed)
    for row in range(len(observed)):
        rng = np.random.default_rng(
            np.random.SeedSequence([20260830, row])
        )
        atom_uniforms = rng.random((n_draws, 3))[:, 1]
        uniforms.append(atom_uniforms)
        scalar.append(
            proxy.draw_conditioned_mis_uniforms(
                values[row],
                atom_uniforms,
                schedule,
                candidates.indices[row],
                recipe=recipe,
                atom_chunk=3,
            )
        )
    batched = proxy.draw_conditioned_mis_uniforms_batched(
        values,
        np.stack(uniforms),
        np.broadcast_to(schedule, (len(observed), n_draws)),
        candidates.indices,
        recipe=recipe,
        atom_chunk=3,
    )
    np.testing.assert_array_equal(
        batched[0], np.stack([result[0] for result in scalar])
    )
    np.testing.assert_allclose(
        batched[1],
        np.stack([result[1] for result in scalar]),
        rtol=5e-15,
        atol=1e-15,
    )
    np.testing.assert_allclose(
        batched[2],
        np.stack([result[2] for result in scalar]),
        rtol=5e-15,
        atol=1e-15,
    )

    one = proposal.draw_tilted_mis(
        candidates,
        observed,
        n_draws=n_draws,
        seed=20260830,
        atom_chunk=3,
        object_chunk=1,
    )
    two = proposal.draw_tilted_mis(
        candidates,
        observed,
        n_draws=n_draws,
        seed=20260830,
        atom_chunk=3,
        object_chunk=2,
    )
    all_rows = proposal.draw_tilted_mis(
        candidates,
        observed,
        n_draws=n_draws,
        seed=20260830,
        atom_chunk=3,
        object_chunk=32,
    )
    for candidate in (one, two):
        np.testing.assert_array_equal(candidate.indices, all_rows.indices)
        np.testing.assert_array_equal(candidate.component_id, all_rows.component_id)
        np.testing.assert_allclose(
            candidate.probability, all_rows.probability, rtol=5e-15, atol=1e-15
        )
        np.testing.assert_allclose(
            candidate.component_complement_mass,
            all_rows.component_complement_mass,
            rtol=5e-15,
            atol=1e-15,
        )
        assert candidate.component_recipe == all_rows.component_recipe == recipe
        assert candidate.atom_chunk == all_rows.atom_chunk == 3
        assert (
            candidate.object_excluded_candidate_indices_sha256
            == all_rows.object_excluded_candidate_indices_sha256
        )
    assert one.object_chunk == 1
    assert two.object_chunk == 2
    assert all_rows.object_chunk == 32


def test_conditioned_dm_mis_temperature_preserves_detected_prior_base():
    proposal, observed, _ = _mis_fixture()
    proxy = proposal._tilted_proxy(None)
    value = observed.loc[0, ["x", "y"]].to_numpy(float)
    likelihood = proxy.proxy_likelihood(value).cpu().numpy()
    log_base = proxy.log_base.cpu().numpy()
    score_t1, score_t2 = [
        value.cpu().numpy()
        for value in proxy.component_log_scores(value, TAIL_PROPOSAL_RECIPE)
    ]
    left, right = 0, 4
    np.testing.assert_allclose(score_t1, log_base + likelihood, rtol=0, atol=1e-14)
    np.testing.assert_allclose(score_t2, log_base + 0.5 * likelihood, rtol=0, atol=1e-14)
    expected_difference = (
        log_base[left] - log_base[right]
        + 0.5 * (likelihood[left] - likelihood[right])
    )
    assert score_t2[left] - score_t2[right] == pytest.approx(expected_difference)
    assert not np.isclose(
        score_t2[left] - score_t2[right],
        0.5 * (score_t1[left] - score_t1[right]),
    )


def test_conditioned_dm_mis_is_unbiased_and_dominates_iid_qbar_variance():
    proposal, observed, candidates = _mis_fixture()
    components, qbar, _, complement = _dense_conditioned_components(
        proposal,
        observed.loc[1, ["x", "y"]].to_numpy(float),
        candidates.indices[1],
    )
    target = np.zeros_like(qbar)
    target[complement] = np.array([0.03, 0.4, 0.08, 0.27, 0.12, 0.2])
    importance_value = np.divide(
        target, qbar, out=np.zeros_like(target), where=qbar > 0
    )
    alpha = np.asarray(TAIL_PROPOSAL_RECIPE.component_weights)
    component_mean = (components * importance_value).sum(axis=1)
    exact_dm_mean = alpha @ component_mean
    np.testing.assert_allclose(exact_dm_mean, target.sum(), rtol=0, atol=2e-15)

    component_variance = (
        components * np.square(importance_value[None, :] - component_mean[:, None])
    ).sum(axis=1)
    mixture_mean = (qbar * importance_value).sum()
    mixture_variance = (
        qbar * np.square(importance_value - mixture_mean)
    ).sum()
    n_draws = 80
    dm_variance = alpha @ component_variance / n_draws
    iid_variance = mixture_variance / n_draws
    between_component = (
        alpha * np.square(component_mean - alpha @ component_mean)
    ).sum() / n_draws
    assert dm_variance <= iid_variance + 1e-15
    np.testing.assert_allclose(
        iid_variance - dm_variance, between_component, rtol=0, atol=2e-15
    )


def test_conditioned_dm_mis_empirical_sources_and_estimate_match_dense_oracle():
    proposal, observed, candidates = _mis_fixture()
    row = 1
    n_draws = 32000
    draw = proposal.draw_tilted_mis(
        candidates.take(np.array([row])),
        observed.iloc[[row]].reset_index(drop=True),
        n_draws=n_draws,
        seed=8127,
        atom_chunk=3,
        object_chunk=1,
    )
    components, qbar, _, complement = _dense_conditioned_components(
        proposal,
        observed.loc[row, ["x", "y"]].to_numpy(float),
        candidates.indices[row],
    )
    positions = np.searchsorted(proposal.active_indices, draw.indices[0])
    for component in range(len(TAIL_PROPOSAL_RECIPE.component_names)):
        source = draw.component_id[0] == component
        empirical = np.bincount(
            positions[source], minlength=len(proposal.active_indices)
        ) / np.sum(source)
        np.testing.assert_allclose(
            empirical[complement], components[component, complement], atol=0.025
        )
        np.testing.assert_array_equal(empirical[~complement], 0.0)

    target = np.zeros_like(qbar)
    target[complement] = np.array([0.03, 0.4, 0.08, 0.27, 0.12, 0.2])
    estimate = np.mean(target[positions] / draw.probability[0])
    component_mean = (components * np.divide(
        target,
        qbar,
        out=np.zeros_like(target),
        where=qbar > 0,
    )[None, :]).sum(axis=1)
    component_variance = (
        components
        * np.square(
            np.divide(
                target,
                qbar,
                out=np.zeros_like(target),
                where=qbar > 0,
            )[None, :]
            - component_mean[:, None]
        )
    ).sum(axis=1)
    standard_error = np.sqrt(
        np.asarray(TAIL_PROPOSAL_RECIPE.component_weights)
        @ component_variance
        / n_draws
    )
    assert abs(estimate - target.sum()) < 5.0 * standard_error


def test_conditioned_dm_mis_fallback_is_per_object_and_component():
    coordinates = ProposalCoordinateTable(
        values=np.array([[0.0], [0.2], [1.0], [1.2]]),
        target_names=("x",),
        center=np.zeros(1),
        scale=np.ones(1),
        dispersion=np.ones((4, 1)),
        statistic="mean",
    )
    proposal = DefensiveLocalProposal(
        coordinates,
        np.full(4, 0.25),
        local_base_weights=np.array([1.0, 1.0, 0.0, 0.0]),
    )
    candidates = ProposalCandidates(
        np.array([[0, 1], [2, 3]]),
        np.zeros((2, 2)),
        np.ones(2),
    )
    observed = pd.DataFrame({"x": [0.1, 0.1]})
    draw = proposal.draw_tilted_mis(
        candidates,
        observed,
        n_draws=800,
        seed=992,
        atom_chunk=2,
        object_chunk=2,
    )
    np.testing.assert_array_equal(draw.component_complement_mass[0, :-1], 0.0)
    assert np.all(draw.component_complement_mass[1, :-1] > 0.0)
    assert np.isin(draw.indices[0], [2, 3]).all()
    assert np.isin(draw.indices[1], [0, 1]).all()
    np.testing.assert_allclose(draw.probability[0], 0.5, atol=1e-15)


def test_conditioned_dm_mis_normalizes_proxy_components_after_underflow():
    coordinates = ProposalCoordinateTable(
        values=np.array([[0.0], [0.1], [100.0], [101.0]]),
        target_names=("x",),
        center=np.zeros(1),
        scale=np.ones(1),
        dispersion=np.ones((4, 1)),
        statistic="mean",
    )
    proposal = DefensiveLocalProposal(coordinates, np.full(4, 0.25))
    observed = pd.DataFrame({"x": [0.0]})
    candidates = ProposalCandidates(
        np.array([[0, 1]]), np.zeros((1, 2)), np.ones(1)
    )
    draw = proposal.draw_tilted_mis(
        candidates, observed, n_draws=80, seed=193, atom_chunk=1
    )
    components, qbar, masses, _ = _dense_conditioned_components(
        proposal, np.array([0.0]), np.array([0, 1])
    )
    assert np.all(masses[:-1] == 0.0)
    np.testing.assert_array_equal(draw.component_complement_mass[0, :2], [0.0, 0.0])
    positions = np.searchsorted(proposal.active_indices, draw.indices[0])
    np.testing.assert_allclose(draw.probability[0], qbar[positions], atol=1e-15)
    # Separate log-domain conditioning keeps the proxy concentrated on atom 2;
    # falling back merely because the diagnostic mass underflowed would give .5.
    assert components[0, 2] > 0.999999999999
    assert qbar[2] > 0.9


def test_conditioned_dm_mis_rejects_nonfinite_proxy_normalizers():
    coordinates = ProposalCoordinateTable(
        values=np.array([[0.0], [1.0], [2.0]]),
        target_names=("x",),
        center=np.zeros(1),
        scale=np.ones(1),
        dispersion=np.full((3, 1), 1.0e-200),
        statistic="mean",
    )
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        proposal = DefensiveLocalProposal(coordinates, np.full(3, 1.0 / 3.0))
        with pytest.raises(RuntimeError, match="NaN or positive-infinite normalizer"):
            proposal.draw_tilted_mis(
                ProposalCandidates(
                    np.array([[0]]), np.zeros((1, 1)), np.ones(1)
                ),
                pd.DataFrame({"x": [0.5]}),
                n_draws=16,
                seed=7,
                atom_chunk=1,
            )


def test_conditioned_dm_mis_blocking_and_stencil_streams_are_invariant():
    proposal, observed, candidates = _mis_fixture()
    blocked = proposal.draw_tilted_mis(
        candidates, observed, n_draws=80, seed=917, atom_chunk=1
    )
    dense_block = proposal.draw_tilted_mis(
        candidates, observed, n_draws=80, seed=917, atom_chunk=1000
    )
    np.testing.assert_array_equal(blocked.indices, dense_block.indices)
    np.testing.assert_allclose(
        blocked.probability, dense_block.probability, rtol=2e-14, atol=1e-15
    )
    np.testing.assert_allclose(
        blocked.component_complement_mass,
        dense_block.component_complement_mass,
        rtol=2e-14,
        atol=1e-15,
    )

    # Reusing one absolute object stream at two stencil views keeps the fixed
    # source schedule and pi_C component exactly paired.
    same_candidates = candidates.take(np.array([0]))
    view_a = proposal.draw_tilted_mis(
        same_candidates,
        observed.iloc[[0]],
        n_draws=80,
        seed=917,
        object_ids=np.array([44]),
        atom_chunk=3,
    )
    view_b = proposal.draw_tilted_mis(
        same_candidates,
        observed.iloc[[2]].reset_index(drop=True),
        n_draws=80,
        seed=917,
        object_ids=np.array([44]),
        atom_chunk=3,
    )
    np.testing.assert_array_equal(view_a.component_id, view_b.component_id)
    prior_source = (
        view_a.component_id == len(TAIL_PROPOSAL_RECIPE.component_names) - 1
    )
    np.testing.assert_array_equal(
        view_a.indices[prior_source], view_b.indices[prior_source]
    )


def test_conditioned_dm_mis_validates_recipe_support_and_full_support_sentinel():
    proposal, observed, candidates = _mis_fixture()
    with pytest.raises(ValueError, match="positive multiple of 8"):
        proposal.draw_tilted_mis(
            candidates, observed, n_draws=7, seed=1
        )
    with pytest.raises(ValueError, match="atom_chunk must be positive"):
        proposal.draw_tilted_mis(
            candidates, observed, n_draws=8, seed=1, atom_chunk=0
        )
    with pytest.raises(ValueError, match="object_chunk must be positive"):
        proposal.draw_tilted_mis(
            candidates, observed, n_draws=8, seed=1, object_chunk=0
        )
    duplicate = ProposalCandidates(
        np.array([[0, 0]]), np.zeros((1, 2)), np.ones(1)
    )
    with pytest.raises(ValueError, match="must be unique"):
        proposal.draw_tilted_mis(
            duplicate, observed.iloc[[0]], n_draws=8, seed=1
        )
    inactive = ProposalCandidates(
        np.array([[0, 8]]), np.zeros((1, 2)), np.ones(1)
    )
    with pytest.raises(ValueError, match="active support"):
        proposal.draw_tilted_mis(
            inactive, observed.iloc[[0]], n_draws=8, seed=1
        )
    with pytest.raises(ValueError, match="cycle frequencies"):
        TailProposalRecipe(
            name="invalid",
            component_names=("a", "b", "prior"),
            component_weights=(0.5, 0.375, 0.125),
            component_betas=(1.0, 0.5, None),
            component_cycle=(0, 1, 2),
        )
    with pytest.raises(ValueError, match="final.*named prior"):
        TailProposalRecipe(
            name="invalid",
            component_names=("proxy", "defense"),
            component_weights=(0.5, 0.5),
            component_betas=(1.0, None),
            component_cycle=(0, 1),
        )
    with pytest.raises(ValueError, match="prior-times-detection"):
        replace(TAIL_PROPOSAL_RECIPE, base_measure="prior")

    without_dispersion = DefensiveLocalProposal(
        replace(proposal.coordinates, dispersion=None), proposal.prior_weights
    )
    with pytest.raises(ValueError, match="version-3 proposal cache"):
        without_dispersion.draw_tilted_mis(
            candidates.take(np.array([0])),
            observed.iloc[[0]],
            n_draws=8,
            seed=1,
        )

    active = proposal.active_indices
    full = ProposalCandidates(
        active[None, :], np.zeros((1, len(active))), np.ones(1)
    )
    sentinel = proposal.draw_tilted_mis(
        full, observed.iloc[[0]], n_draws=16, seed=31, atom_chunk=2
    )
    np.testing.assert_array_equal(sentinel.indices, active[0])
    np.testing.assert_array_equal(sentinel.probability, 1.0)
    assert sentinel.local_member.all()
    assert (sentinel.local_position == 0).all()
    np.testing.assert_array_equal(sentinel.component_complement_mass, 0.0)
    np.testing.assert_array_equal(
        sentinel.component_id[0, :8], TAIL_PROPOSAL_RECIPE.component_cycle
    )
    prefix = sentinel.prefix(5)
    np.testing.assert_array_equal(prefix.component_id, sentinel.component_id[:, :5])
    np.testing.assert_array_equal(
        prefix.component_complement_mass, sentinel.component_complement_mass
    )


def test_pareto_tail_index_recovers_a_known_generalized_pareto_shape():
    """The k-hat fit must be accurate where it is used to make decisions.

    The reliability threshold sits at 0.7 and the finite-variance boundary at
    0.5, so a bias of a few hundredths there would change conclusions.  Draws
    come from the exact inverse CDF of a unit-scale generalized Pareto.
    """

    rng = np.random.default_rng(20260829)
    for shape in (-0.2, 0.3, 0.5, 0.7, 1.0):
        uniform = rng.random((12, 20000))
        draws = ((1.0 - uniform) ** (-shape) - 1.0) / shape
        khat = pareto_tail_index(draws)
        assert np.isfinite(khat).all()
        assert abs(float(np.mean(khat)) - shape) < 0.05

    # Rescaling a row cannot change its tail index.
    uniform = rng.random((4, 8000))
    draws = ((1.0 - uniform) ** (-0.6) - 1.0) / 0.6
    scaled = pareto_tail_index(draws * 1.0e7)
    assert np.allclose(pareto_tail_index(draws), scaled, atol=1e-9)


def test_pareto_tail_index_reports_undefined_rows_rather_than_a_number():
    # No spread above the threshold, and too few draws to place one.
    flat = pareto_tail_index(np.ones((2, 4000)))
    assert np.isnan(flat).all()
    assert np.isnan(pareto_tail_index(np.arange(8.0).reshape(2, 4))).all()

    mixed = np.vstack((np.ones(4000), np.exp(np.random.default_rng(3).normal(size=4000))))
    index = pareto_tail_index(mixed)
    assert np.isnan(index[0]) and np.isfinite(index[1])

    with pytest.raises(ValueError):
        pareto_tail_index(np.array([1.0, 2.0]))
    with pytest.raises(ValueError):
        pareto_tail_index(np.array([[1.0, -2.0]]))


def test_pareto_tail_index_refuses_a_tied_discrete_tail():
    """A weight vector with few distinct values has no fittable tail.

    This is not hypothetical: a small catalogue produces exactly this, and an
    unguarded fit divides by a zero quartile and returns a large meaningless
    shape rather than declining to answer.
    """

    rng = np.random.default_rng(77)
    tied = rng.integers(0, 4, size=(6, 4000)).astype(np.float64) + 1.0
    assert np.isnan(pareto_tail_index(tied)).all()

    # One strictly larger value on top of a tied bulk is still not a tail.
    almost = np.ones((1, 4000))
    almost[0, -1] = 5.0
    assert np.isnan(pareto_tail_index(almost)).all()
