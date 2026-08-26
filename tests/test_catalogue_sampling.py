from dataclasses import asdict, replace
import json

import numpy as np
import pandas as pd
import pytest
from scipy.special import logsumexp

from sbsi.catalogue_closure import generate_mock_catalogue
from sbsi.catalogue_likelihood import CatalogueLikelihood, CatalogueSelection
from sbsi.catalogue_sampling import (
    DefensiveLocalProposal,
    ProposalCoordinateTable,
    assess_importance_convergence,
    candidate_support_diagnostics,
    importance_diagnostics,
    run_importance_curvature_scan,
    run_importance_ladder,
    run_importance_profiles,
    select_adaptive_draw_counts,
    select_independent_pilot_draw_counts,
)
from sbsi.score_inference import OutputCut
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


def test_candidate_support_diagnostic_separates_ranking_from_support_width():
    target = np.log(
        np.array(
            [
                [0.05, 0.05, 0.80, 0.10],
                [0.40, 0.30, 0.20, 0.10],
            ]
        )
    )
    result = candidate_support_diagnostics(target, (1, 2, 4))

    np.testing.assert_allclose(
        result.distance_prefix_mass,
        [[0.05, 0.10, 1.0], [0.40, 0.70, 1.0]],
    )
    np.testing.assert_allclose(
        result.optimal_prefix_mass,
        [[0.80, 0.90, 1.0], [0.40, 0.70, 1.0]],
    )
    np.testing.assert_allclose(result.reference_max_mass_fraction, [0.80, 0.40])
    assert np.all(result.reference_effective_atoms > 1)


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


def test_curvature_scan_decomposes_marginal_information_and_true_atom_curve():
    likelihood = _likelihood()
    mock = generate_mock_catalogue(
        likelihood,
        n_detected=32,
        g1=0.02,
        g2=0.0,
        scene_seed=140,
        detection_seed=141,
        flow_seed=142,
    )
    proposal = DefensiveLocalProposal(
        _coordinates(likelihood),
        likelihood.cache.prior.weights,
        local_base_weights=likelihood.cache.get(0.0, 0.0).detection_probability,
    )
    protected = set(likelihood.cache.available_shears)
    result = run_importance_curvature_scan(
        likelihood,
        mock,
        proposal,
        centers=(0.0, 0.02),
        h=0.0025,
        profile_shears=(-0.005, 0.005),
        ladder=(256, 1024),
        n_candidates=4,
        epsilon=0.2,
        bandwidth=0.8,
        proposal_seed=143,
        truth_atom_indices=mock.truth["scene_row"].to_numpy(np.int64),
        object_chunk=32,
        decomposition_chunk=16,
    )
    assert result.shears == (-0.005, -0.0025, 0.0, 0.0025, 0.005, 0.0175, 0.02, 0.0225)
    assert result.true_atom_log_likelihood.shape == (len(result.shears), 32)
    final = result.rungs[-1]
    assert final.marginalized_information.shape == (2, 32)
    assert final.true_atom_information.shape == (2, 32)
    # The identity is exact for analytic derivatives; applying the same
    # three-point finite difference to both sides leaves an O(h^2) residual.
    np.testing.assert_allclose(final.decomposition_residual, 0.0, atol=0.01)
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
