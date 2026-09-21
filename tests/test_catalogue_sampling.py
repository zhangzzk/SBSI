from dataclasses import asdict, replace
import warnings
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


def test_whole_catalogue_proxy_candidates_match_brute_gaussian_ranking():
    """The direct shortlist ranks the pure proxy, without the prior floor."""

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
    result = proposal.whole_catalogue_proxy_candidates(
        observed, n_candidates=3, device="cpu", object_chunk=1
    )

    values = observed[["x", "y"]].to_numpy()
    score = (
        np.log(proposal.local_base_weights)[None, :]
        - np.log(coordinates.dispersion).sum(axis=1)[None, :]
        - 0.5
        * np.square(
            (values[:, None, :] - coordinates.values[None, :, :])
            / coordinates.dispersion[None, :, :]
        ).sum(axis=2)
    )
    expected = np.argsort(-score, axis=1)[:, :3]
    np.testing.assert_array_equal(result.indices, expected)
    np.testing.assert_allclose(
        result.distances,
        score.max(axis=1)[:, None] - np.take_along_axis(score, expected, axis=1),
    )
    np.testing.assert_allclose(result.radius, result.distances[:, -1])


def test_fused_whole_catalogue_candidates_and_tilted_draw_match_two_pass_path():
    likelihood = _likelihood()
    base = _coordinates(likelihood)
    rng = np.random.default_rng(20260831)
    coordinates = replace(
        base, dispersion=0.1 + 0.2 * rng.random(base.values.shape)
    )
    proposal = DefensiveLocalProposal(
        coordinates,
        likelihood.cache.prior.weights,
        local_base_weights=likelihood.cache.get(0.0, 0.0).detection_probability,
    )
    observed = pd.DataFrame({"measured_e1": [-0.15, 0.12, 0.03]})
    separate_candidates = proposal.whole_catalogue_proxy_candidates(
        observed, n_candidates=3, device="cpu", object_chunk=2
    )
    separate_draw = proposal.draw_tilted(
        separate_candidates,
        observed,
        n_draws=37,
        delta=0.25,
        seed=99,
        device="cpu",
        object_chunk=2,
    )
    fused_candidates, fused_draw = (
        proposal.whole_catalogue_proxy_candidates_and_tilted_draw(
            observed,
            n_candidates=3,
            n_draws=37,
            delta=0.25,
            seed=99,
            device="cpu",
            object_chunk=2,
        )
    )
    np.testing.assert_array_equal(fused_candidates.indices, separate_candidates.indices)
    np.testing.assert_allclose(fused_candidates.distances, separate_candidates.distances)
    np.testing.assert_array_equal(fused_draw.indices, separate_draw.indices)
    np.testing.assert_array_equal(fused_draw.local_member, separate_draw.local_member)
    np.testing.assert_array_equal(fused_draw.local_position, separate_draw.local_position)
    np.testing.assert_allclose(fused_draw.probability, separate_draw.probability, rtol=1e-12)

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
        lookup = dict(zip(proposal.active_indices.tolist(), mixture.tolist()))
        expected = np.array([lookup[int(j)] for j in long.indices[row]])
        np.testing.assert_allclose(long.probability[row], expected, rtol=1e-12)

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


def test_top_atoms_are_the_largest_proposal_mass_and_need_no_draws():
    """The extended exact stratum must be exactly the heaviest atoms.

    Moving atoms into the exact sum is only variance-free if the atoms chosen
    are the ones the proposal actually concentrates on, and only unbiased if
    the choice never looks at the draws.  Both are checked here against a
    brute-force mixture.
    """

    likelihood = _likelihood()
    base = _coordinates(likelihood)
    observed = pd.DataFrame({"measured_e1": [-0.15, 0.12]})
    delta = 0.25

    rng = np.random.default_rng(20260830)
    coordinates = replace(
        base, dispersion=0.1 + 0.2 * rng.random(base.values.shape)
    )
    proposal = DefensiveLocalProposal(
        coordinates,
        likelihood.cache.prior.weights,
        local_base_weights=likelihood.cache.get(0.0, 0.0).detection_probability,
    )
    proxy = proposal._tilted_proxy(None)
    values = proposal._observed_values(observed)

    assert proxy.top_atoms(values[0], 0, delta=delta).size == 0

    for row in range(len(observed)):
        score = proxy.score(values[row]).numpy()
        finite = np.isfinite(score)
        tilted = np.zeros(len(score))
        tilted[finite] = np.exp(score[finite] - score[finite].max())
        tilted /= tilted.sum()
        mixture = (1.0 - delta) * tilted + delta * proposal.prior_weights[
            proposal.active_indices
        ]
        for count in (1, 3):
            expected = proposal.active_indices[
                np.argsort(mixture)[::-1][:count]
            ]
            got = proxy.top_atoms(values[row], count, delta=delta)
            assert sorted(got.tolist()) == sorted(expected.tolist())

    with pytest.raises(ValueError, match="temperature must be positive"):
        proxy.top_atoms(values[0], 2, delta=delta, temperature=0.0)


def test_batched_tilted_draws_match_the_single_row_path():
    """Batching is a speed change and must not be a numerical one.

    cont.328 batches the whole-catalogue mixture across observations to remove
    a step that was 78% of the estimator's wall clock.  Reassociating the
    arithmetic across rows perturbs the reported probabilities at the 1e-15
    level, but the atoms drawn must be identical -- a changed atom is a changed
    estimate, not a rounding difference.
    """

    likelihood = _likelihood()
    base = _coordinates(likelihood)
    observed = pd.DataFrame({"measured_e1": [-0.15, 0.12, 0.03, -0.28, 0.19]})
    delta = 0.25

    rng = np.random.default_rng(20260830)
    coordinates = replace(
        base, dispersion=0.1 + 0.2 * rng.random(base.values.shape)
    )
    proposal = DefensiveLocalProposal(
        coordinates,
        likelihood.cache.prior.weights,
        local_base_weights=likelihood.cache.get(0.0, 0.0).detection_probability,
    )
    candidates = proposal.candidates(observed, n_candidates=2)

    one = proposal.draw_tilted(
        candidates, observed, n_draws=37, delta=delta, seed=99, object_chunk=1
    )
    for chunk in (2, 3, 5, 64):
        many = proposal.draw_tilted(
            candidates,
            observed,
            n_draws=37,
            delta=delta,
            seed=99,
            object_chunk=chunk,
        )
        np.testing.assert_array_equal(one.indices, many.indices)
        np.testing.assert_array_equal(one.local_member, many.local_member)
        np.testing.assert_array_equal(one.local_position, many.local_position)
        np.testing.assert_allclose(
            one.probability, many.probability, rtol=1e-12, atol=0.0
        )

    with pytest.raises(ValueError, match="object_chunk must be positive"):
        proposal.draw_tilted(
            candidates, observed, n_draws=4, delta=delta, seed=99, object_chunk=0
        )


def _floor_catalogue(n=4096, seed=17):
    """Atoms spanning five decades in flux, with a brightness-dependent scatter.

    Mirrors the V3.6 table's structure without touching it: a bright atom's
    predicted measurement is sharp in every coordinate, a faint one's is vague,
    and the flux scatter is a fixed fraction of the flux so that it spans the
    same five decades the flux does.
    """
    rng = np.random.default_rng(seed)
    magnitude = rng.uniform(17.0, 28.0, size=n)
    flux = 10.0 ** (-0.4 * (magnitude - 30.0))
    faintness = np.clip((magnitude - 17.0) / 11.0, 0.0, 1.0)
    values = np.column_stack([
        rng.normal(0.0, 0.3, size=n),
        np.full(n, 4.0) + rng.normal(0.0, 0.5, size=n),
        flux,
    ])
    dispersion = np.column_stack([
        0.02 + 0.35 * faintness,
        0.2 + 1.2 * faintness,
        flux * (0.02 + 1.2 * faintness),
    ])
    return values, dispersion, magnitude


def test_fractional_floor_binds_on_bright_atoms_where_an_absolute_floor_cannot():
    """The defect the fractional floor exists to remove.

    Flux spans five decades, so the percentile of the *absolute* flux scatter
    is set by the faintest atoms and leaves a bright atom's tolerance untouched
    however high the percentile is raised.  Ranking the same atoms on their
    fractional scatter binds at every brightness.
    """
    from sbsi.catalogue_sampling import floored_dispersion

    values, dispersion, magnitude = _floor_catalogue()
    bright = magnitude < 19.0
    fallback = np.full(3, 1e-12)
    absolute = floored_dispersion(
        values, dispersion, percentile=50.0,
        fractional=np.array([False, False, False]), fallback=fallback,
    )
    # An absolute floor at the median does not move a bright atom's flux
    # tolerance at all: its own scatter already exceeds the median.
    np.testing.assert_allclose(absolute[bright, 2], dispersion[bright, 2])
    fractional = floored_dispersion(
        values, dispersion, percentile=50.0,
        fractional=np.array([False, False, True]), fallback=fallback,
    )
    # The fractional floor does move it: on this catalogue every bright atom's
    # flux tolerance is lifted, here by a median factor of 4.59.
    assert (fractional[bright, 2] > dispersion[bright, 2]).all()
    assert np.median(fractional[bright, 2] / dispersion[bright, 2]) > 4.0
    # In fractional terms every atom now sits at or above the same tolerance,
    # and the bright ones sit exactly on it because their own scatter is below.
    median_ratio = np.median(dispersion[:, 2] / values[:, 2])
    ratio = fractional[:, 2] / values[:, 2]
    assert ratio.min() >= median_ratio - 1e-12
    np.testing.assert_allclose(ratio[bright], median_ratio, rtol=1e-9)


def test_additive_floor_equalizes_bright_and_faint_tolerances():
    """Shape and size need no fractional treatment, only a higher percentile."""
    from sbsi.catalogue_sampling import floored_dispersion

    values, dispersion, magnitude = _floor_catalogue()
    bright, faint = magnitude < 19.0, magnitude > 26.0
    before = np.median(dispersion[faint, 0]) / np.median(dispersion[bright, 0])
    floored = floored_dispersion(
        values, dispersion, percentile=50.0,
        fractional=np.array([False, False, False]),
        fallback=np.full(3, 1e-12),
    )
    after = np.median(floored[faint, 0]) / np.median(floored[bright, 0])
    # Measured on this catalogue: 6.38 before, 1.76 after.
    assert before > 5.0
    assert after < before / 3.0


def test_refloor_of_an_additive_coordinate_matches_a_fresh_build():
    """Re-flooring a cached table must not differ from building it that way.

    The shipped table was already floored at the 1st percentile.  Applying a
    higher percentile afterwards has to give exactly what a fresh build at that
    percentile would, or the cached 24m table cannot be reused.
    """
    from sbsi.catalogue_sampling import floored_dispersion

    values, dispersion, _ = _floor_catalogue()
    flat = np.array([False, False, False])
    fallback = np.full(3, 1e-12)
    once = floored_dispersion(values, dispersion, percentile=1.0,
                              fractional=flat, fallback=fallback)
    twice = floored_dispersion(values, once, percentile=50.0,
                               fractional=flat, fallback=fallback)
    direct = floored_dispersion(values, dispersion, percentile=50.0,
                                fractional=flat, fallback=fallback)
    np.testing.assert_allclose(twice, direct, rtol=0.0, atol=0.0)


def test_floor_never_lowers_a_dispersion():
    from sbsi.catalogue_sampling import floored_dispersion

    values, dispersion, _ = _floor_catalogue()
    for percentile in (0.0, 1.0, 50.0, 100.0):
        floored = floored_dispersion(
            values, dispersion, percentile=percentile,
            fractional=np.array([False, False, True]),
            fallback=np.full(3, 1e-12),
        )
        assert (floored >= dispersion - 1e-12).all()
        assert np.isfinite(floored).all() and (floored > 0).all()


def test_floor_repairs_a_non_positive_dispersion():
    from sbsi.catalogue_sampling import floored_dispersion

    values, dispersion, _ = _floor_catalogue(n=256)
    dispersion[3, 0] = 0.0
    dispersion[7, 1] = np.nan
    floored = floored_dispersion(
        values, dispersion, percentile=50.0,
        fractional=np.array([False, False, True]),
        fallback=np.full(3, 1e-12),
    )
    assert np.isfinite(floored).all() and (floored > 0).all()


def test_floor_rejects_a_bad_percentile_and_an_unknown_target():
    from sbsi.catalogue_sampling import fractional_mask, floored_dispersion

    values, dispersion, _ = _floor_catalogue(n=64)
    with pytest.raises(ValueError, match="percentile"):
        floored_dispersion(values, dispersion, percentile=101.0,
                           fractional=np.array([False, False, True]),
                           fallback=np.full(3, 1e-12))
    with pytest.raises(ValueError, match="absent from targets"):
        fractional_mask(("a", "b"), ("c",))
    assert fractional_mask(("a", "b"), ("b",)).tolist() == [False, True]


def test_with_dispersion_floor_preserves_values_and_records_the_refloor():
    values, dispersion, _ = _floor_catalogue(n=512)
    table = ProposalCoordinateTable(
        values=values,
        target_names=("g1", "radius", "flux"),
        center=np.median(values, axis=0),
        scale=np.std(values, axis=0),
        dispersion=dispersion,
        statistic="mean",
        dispersion_statistic="std",
        n_flow_samples=128,
    )
    floored = table.with_dispersion_floor(
        percentile=50.0, fractional_targets=("flux",)
    )
    np.testing.assert_array_equal(floored.values, table.values)
    assert floored.target_names == table.target_names
    assert (floored.dispersion >= table.dispersion - 1e-12).all()
    assert floored.metadata["dispersion_refloor"]["percentile"] == 50.0
    assert floored.metadata["dispersion_refloor"]["fractional_targets"] == ["flux"]
    # The original table is untouched.
    np.testing.assert_array_equal(table.dispersion, dispersion)


def test_with_dispersion_floor_requires_a_dispersion():
    table = ProposalCoordinateTable(
        values=np.array([[0.0], [1.0]]),
        target_names=("g1",),
        center=np.array([0.5]),
        scale=np.array([1.0]),
    )
    with pytest.raises(ValueError, match="no dispersion"):
        table.with_dispersion_floor(percentile=50.0)


def test_fractional_floor_survives_a_coordinate_that_passes_through_zero():
    """A shear component sits arbitrarily close to zero, where sigma/|x| is not
    a finite number.  Such an atom has no fractional scatter, so it keeps the
    absolute floor instead of poisoning the percentile with an infinity."""

    from sbsi.catalogue_sampling import floored_dispersion

    values = np.array([[0.0, 1.0],
                       [5.0e-320, 2.0],
                       [-0.5, 4.0],
                       [0.25, 8.0]])
    dispersion = np.array([[0.1, 0.5],
                           [0.2, 1.0],
                           [0.3, 2.0],
                           [0.4, 4.0]])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        floored = floored_dispersion(
            values, dispersion, percentile=50.0,
            fractional=np.array([True, True]),
            fallback=np.array([1.0e-6, 1.0e-6]),
        )
    assert np.isfinite(floored).all()
    assert (floored >= dispersion).all()
    # The two atoms that cannot form the quotient fall back to the absolute
    # floor, the median of the column's own scatter.
    np.testing.assert_allclose(floored[:2, 0], np.median(dispersion[:, 0]))
    # The two that can are lifted to the median ratio times their own |x|.
    ratio = np.median(dispersion[2:, 0] / np.abs(values[2:, 0]))
    np.testing.assert_allclose(
        floored[2:, 0],
        np.maximum(dispersion[2:, 0], np.abs(values[2:, 0]) * ratio),
    )

