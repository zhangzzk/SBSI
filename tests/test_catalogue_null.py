import numpy as np
from dataclasses import replace
import pytest
import torch
from scipy.optimize import minimize
from scipy.stats import norm

from sbsi.catalogue_closure import generate_mock_catalogue, generate_ring_mock_catalogue
from sbsi.catalogue_blend import CatalogueBlendResponse
from sbsi.catalogue_likelihood import (
    CatalogueLikelihood,
    CatalogueModelCache,
    CatalogueSelection,
)
from sbsi.catalogue_null import (
    Section5SeedResult,
    _draw_initial_center_posterior_adapted,
    _numerical_stencil,
    _safeguarded_numerical_recenter,
    autograd_exact_section5,
    autograd_importance_section5,
    assess_section5_null,
    estimate_one_step_adaptive_section5,
    evaluate_fixed_draw_log_likelihood,
    optimize_shear_numerical,
    run_adaptive_section5,
    run_stratified_section5,
    run_exact_section5,
    run_streamed_section5,
    summarize_paired_section5,
    summarize_section5,
)
from sbsi.catalogue_sampling import (
    DefensiveLocalProposal,
    ProposalDraw,
    ProposalCoordinateTable,
    run_importance_profiles,
)
from sbsi.measurement_model import (
    ConditionalMeanFlow,
    MeasurementModelBundle,
    TargetStandardizer,
)
from sbsi.scene_prior import ScenePrior
from sbsi.selection_model import TabularPreprocessor
from sbsi.score_inference import OutputCut
from test_catalogue_likelihood import CONDITIONS, SpinZeroDetector, _catalogue, _two_shape_likelihood


def _proposal(likelihood):
    values = likelihood.cache.get(0.0, 0.0).flow[["e1_input_p", "e2_input_p"]].to_numpy(float)
    coordinates = ProposalCoordinateTable(
        values=values,
        target_names=("measured_ngmix_g1", "measured_ngmix_g2"),
        center=np.zeros(2),
        scale=np.full(2, 0.2),
        dispersion=np.full_like(values, 0.12),
        statistic="mean",
    )
    return DefensiveLocalProposal(
        coordinates,
        likelihood.cache.prior.weights,
        local_base_weights=likelihood.cache.get(0.0, 0.0).detection_probability,
    )


def _torch_two_shape_likelihood(sigma=0.12, *, blend_values=None, selection=None):
    features = ["e1_input_p", "e2_input_p"]
    preprocessor = TabularPreprocessor(
        feature_names=features,
        fill_values=np.zeros(2, dtype=np.float32),
        means=np.zeros(2, dtype=np.float32),
        scales=np.ones(2, dtype=np.float32),
        add_missing_indicators=True,
    )
    targets = ["measured_ngmix_g1", "measured_ngmix_g2"]
    target_transform = TargetStandardizer(
        targets,
        np.zeros(2, dtype=np.float32),
        np.full(2, sigma, dtype=np.float32),
    )
    model = ConditionalMeanFlow(
        target_dim=2,
        context_dim=4,
        hidden_dim=8,
        n_layers=1,
        n_flows=1,
        mean_hidden=0,
        flow_drop_indices=(0, 1, 2, 3),
    )
    with torch.no_grad():
        model.mean_net.weight.zero_()
        model.mean_net.bias.zero_()
        model.mean_net.weight[0, 0] = 1.0 / sigma
        model.mean_net.weight[1, 1] = 1.0 / sigma
    bundle = MeasurementModelBundle(
        model,
        preprocessor,
        target_transform,
        device="cpu",
    )
    prior = ScenePrior.from_catalogue(_catalogue(), guard_radius_arcsec=8.0, weight_column="prior_weight")
    detector = SpinZeroDetector()
    detector.predict_proba = lambda frame: np.ones(len(frame), dtype=float)
    blend_response = None
    if blend_values is not None:
        blend_response = CatalogueBlendResponse(
            np.asarray(blend_values, dtype=float), metadata={"test": True}, report={}
        )
    cache = CatalogueModelCache(
        prior,
        detector=detector,
        conditions=CONDITIONS,
        detection_radius_arcsec=3.0,
        flow_neighbour_radius_arcsec=7.0,
        crowding_radii_arcsec=(3.0, 7.0),
        blend_response=blend_response,
        flow_features=features,
    )
    return CatalogueLikelihood(bundle, cache, selection=selection)


def _enumerated_draw(likelihood, n_objects):
    """Represent one exact finite-catalogue sum as an importance draw."""

    rows = np.flatnonzero(likelihood.cache.prior.weights > 0).astype(np.int64)
    indices = np.broadcast_to(rows, (n_objects, len(rows))).copy()
    probability = np.full(indices.shape, 1.0 / len(rows), dtype=np.float64)
    flags = np.zeros(indices.shape, dtype=bool)
    return ProposalDraw(
        indices=indices,
        probability=probability,
        local_member=flags,
        global_component=flags,
        candidate_radius=np.zeros(n_objects, dtype=np.float64),
        seed=0,
    )


class _AnalyticTwoShapeSelection:
    """Exact Gaussian pass mass for a cut on the first measured shape."""

    def __init__(self, upper=0.2):
        self.upper = float(upper)
        self.output_cut = OutputCut(
            ["measured_ngmix_g1", "measured_ngmix_g2"],
            bounds=[("measured_ngmix_g1", None, self.upper)],
        )

    def probability(self, flow_model, view, *, active_indices=None):
        mean = view.flow["e1_input_p"].to_numpy(float) + view.blend_shift[:, 0]
        return norm.cdf((self.upper - mean) / flow_model.sigma)


class _EnumeratingProposal:
    """Public optimizer proposal whose draw is the exact atom enumeration."""

    def __init__(self, likelihood):
        self.likelihood = likelihood

    def draw(self, observed, **kwargs):
        return _enumerated_draw(self.likelihood, len(observed))


def test_autograd_exact_section5_matches_direct_finite_difference():
    likelihood = _torch_two_shape_likelihood()
    mock = generate_mock_catalogue(
        likelihood,
        n_detected=12,
        g1=0.0,
        g2=0.0,
        scene_seed=791,
        detection_seed=792,
        flow_seed=793,
    )
    for center in ((0.0, 0.0), (0.02, 0.0)):
        autograd = autograd_exact_section5(likelihood, mock, center=center)
        assert autograd.score.shape == (12, 2)
        assert autograd.information.shape == (12, 2, 2)
        for component, direction in enumerate(((1.0, 0.0), (0.0, 1.0))):
            finite = likelihood.score_and_information(
                mock.measurements,
                center=center,
                direction=direction,
                delta=0.002,
                richardson=False,
                object_chunk=12,
                atom_chunk=4,
            )
            np.testing.assert_allclose(autograd.score[:, component], finite.score, rtol=2e-3, atol=2e-3)
            np.testing.assert_allclose(
                autograd.information[:, component, component],
                finite.information,
                rtol=2e-2,
                atol=5e-2,
            )


def test_fixed_draw_two_component_evaluator_matches_directional_profile():
    likelihood = _torch_two_shape_likelihood()
    mock = generate_mock_catalogue(
        likelihood,
        n_detected=12,
        g1=0.01,
        g2=0.0,
        scene_seed=781,
        detection_seed=782,
        flow_seed=783,
    )
    proposal = _proposal(likelihood)
    kwargs = dict(
        n_draws=64,
        n_candidates=4,
        epsilon=0.2,
        bandwidth=1.0,
        seed=784,
    )
    draw = proposal.draw(mock.measurements, **kwargs)
    surface = evaluate_fixed_draw_log_likelihood(
        likelihood,
        mock,
        draw,
        ((0.0, 0.0), (0.01, 0.0)),
        object_chunk=6,
        atom_chunk=16,
    )
    profile = run_importance_profiles(
        likelihood,
        mock,
        proposal,
        shears=(0.0, 0.01),
        ladder=(64,),
        n_candidates=4,
        epsilon=0.2,
        bandwidth=1.0,
        proposal_seeds=(784,),
        direction=(1.0, 0.0),
        object_chunk=6,
    )[0]
    expected = {point.shear: point.log_likelihood_sum for point in profile.rungs[0].points}
    np.testing.assert_allclose(surface[(0.0, 0.0)], expected[0.0], atol=2e-5)
    np.testing.assert_allclose(surface[(0.01, 0.0)], expected[0.01], atol=2e-5)


@pytest.mark.parametrize(
    ("use_blend", "use_selection"),
    ((False, False), (True, False), (False, True), (True, True)),
)
def test_fixed_draw_surface_matches_exact_likelihood_with_full_model(use_blend, use_selection):
    selection = None
    if use_selection:
        cut = OutputCut(
            ["measured_ngmix_g1", "measured_ngmix_g2"],
            bounds=[("measured_ngmix_g1", None, 0.2)],
        )
        selection = CatalogueSelection(cut, n_samples=32, seed=771, row_chunk=2)
    blend = [0.6, -0.1, 0.3, 0.8] if use_blend else None
    likelihood = _torch_two_shape_likelihood(
        blend_values=blend,
        selection=selection,
    )
    mock = generate_mock_catalogue(
        likelihood,
        n_detected=24,
        g1=0.015,
        g2=-0.01,
        scene_seed=772,
        detection_seed=773,
        flow_seed=774,
    )
    shears = ((0.0, 0.0), (0.015, -0.01), (0.025, 0.005))
    actual = evaluate_fixed_draw_log_likelihood(
        likelihood,
        mock,
        _enumerated_draw(likelihood, len(mock.measurements)),
        shears,
        object_chunk=8,
        atom_chunk=4,
    )
    expected = {
        point: float(
            likelihood.log_likelihood(
                mock.measurements,
                point[0],
                point[1],
                object_chunk=8,
                atom_chunk=4,
            ).sum()
        )
        for point in shears
    }
    for point in shears:
        np.testing.assert_allclose(actual[point], expected[point], rtol=0, atol=5e-5)


@pytest.mark.parametrize(
    ("use_blend", "use_selection"),
    ((False, False), (True, False), (False, True), (True, True)),
)
def test_fixed_draw_surface_matches_hand_catalogue_sum(use_blend, use_selection):
    response = np.array([0.6, -0.1, 0.3, 0.8]) if use_blend else None
    selection = _AnalyticTwoShapeSelection() if use_selection else None
    likelihood = _two_shape_likelihood(
        blend_values=response,
        selection=selection,
    )
    # The toy detector is a bare callable, so declare its spin-0 input here;
    # production checkpoints expose this metadata themselves.
    likelihood.cache.detection_features = ("r_input_p_scaled",)
    mock = generate_mock_catalogue(
        likelihood,
        n_detected=24,
        g1=0.015,
        g2=-0.01,
        scene_seed=775,
        detection_seed=776,
        flow_seed=777,
    )
    shears = ((0.0, 0.0), (0.015, -0.01), (0.025, 0.005))
    actual = evaluate_fixed_draw_log_likelihood(
        likelihood,
        mock,
        _enumerated_draw(likelihood, len(mock.measurements)),
        shears,
        object_chunk=8,
        atom_chunk=4,
    )
    observed = mock.measurements[["measured_ngmix_g1", "measured_ngmix_g2"]].to_numpy(float)
    for point in shears:
        view = likelihood.cache.get(*point)
        mean = view.flow[["e1_input_p", "e2_input_p"]].to_numpy(float)
        mean = mean + view.blend_shift
        residual = observed[:, None, :] - mean[None, :, :]
        sigma = likelihood.flow_model.sigma
        density = np.exp(-0.5 * np.square(residual / sigma).sum(axis=2)) / (2.0 * np.pi * sigma**2)
        detected_mass = likelihood.cache.prior.weights * view.detection_probability
        pass_probability = (
            np.ones(len(mean), dtype=float)
            if selection is None
            else norm.cdf((selection.upper - mean[:, 0]) / sigma)
        )
        expected = np.log(density @ detected_mass).sum() - len(observed) * np.log(
            np.sum(detected_mass * pass_probability)
        )
        np.testing.assert_allclose(actual[point], expected, rtol=0, atol=1e-10)


def test_posterior_adapted_reference_reuse_matches_fixed_draw_full_model():
    cut = OutputCut(
        ["measured_ngmix_g1", "measured_ngmix_g2"],
        bounds=[("measured_ngmix_g1", None, 0.2)],
    )
    likelihood = _torch_two_shape_likelihood(
        blend_values=[0.6, -0.1, 0.3, 0.8],
        selection=CatalogueSelection(cut, n_samples=32, seed=768, row_chunk=2),
    )
    mock = generate_mock_catalogue(
        likelihood,
        n_detected=12,
        g1=0.015,
        g2=-0.01,
        scene_seed=769,
        detection_seed=770,
        flow_seed=771,
    )
    reference = (0.012, -0.007)
    draw, reused, candidate_evaluations, reuse_evaluations = _draw_initial_center_posterior_adapted(
        likelihood,
        mock,
        _proposal(likelihood),
        reference=reference,
        n_draws=64,
        n_candidates=2,
        epsilon=0.5,
        proposal_seed=772,
        object_chunk=5,
        atom_chunk=4,
    )
    independent = evaluate_fixed_draw_log_likelihood(
        likelihood,
        mock,
        draw,
        (reference,),
        object_chunk=5,
        atom_chunk=4,
    )[reference]
    assert reused == pytest.approx(independent, abs=1e-11)
    assert candidate_evaluations == len(mock.measurements) * 2
    assert reuse_evaluations > 0


def test_selected_likelihood_differs_only_by_population_normalization():
    observed_likelihood = _two_shape_likelihood(
        blend_values=[0.6, -0.1, 0.3, 0.8],
        selection=_AnalyticTwoShapeSelection(),
    )
    mock = generate_mock_catalogue(
        observed_likelihood,
        n_detected=32,
        g1=0.02,
        g2=-0.01,
        scene_seed=778,
        detection_seed=779,
        flow_seed=780,
    )
    configurations = {
        "uncut": _two_shape_likelihood(),
        "blend": _two_shape_likelihood(blend_values=[0.6, -0.1, 0.3, 0.8]),
        "selected": _two_shape_likelihood(selection=_AnalyticTwoShapeSelection()),
        "combined": observed_likelihood,
    }
    at_zero = {
        name: value.log_likelihood(mock.measurements, 0.0, 0.0).sum()
        for name, value in configurations.items()
    }
    np.testing.assert_allclose(at_zero["blend"], at_zero["uncut"], atol=1e-12)
    np.testing.assert_allclose(at_zero["combined"], at_zero["selected"], atol=1e-12)

    point = (0.02, -0.01)
    for selected_name, uncut_name in (
        ("selected", "uncut"),
        ("combined", "blend"),
    ):
        selected_likelihood = configurations[selected_name]
        uncut_likelihood = configurations[uncut_name]
        actual = (
            selected_likelihood.log_likelihood(mock.measurements, *point).sum()
            - uncut_likelihood.log_likelihood(mock.measurements, *point).sum()
        )
        expected = len(mock.measurements) * (
            uncut_likelihood.log_population_normalization(*point)
            - selected_likelihood.log_population_normalization(*point)
        )
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-11)


def test_full_numerical_stencil_recovers_mixed_information():
    information = np.array([[3.0, 0.7], [0.7, 2.0]])
    linear = np.array([0.4, -0.25])

    def evaluate(points):
        return {
            tuple(point): float(
                1.3 + linear @ np.asarray(point) - 0.5 * np.asarray(point) @ information @ np.asarray(point)
            )
            for point in points
        }

    center = np.array([0.03, -0.02])
    value, score, measured_information = _numerical_stencil(evaluate, center, 0.005)
    expected_value = 1.3 + linear @ center - 0.5 * center @ information @ center
    np.testing.assert_allclose(value, expected_value, rtol=0, atol=1e-12)
    np.testing.assert_allclose(score, linear - information @ center, rtol=0, atol=1e-12)
    np.testing.assert_allclose(measured_information, information, rtol=0, atol=1e-10)


def test_combined_likelihood_recenter_matches_independent_exact_mle():
    likelihood = _two_shape_likelihood(
        blend_values=[0.6, -0.1, 0.3, 0.8],
        selection=_AnalyticTwoShapeSelection(),
    )
    likelihood.cache.detection_features = ("r_input_p_scaled",)
    mock = generate_mock_catalogue(
        likelihood,
        n_detected=600,
        g1=0.02,
        g2=-0.01,
        scene_seed=785,
        detection_seed=786,
        flow_seed=787,
    )

    reference = minimize(
        lambda shear: (
            -float(
                likelihood.log_likelihood(
                    mock.measurements,
                    float(shear[0]),
                    float(shear[1]),
                    object_chunk=200,
                    atom_chunk=4,
                ).sum()
            )
        ),
        x0=np.array([0.02, -0.01]),
        method="L-BFGS-B",
        bounds=((-0.1, 0.1), (-0.1, 0.1)),
        options={"gtol": 1e-8, "ftol": 1e-12},
    )
    assert reference.success or np.linalg.norm(reference.jac) < 1e-4
    result = optimize_shear_numerical(
        likelihood,
        mock,
        _EnumeratingProposal(likelihood),
        initial=(0.0, 0.0),
        h=0.001,
        n_draws=4,
        n_candidates=4,
        epsilon=0.2,
        bandwidth=1.0,
        proposal_seed=788,
        proposal_method="distance_kernel",
        max_iterations=10,
        tolerance=1e-5,
        max_step=0.02,
        shear_bound=0.1,
        max_backtracks=8,
        object_chunk=200,
        atom_chunk=4,
    )
    assert result.converged, result.reason
    np.testing.assert_allclose(result.estimate, reference.x, rtol=0, atol=3e-4)
    assert all(
        iteration.next_log_likelihood_sum >= iteration.log_likelihood_sum for iteration in result.iterations
    )
    assert min(result.iterations[-1].information_eigenvalues) > 0
    assert result.diagnostic_flow_evaluations == len(mock.measurements) * 4
    assert result.flow_evaluations == (
        result.numerator_flow_evaluations
        + result.selection_flow_evaluations
        + result.diagnostic_flow_evaluations
    )
    assert 0 < result.importance.mean_ess <= 4
    assert 0 < result.importance.mean_ess_fraction <= 1
    assert 0 < result.importance.p90_max_weight_fraction <= 1


def test_posterior_adapted_optimizer_reuses_initial_and_counts_flow_calls():
    likelihood = _torch_two_shape_likelihood(blend_values=[0.6, -0.1, 0.3, 0.8])
    mock = generate_mock_catalogue(
        likelihood,
        n_detected=12,
        g1=0.015,
        g2=-0.01,
        scene_seed=789,
        detection_seed=790,
        flow_seed=791,
    )
    initial = (0.01, -0.005)
    result = optimize_shear_numerical(
        likelihood,
        mock,
        _proposal(likelihood),
        initial=initial,
        h=0.001,
        n_draws=64,
        n_candidates=2,
        epsilon=0.5,
        proposal_seed=792,
        proposal_method="initial_center_posterior_adapted",
        max_iterations=5,
        tolerance=1e-4,
        max_step=0.02,
        shear_bound=0.1,
        max_backtracks=8,
        object_chunk=5,
        atom_chunk=4,
    )
    assert result.proposal_method == "initial_center_posterior_adapted"
    assert result.proposal_reference_shear == initial
    assert result.bandwidth is None
    assert result.initial_likelihood_reused
    assert result.evaluations[0].shear == initial
    assert result.proposal_candidate_flow_evaluations == len(mock.measurements) * 2
    assert result.proposal_reuse_flow_evaluations > 0
    assert result.proposal_flow_evaluations == (
        result.proposal_candidate_flow_evaluations + result.proposal_reuse_flow_evaluations
    )
    assert result.numerator_flow_evaluations == (len(mock.measurements) * 64 * (len(result.evaluations) - 1))
    assert result.flow_evaluations == (
        result.proposal_flow_evaluations
        + result.numerator_flow_evaluations
        + result.selection_flow_evaluations
        + result.diagnostic_flow_evaluations
    )


def test_safeguarded_recenter_crosses_negative_curvature_from_zero():
    truth = 0.05

    def evaluate(points):
        result = {}
        for point in points:
            x, y = point
            displacement = x - truth
            result[point] = (-(displacement**2) - 10.0 * displacement**3 - y**2) * 1.0e6
        return result

    estimate, converged, reason, history = _safeguarded_numerical_recenter(
        evaluate,
        initial=(0.0, 0.0),
        h=0.005,
        max_iterations=12,
        tolerance=2e-4,
        max_step=0.01,
        shear_bound=0.1,
        max_backtracks=8,
    )
    assert history[0].information_eigenvalues[0] < 0
    assert history[0].step_method == "gradient"
    assert converged, reason
    assert estimate == pytest.approx((truth, 0.0), abs=3e-4)
    assert all(iteration.next_log_likelihood_sum >= iteration.log_likelihood_sum for iteration in history)


def test_tensor_native_importance_weights_and_stream_match_dataframe_path():
    likelihood = _torch_two_shape_likelihood()
    mock = generate_mock_catalogue(
        likelihood,
        n_detected=24,
        g1=0.0,
        g2=0.0,
        scene_seed=794,
        detection_seed=795,
        flow_seed=796,
    )
    proposal = _proposal(likelihood)
    candidates = proposal.candidates(mock.measurements, n_candidates=4)
    candidate_target = likelihood.log_importance_weights(
        mock.measurements,
        0.0,
        0.0,
        atom_indices=candidates.indices,
        proposal_probability=np.ones_like(candidates.indices, dtype=float),
    )
    draw = proposal.draw_adapted(
        candidates,
        candidate_target,
        n_draws=64,
        epsilon=0.2,
        seed=797,
    )
    dataframe = likelihood.log_importance_weights(
        mock.measurements,
        0.005,
        0.0,
        atom_indices=draw.indices,
        proposal_probability=draw.probability,
        object_chunk=12,
        atom_chunk=16,
    )
    tensor = likelihood.log_importance_weights_tensor(
        mock.measurements,
        0.005,
        0.0,
        atom_indices=draw.indices,
        proposal_probability=draw.probability,
        object_chunk=12,
        atom_chunk=16,
    )
    np.testing.assert_allclose(tensor.detach().cpu().numpy(), dataframe, rtol=2e-5, atol=2e-5)

    kwargs = dict(
        steps=(0.005,),
        ladder=(32, 64),
        n_candidates=4,
        epsilon=0.2,
        bandwidth=0.8,
        proposal_seeds=(798,),
        object_chunk=12,
        atom_chunk=16,
        posterior_adapt_proposal=True,
        retain_object_moments=True,
    )
    old = run_streamed_section5(likelihood, mock, _proposal(likelihood), use_tensor_native=False, **kwargs)
    new = run_streamed_section5(likelihood, mock, _proposal(likelihood), use_tensor_native=True, **kwargs)
    for key in old.object_moments[798].score:
        np.testing.assert_allclose(
            new.object_moments[798].score[key],
            old.object_moments[798].score[key],
            rtol=5e-3,
            atol=5e-3,
        )
        np.testing.assert_allclose(
            new.object_moments[798].information[key],
            old.object_moments[798].information[key],
            rtol=2e-2,
            atol=2e-2,
        )


def test_adaptive_tensor_path_matches_fixed_maximum_when_all_rows_continue():
    likelihood = _torch_two_shape_likelihood()
    mock = generate_mock_catalogue(
        likelihood,
        n_detected=20,
        g1=0.0,
        g2=0.0,
        scene_seed=799,
        detection_seed=800,
        flow_seed=801,
    )
    fixed = run_streamed_section5(
        likelihood,
        mock,
        _proposal(likelihood),
        steps=(0.005,),
        ladder=(64,),
        n_candidates=4,
        epsilon=0.2,
        bandwidth=1.0,
        proposal_seeds=(802,),
        object_chunk=10,
        atom_chunk=16,
        posterior_adapt_proposal=True,
        retain_object_moments=True,
    )
    adaptive = run_adaptive_section5(
        likelihood,
        mock,
        _proposal(likelihood),
        h=0.005,
        draw_ladder=(32, 64),
        n_candidates=4,
        epsilon=0.2,
        proposal_seed=802,
        min_ess=1e9,
        max_weight_fraction=1.0,
        object_chunk=10,
        atom_chunk=16,
    )
    np.testing.assert_array_equal(adaptive.draw_counts, 64)
    moments = fixed.object_moments[802]
    for component, index in (("g1", 0), ("g2", 1)):
        key = (component, 0.005, 64)
        np.testing.assert_allclose(adaptive.score[:, index], moments.score[key], rtol=5e-3, atol=5e-3)
        np.testing.assert_allclose(
            adaptive.information[:, index],
            moments.information[key],
            rtol=2e-2,
            atol=2e-2,
        )


def test_adaptive_tensor_path_accepts_uncertainty_reranked_prefilter():
    likelihood = _torch_two_shape_likelihood()
    mock = generate_mock_catalogue(
        likelihood,
        n_detected=12,
        g1=0.01,
        g2=0.0,
        scene_seed=903,
        detection_seed=904,
        flow_seed=905,
    )
    result = run_adaptive_section5(
        likelihood,
        mock,
        _proposal(likelihood),
        center=(0.01, 0.0),
        h=0.005,
        draw_ladder=(32,),
        n_candidates=2,
        proposal_prefilter_candidates=4,
        epsilon=0.2,
        proposal_seed=906,
        min_ess=1e9,
        max_weight_fraction=1.0,
        object_chunk=6,
        atom_chunk=16,
    )

    assert result.n_candidates == 2
    assert result.proposal_prefilter_candidates == 4
    assert np.isfinite(result.score).all()
    assert np.isfinite(result.information).all()


def test_independent_pilot_allocation_is_fixed_before_production_draw():
    likelihood = _torch_two_shape_likelihood()
    mock = generate_mock_catalogue(
        likelihood,
        n_detected=18,
        g1=0.01,
        g2=-0.005,
        scene_seed=916,
        detection_seed=917,
        flow_seed=918,
    )
    common = dict(
        likelihood=likelihood,
        mock=mock,
        proposal=_proposal(likelihood),
        center=(0.01, -0.005),
        h=0.005,
        draw_ladder=(16, 32, 64),
        n_candidates=4,
        epsilon=0.2,
        min_ess=4.0,
        max_weight_fraction=0.5,
        allocation_method="independent_pilot",
        pilot_draws=16,
        pilot_seed=919,
        pilot_safety_factor=1.0,
        full_information=True,
        object_chunk=6,
        atom_chunk=16,
    )
    first = run_adaptive_section5(proposal_seed=920, **common)
    second = run_adaptive_section5(proposal_seed=921, **common)

    np.testing.assert_array_equal(first.draw_counts, second.draw_counts)
    np.testing.assert_allclose(first.pilot_ess_fraction, second.pilot_ess_fraction)
    np.testing.assert_allclose(first.pilot_max_weight_fraction, second.pilot_max_weight_fraction)
    assert first.allocation_method == "independent_pilot"
    assert first.pilot_draws == 16
    assert first.pilot_seed == 919
    assert first.information.shape == (18, 2, 2)
    assert np.isfinite(first.score).all()
    assert np.isfinite(first.information).all()


def test_independent_pilot_rejects_reusing_production_seed():
    likelihood = _torch_two_shape_likelihood()
    mock = generate_mock_catalogue(
        likelihood,
        n_detected=4,
        g1=0.0,
        g2=0.0,
        scene_seed=922,
        detection_seed=923,
        flow_seed=924,
    )
    with pytest.raises(ValueError, match="must differ"):
        run_adaptive_section5(
            likelihood,
            mock,
            _proposal(likelihood),
            draw_ladder=(16,),
            n_candidates=4,
            proposal_seed=925,
            allocation_method="independent_pilot",
            pilot_draws=8,
            pilot_seed=925,
        )


def test_stratified_path_is_exact_when_candidates_cover_prior_support():
    likelihood = _torch_two_shape_likelihood()
    mock = generate_mock_catalogue(
        likelihood,
        n_detected=10,
        g1=0.01,
        g2=0.0,
        scene_seed=907,
        detection_seed=908,
        flow_seed=909,
    )
    kwargs = dict(
        center=(0.01, 0.0),
        h=0.005,
        complement_draw_ladder=(8, 16),
        n_candidates=4,
        proposal_prefilter_candidates=None,
        retain_full_ladder=True,
        object_chunk=5,
        atom_chunk=16,
    )
    first = run_stratified_section5(likelihood, mock, _proposal(likelihood), proposal_seed=910, **kwargs)
    second = run_stratified_section5(likelihood, mock, _proposal(likelihood), proposal_seed=911, **kwargs)

    np.testing.assert_allclose(first.score, second.score, rtol=0, atol=0)
    np.testing.assert_allclose(first.information, second.information, rtol=0, atol=0)
    np.testing.assert_allclose(first.ladder_score[0], first.ladder_score[1])
    np.testing.assert_allclose(first.ladder_information[0], first.ladder_information[1])


def test_retained_full_ladder_matches_separate_nested_prefix_runs():
    likelihood = _torch_two_shape_likelihood()
    mock = generate_mock_catalogue(
        likelihood,
        n_detected=12,
        g1=0.02,
        g2=0.0,
        scene_seed=912,
        detection_seed=913,
        flow_seed=914,
    )
    common = dict(
        likelihood=likelihood,
        mock=mock,
        proposal=_proposal(likelihood),
        center=(0.02, 0.0),
        h=0.005,
        n_candidates=4,
        epsilon=0.2,
        proposal_seed=915,
        min_ess=1e9,
        max_weight_fraction=1.0,
        full_information=True,
        object_chunk=6,
        atom_chunk=16,
    )
    retained = run_adaptive_section5(draw_ladder=(32, 64), retain_full_ladder=True, **common)
    shallow = run_adaptive_section5(draw_ladder=(32,), **common)
    deep = run_adaptive_section5(draw_ladder=(64,), **common)

    np.testing.assert_array_equal(retained.draw_counts, 64)
    np.testing.assert_allclose(retained.ladder_score[0], shallow.score)
    np.testing.assert_allclose(retained.ladder_information[0], shallow.information)
    np.testing.assert_allclose(retained.ladder_score[1], deep.score)
    np.testing.assert_allclose(retained.ladder_information[1], deep.information)
    np.testing.assert_allclose(retained.score, deep.score)
    np.testing.assert_allclose(retained.information, deep.information)


def test_richardson_correction_matches_nested_moment_combination():
    likelihood = _torch_two_shape_likelihood()
    mock = generate_mock_catalogue(
        likelihood,
        n_detected=12,
        g1=0.02,
        g2=-0.01,
        scene_seed=926,
        detection_seed=927,
        flow_seed=928,
    )
    common = dict(
        likelihood=likelihood,
        mock=mock,
        proposal=_proposal(likelihood),
        center=(0.02, -0.01),
        h=0.005,
        n_candidates=4,
        epsilon=0.2,
        proposal_seed=929,
        min_ess=1e9,
        max_weight_fraction=1.0,
        full_information=True,
        object_chunk=6,
        atom_chunk=16,
    )
    retained = run_adaptive_section5(draw_ladder=(32, 64), retain_full_ladder=True, **common)
    corrected = run_adaptive_section5(draw_ladder=(64,), bias_correction="richardson_1_over_m", **common)

    np.testing.assert_allclose(
        corrected.score,
        2.0 * retained.ladder_score[1] - retained.ladder_score[0],
    )
    np.testing.assert_allclose(
        corrected.information,
        2.0 * retained.ladder_information[1] - retained.ladder_information[0],
    )
    assert corrected.bias_correction == "richardson_1_over_m"


def test_richardson_correction_rejects_retained_ladder_mode():
    likelihood = _torch_two_shape_likelihood()
    mock = generate_mock_catalogue(
        likelihood,
        n_detected=4,
        g1=0.0,
        g2=0.0,
        scene_seed=930,
        detection_seed=931,
        flow_seed=932,
    )
    with pytest.raises(ValueError, match="separate modes"):
        run_adaptive_section5(
            likelihood,
            mock,
            _proposal(likelihood),
            draw_ladder=(16, 32),
            n_candidates=4,
            retain_full_ladder=True,
            bias_correction="richardson_1_over_m",
        )


def test_batched_importance_autograd_matches_local_finite_difference():
    likelihood = _torch_two_shape_likelihood()
    mock = generate_mock_catalogue(
        likelihood,
        n_detected=16,
        g1=0.0,
        g2=0.0,
        scene_seed=803,
        detection_seed=804,
        flow_seed=805,
    )
    kwargs = dict(
        draw_ladder=(32, 64),
        n_candidates=4,
        epsilon=0.2,
        proposal_seed=806,
        min_ess=1e9,
        max_weight_fraction=1.0,
        object_chunk=8,
        atom_chunk=16,
    )
    finite = run_adaptive_section5(
        likelihood,
        mock,
        _proposal(likelihood),
        h=0.005,
        full_information=True,
        **kwargs,
    )
    exact = autograd_importance_section5(
        likelihood,
        mock,
        _proposal(likelihood),
        **kwargs,
    )
    np.testing.assert_array_equal(exact.draw_counts, finite.draw_counts)
    np.testing.assert_allclose(exact.score, finite.score, rtol=2e-3, atol=2e-3)
    np.testing.assert_allclose(
        exact.information,
        finite.information,
        rtol=2e-2,
        atol=5e-2,
    )
    np.testing.assert_allclose(
        exact.information[:, 0, 1],
        exact.information[:, 1, 0],
        rtol=2e-4,
        atol=2e-4,
    )

    one_step = estimate_one_step_adaptive_section5(
        likelihood,
        mock,
        _proposal(likelihood),
        center=(0.0, 0.0),
        h=0.005,
        **kwargs,
    )
    score_sum = one_step.moments.score.sum(axis=0)
    information_sum = one_step.moments.information.sum(axis=0)
    np.testing.assert_allclose(information_sum @ one_step.step, score_sum)
    np.testing.assert_allclose(one_step.estimate, one_step.step)
    assert np.min(one_step.information_eigenvalues) > 0
    assert np.isfinite(one_step.robust_standard_error).all()


def _estimate_index(result):
    return {(estimate.component, estimate.h, estimate.n_draws): estimate for estimate in result.estimates}


def test_streamed_section5_matches_small_exact_catalogue_without_atom_tensor():
    likelihood = _two_shape_likelihood()
    mock = generate_mock_catalogue(
        likelihood,
        n_detected=80,
        g1=0.0,
        g2=0.0,
        scene_seed=801,
        detection_seed=802,
        flow_seed=803,
    )
    exact = run_exact_section5(likelihood, mock, steps=(0.005,), object_chunk=40, atom_chunk=4)
    weights = likelihood.cache.prior.weights
    bank_a = weights * np.array([1, 1, 0, 0], dtype=float)
    bank_b = weights * np.array([0, 0, 1, 1], dtype=float)
    streamed = run_streamed_section5(
        likelihood,
        mock,
        _proposal(likelihood),
        steps=(0.005,),
        ladder=(128, 512),
        n_candidates=4,
        epsilon=0.2,
        bandwidth=0.8,
        proposal_seeds=(804,),
        object_chunk=13,
        atom_chunk=64,
        independent_bank_weights={"a": bank_a, "b": bank_b},
    )
    sampled = streamed.results[0]
    exact_by_component = {estimate.component: estimate for estimate in exact.estimates}
    sampled_index = _estimate_index(sampled)
    for component in ("g1", "g2"):
        actual = sampled_index[(component, 0.005, 512)]
        expected = exact_by_component[component]
        assert abs(actual.estimated_shear - expected.estimated_shear) < 0.03
        assert actual.importance.mean_ess > 1
        assert np.isfinite(actual.information_identity_ratio)
    assert sampled.n_views == 5
    assert sampled.flow_evaluations == 80 * 512 * 5
    assert set(streamed.independent_banks) == {"a", "b"}
    assert all(result.flow_evaluations == 0 for result in streamed.independent_banks.values())


def test_streamed_section5_supports_zero_shear_posterior_adaptation():
    likelihood = _two_shape_likelihood()
    mock = generate_mock_catalogue(
        likelihood,
        n_detected=40,
        g1=0.0,
        g2=0.0,
        scene_seed=805,
        detection_seed=806,
        flow_seed=807,
    )
    result = run_streamed_section5(
        likelihood,
        mock,
        _proposal(likelihood),
        steps=(0.005,),
        ladder=(64, 256),
        n_candidates=4,
        epsilon=0.1,
        bandwidth=0.8,
        proposal_seeds=(808,),
        object_chunk=11,
        atom_chunk=32,
        posterior_adapt_proposal=True,
    ).results[0]
    assert result.flow_evaluations == 40 * (5 * 256 + 4)
    assert all(estimate.importance.mean_ess_fraction > 0.5 for estimate in result.estimates)


def test_streamed_section5_retains_nonzero_object_moments_only_when_requested():
    likelihood = _two_shape_likelihood()
    mock = generate_mock_catalogue(
        likelihood,
        n_detected=12,
        g1=0.02,
        g2=0.0,
        scene_seed=809,
        detection_seed=810,
        flow_seed=811,
    )
    with np.testing.assert_raises_regex(ValueError, "requires g_true"):
        run_streamed_section5(
            likelihood,
            mock,
            _proposal(likelihood),
            steps=(0.005,),
            ladder=(32, 64),
            n_candidates=4,
            epsilon=0.2,
            bandwidth=0.8,
            proposal_seeds=(812,),
        )
    streamed = run_streamed_section5(
        likelihood,
        mock,
        _proposal(likelihood),
        steps=(0.005,),
        ladder=(32, 64),
        n_candidates=4,
        epsilon=0.2,
        bandwidth=0.8,
        proposal_seeds=(812,),
        require_null_truth=False,
        retain_object_moments=True,
    )
    moments = streamed.object_moments[812]
    assert moments.score[("g1", 0.005, 64)].shape == (12,)
    assert moments.information[("g2", 0.005, 32)].shape == (12,)


def test_paired_section5_summary_uses_aligned_influence_cancellation():
    rng = np.random.default_rng(813)
    n = 2000
    common = rng.normal(size=n)
    amplitude = 0.02
    positive_score = common + amplitude
    negative_score = common - amplitude
    information = np.ones(n)
    paired = summarize_paired_section5(
        positive_score,
        information,
        negative_score,
        information,
        injected_component="g1",
        estimated_component="g1",
        amplitude=amplitude,
        h=0.00125,
        n_draws=64,
    )
    assert paired.response == pytest.approx(1.0)
    assert paired.response_bias == pytest.approx(0.0)
    assert paired.robust_standard_error < 1.0e-12
    assert paired.symmetric_offset == pytest.approx(common.mean())


def test_paired_section5_response_is_joint_population_derivative_not_arm_difference():
    n = 100
    amplitude = 0.02
    positive_information = np.full(n, 2.0)
    negative_information = np.full(n, 4.0)
    symmetric_information = (positive_information + negative_information) / 2.0
    paired = summarize_paired_section5(
        amplitude * symmetric_information,
        positive_information,
        -amplitude * symmetric_information,
        negative_information,
        injected_component="g1",
        estimated_component="g1",
        amplitude=amplitude,
        h=0.00125,
        n_draws=64,
    )
    assert paired.response == pytest.approx(1.0)
    arm_difference = (paired.positive_estimated_shear - paired.negative_estimated_shear) / (2.0 * amplitude)
    assert arm_difference == pytest.approx(1.125)


def test_ring_mock_has_rotated_partners_and_shared_shear_streams():
    likelihood = _torch_two_shape_likelihood()
    base = generate_mock_catalogue(
        likelihood,
        n_detected=10,
        g1=0.0,
        g2=0.0,
        scene_seed=815,
        detection_seed=816,
        flow_seed=817,
    )
    positive = generate_ring_mock_catalogue(
        likelihood,
        base,
        g1=0.005,
        g2=0.0,
        orientation_seed=818,
        flow_seed=819,
    )
    negative = generate_ring_mock_catalogue(
        likelihood,
        base,
        g1=-0.005,
        g2=0.0,
        orientation_seed=818,
        flow_seed=819,
    )
    assert len(positive.measurements) == 20
    np.testing.assert_array_equal(positive.truth["ring_id"], negative.truth["ring_id"])
    np.testing.assert_array_equal(positive.truth["scene_row"], negative.truth["scene_row"])
    np.testing.assert_allclose(
        positive.truth.loc[:9, "intrinsic_e1"].to_numpy(),
        -positive.truth.loc[10:, "intrinsic_e1"].to_numpy(),
    )
    np.testing.assert_allclose(
        positive.truth.loc[:9, "intrinsic_e2"].to_numpy(),
        -positive.truth.loc[10:, "intrinsic_e2"].to_numpy(),
    )
    assert np.all(positive.truth["injected_g1"] == 0.005)
    assert np.all(negative.truth["injected_g1"] == -0.005)


def test_streamed_object_offsets_reproduce_one_full_proposal_draw():
    likelihood = _two_shape_likelihood()
    proposal = _proposal(likelihood)
    mock = generate_mock_catalogue(
        likelihood,
        n_detected=7,
        g1=0.0,
        g2=0.0,
        scene_seed=811,
        detection_seed=812,
        flow_seed=813,
    )
    kwargs = dict(
        n_draws=31,
        n_candidates=4,
        epsilon=0.2,
        bandwidth=0.8,
        seed=814,
    )
    full = proposal.draw(mock.measurements, **kwargs)
    first = proposal.draw(mock.measurements.iloc[:3], object_offset=0, **kwargs)
    second = proposal.draw(mock.measurements.iloc[3:], object_offset=3, **kwargs)
    np.testing.assert_array_equal(full.indices, np.concatenate([first.indices, second.indices]))
    np.testing.assert_array_equal(full.probability, np.concatenate([first.probability, second.probability]))


def _passing_result(seed, ladder=(8192, 32768, 65536), steps=(0.005, 0.01, 0.02)):
    rng = np.random.default_rng(821)
    score = rng.normal(size=10000)
    score -= score.mean()
    information = np.full(len(score), np.var(score, ddof=1))
    estimates = tuple(
        summarize_section5(
            score,
            information,
            component=component,
            h=h,
            n_draws=n_draws,
        )
        for component in ("g1", "g2")
        for h in steps
        for n_draws in ladder
    )
    return Section5SeedResult(
        proposal_seed=seed,
        steps=steps,
        ladder=ladder,
        n_candidates=131072,
        n_objects=len(score),
        n_views=1 + 4 * len(steps),
        flow_evaluations=0,
        elapsed_seconds=1.0,
        estimates=estimates,
    )


def test_section5_assessment_requires_and_accepts_all_declared_gates():
    first = _passing_result(1)
    second = _passing_result(2)
    exact = _passing_result(None, ladder=(100,), steps=(0.005, 0.01, 0.02))
    oracle_sampled = _passing_result(1, ladder=(32768, 65536))
    banks = (
        _passing_result(1, ladder=(32768, 65536), steps=(0.005,)),
        _passing_result(1, ladder=(32768, 65536), steps=(0.005,)),
    )
    assessment = assess_section5_null(
        (first, second),
        exact_oracle=exact,
        oracle_importance=oracle_sampled,
        independent_banks=banks,
    )
    assert assessment.passed
    assert all(assessment.checks.values())

    bad_estimates = tuple(
        replace(
            estimate,
            tail=replace(estimate.tail, hill_tail_index=1.3),
        )
        if (estimate.component, estimate.h, estimate.n_draws) == ("g1", 0.005, 65536)
        else estimate
        for estimate in first.estimates
    )
    bad = replace(first, estimates=bad_estimates)
    rejected = assess_section5_null(
        (bad, second),
        exact_oracle=exact,
        oracle_importance=oracle_sampled,
        independent_banks=banks,
    )
    assert not rejected.checks["tail_stability"]
