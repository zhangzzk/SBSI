import numpy as np
from dataclasses import replace
import pytest
import torch
from scipy.stats import norm

from sbsi.catalogue_closure import generate_mock_catalogue, generate_ring_mock_catalogue
from sbsi.catalogue_blend import CatalogueBlendResponse
from sbsi.catalogue_likelihood import (
    CatalogueLikelihood,
    CatalogueModelCache,
)
from sbsi.catalogue_null import (
    Section5SeedResult,
    _coalesced_logsumexp,
    autograd_exact_section5,
    autograd_importance_section5,
    assess_section5_null,
    estimate_one_step_adaptive_section5,
    run_adaptive_section5,
    run_exact_section5,
    run_streamed_section5,
    summarize_paired_section5,
    summarize_section5,
)
from sbsi.catalogue_sampling import (
    DefensiveLocalProposal,
    ProposalCoordinateTable,
    pareto_tail_index,
)
from sbsi.measurement_model import (
    ConditionalMeanFlow,
    MeasurementModelBundle,
    TargetStandardizer,
)
from sbsi.scene_prior import ScenePrior
from sbsi.selection_model import TabularPreprocessor
from sbsi.catalogue_likelihood import OutputCut
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


def _torch_two_shape_likelihood(
    sigma=0.12,
    *,
    blend_values=None,
    selection=None,
    shape_sensitive_detection=False,
):
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
    if shape_sensitive_detection:
        detector.preprocessor.feature_names = ["e1_input_p", "e2_input_p"]
        detector.predict_proba = lambda frame: 1.0 / (
            1.0 + np.exp(-frame["e1_input_p"].to_numpy(float))
        )
    else:
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


class _TorchAnalyticTwoShapeSelection(_AnalyticTwoShapeSelection):
    def __init__(self, sigma, upper=0.2):
        super().__init__(upper=upper)
        self.sigma = float(sigma)

    def probability(self, flow_model, view, *, active_indices=None):
        mean = view.flow["e1_input_p"].to_numpy(float) + view.blend_shift[:, 0]
        return norm.cdf((self.upper - mean) / self.sigma)


def _adaptive_versus_exact_complete_likelihood(**fixture):
    likelihood = _torch_two_shape_likelihood(0.12, **fixture)
    mock = generate_mock_catalogue(
        likelihood,
        n_detected=24,
        g1=0.0,
        g2=0.0,
        scene_seed=901,
        detection_seed=902,
        flow_seed=903,
    )
    h = 0.005
    adaptive = run_adaptive_section5(
        likelihood,
        mock,
        _proposal(likelihood),
        h=h,
        draw_ladder=(32,),
        n_candidates=4,
        epsilon=0.2,
        estimator_mode="stratified",
        retain_full_ladder=True,
        proposal_seed=904,
        min_ess=1e9,
        max_weight_fraction=1.0,
        object_chunk=8,
        atom_chunk=16,
    )
    report = []
    for component, direction in ((0, (1.0, 0.0)), (1, (0.0, 1.0))):
        exact = likelihood.score_and_information(
            mock.measurements,
            center=(0.0, 0.0),
            direction=direction,
            delta=h,
            richardson=False,
            object_chunk=8,
            atom_chunk=16,
        )
        difference = adaptive.score[:, component] - exact.score
        scale = np.abs(exact.score).max()
        report.append(
            {
                "max_relative": float(
                    (np.abs(difference) / np.maximum(np.abs(exact.score), 1e-12)).max()
                ),
                "common_offset_fraction": float(np.abs(difference.mean()) / scale),
            }
        )
    return report


def test_adaptive_matches_exact_with_a_measured_cut_and_external_r_blend():
    control = _adaptive_versus_exact_complete_likelihood()
    combined = _adaptive_versus_exact_complete_likelihood(
        blend_values=[0.6, -0.1, 0.3, 0.8],
        selection=_TorchAnalyticTwoShapeSelection(0.12),
    )
    for component, (base, test) in enumerate(zip(control, combined)):
        assert test["common_offset_fraction"] < 1e-5, (component, test)
        assert test["max_relative"] < max(10.0 * base["max_relative"], 1e-3), (
            component,
            base,
            test,
        )


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


def test_skipping_padded_atom_slots_leaves_the_scored_slots_unchanged():
    """The ragged path must be a cost change only.

    `coalesce` pads every row of an object chunk out to the widest row's unique
    atom count, and every consumer masks those slots off again by their zero
    draw count.  Scoring them is therefore pure waste, but only if dropping
    them leaves the slots that survive the mask bit-comparable.
    """

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
    coalesced = proposal.draw_adapted(
        candidates,
        candidate_target,
        n_draws=64,
        epsilon=0.2,
        seed=797,
    ).coalesce()
    # The fixture has to actually exercise padding, or the test proves nothing.
    assert not coalesced.valid.all()

    kwargs = dict(
        atom_indices=coalesced.indices,
        proposal_probability=coalesced.probability,
        object_chunk=12,
        atom_chunk=16,
    )
    dense = likelihood.log_importance_weights_tensor(
        mock.measurements, 0.005, 0.0, **kwargs
    )
    ragged = likelihood.log_importance_weights_tensor(
        mock.measurements, 0.005, 0.0, atom_valid=coalesced.valid, **kwargs
    )
    np.testing.assert_allclose(
        ragged.detach().cpu().numpy()[coalesced.valid],
        dense.detach().cpu().numpy()[coalesced.valid],
        rtol=1e-6,
        atol=1e-6,
    )
    # Padded slots are never read downstream, so they only have to be finite.
    assert np.all(ragged.detach().cpu().numpy()[~coalesced.valid] == 0.0)
    # What the reduction actually consumes must agree at every nested rung.
    for rung in (16, 32, 64):
        np.testing.assert_allclose(
            _coalesced_logsumexp(ragged, coalesced, rung).detach().cpu().numpy(),
            _coalesced_logsumexp(dense, coalesced, rung).detach().cpu().numpy(),
            rtol=1e-6,
            atol=1e-6,
        )


def test_physical_flow_dense_and_ragged_weights_preserve_float64():
    from sbsi.flow_physical_disk import ConditionalPhysicalDiskFlow

    base = _torch_two_shape_likelihood(blend_values=[0.1, 0.2, 0.3, 0.4])
    model = ConditionalPhysicalDiskFlow(
        target_dim=4, context_dim=4, hidden_dim=8, n_layers=1, n_flows=1,
        disk_map="radial_tanh", disk_shape_means=[0, 0],
        disk_shape_scales=[0.3, 0.3], disk_coordinate_means=[0, 0],
        disk_coordinate_scales=[1, 1], physical_map="softplus",
        physical_means=[2, 100], physical_scales=[1, 100],
        physical_units=[1, 100], physical_coordinate_means=[0, 0],
        physical_coordinate_scales=[1, 1],
    )
    targets = TargetStandardizer(
        ["measured_ngmix_g1", "measured_ngmix_g2", "measured_flux_radius",
         "measured_flux_from_mag_auto"],
        np.array([0, 0, 2, 100]), np.array([0.3, 0.3, 1, 100]),
        dtype="float64",
    )
    bundle = MeasurementModelBundle(
        model, base.flow_model.condition_preprocessor, targets, device="cpu"
    )
    likelihood = CatalogueLikelihood(bundle, base.cache)
    observed = np.array([[0.1, -0.2, 1.3, 80], [-0.3, 0.2, 2.4, 120],
                         [0.01, 0.02, 0.8, 60]])
    atoms = np.array([[0, 1, 2], [2, 3, 2], [0, 0, 0]])
    valid = np.array([[True, True, True], [True, True, False], [True, False, False]])
    probability = np.array([[0.123456789, 0.2, 0.3], [0.3, 0.4, 0.3],
                            [0.123456789, 0.123456789, 0.123456789]])
    kwargs = dict(atom_indices=atoms, proposal_probability=probability,
                  object_chunk=2, atom_chunk=2)
    dense = likelihood.log_importance_weights_tensor(observed, 0.005, 0, **kwargs)
    ragged = likelihood.log_importance_weights_tensor(
        observed, 0.005, 0, atom_valid=valid, **kwargs
    )
    view = likelihood._tensor_view(0.005, 0)
    flat = torch.as_tensor(atoms.reshape(-1))
    target = bundle.target_tensor(likelihood._observed_frame(observed))
    target = target.repeat_interleave(3, dim=0) - view.blend_shift_standardized[flat]
    with torch.no_grad():
        expected = (
            bundle.log_prob_tensor(target, view.context[flat])
            + view.log_detected_mass[flat]
            - torch.log(torch.as_tensor(probability.reshape(-1)))
        ).reshape(atoms.shape)
    assert view.context.dtype == torch.float32
    assert dense.dtype == ragged.dtype == expected.dtype == torch.float64
    torch.testing.assert_close(dense, expected, rtol=1e-7, atol=1e-7)
    torch.testing.assert_close(ragged[valid], expected[valid], rtol=1e-7, atol=1e-7)
    assert torch.all(ragged[~valid] == 0)
    empty = likelihood.log_importance_weights_tensor(
        observed, 0.005, 0, atom_valid=np.zeros_like(valid), **kwargs
    )
    assert empty.dtype == torch.float64
    assert torch.all(empty == 0)


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


def test_adaptive_tensor_path_accepts_whole_catalogue_proxy_shortlist():
    likelihood = _torch_two_shape_likelihood()
    mock = generate_mock_catalogue(
        likelihood,
        n_detected=12,
        g1=0.01,
        g2=0.0,
        scene_seed=907,
        detection_seed=908,
        flow_seed=909,
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
        proposal_seed=910,
        min_ess=1e9,
        max_weight_fraction=1.0,
        candidate_backend="torch",
        candidate_source="whole_catalogue_gaussian_proxy",
        estimator_mode="tilted_stratified",
        retain_full_ladder=True,
        object_chunk=6,
        atom_chunk=16,
    )

    assert result.n_candidates == 2
    assert result.proposal_prefilter_candidates is None
    assert result.candidate_source == "whole_catalogue_gaussian_proxy"
    assert np.isfinite(result.score).all()
    assert np.isfinite(result.information).all()


def test_adaptive_tensor_path_accepts_shape_sensitive_detection_views():
    likelihood = _torch_two_shape_likelihood(shape_sensitive_detection=True)
    assert likelihood.tensor_shape_only_available is False
    mock = generate_mock_catalogue(
        likelihood,
        n_detected=12,
        g1=0.01,
        g2=0.0,
        scene_seed=911,
        detection_seed=912,
        flow_seed=913,
    )
    result = run_adaptive_section5(
        likelihood,
        mock,
        _proposal(likelihood),
        center=(0.01, 0.0),
        h=0.005,
        draw_ladder=(32,),
        n_candidates=2,
        epsilon=0.2,
        proposal_seed=914,
        min_ess=1e9,
        max_weight_fraction=1.0,
        candidate_backend="torch",
        candidate_source="whole_catalogue_gaussian_proxy",
        estimator_mode="tilted_stratified",
        retain_full_ladder=True,
        object_chunk=6,
        atom_chunk=16,
    )
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


def _stratified_common(likelihood, mock, n_candidates):
    return dict(
        likelihood=likelihood,
        mock=mock,
        proposal=_proposal(likelihood),
        center=(0.0, 0.0),
        h=0.005,
        draw_ladder=(16, 32),
        n_candidates=n_candidates,
        epsilon=0.2,
        min_ess=1e9,
        max_weight_fraction=1.0,
        full_information=True,
        retain_full_ladder=True,
        estimator_mode="stratified",
        object_chunk=4,
        atom_chunk=16,
    )


def test_stratified_estimator_is_exact_when_support_covers_the_catalogue():
    likelihood = _torch_two_shape_likelihood()
    mock = generate_mock_catalogue(
        likelihood,
        n_detected=8,
        g1=0.01,
        g2=0.0,
        scene_seed=1201,
        detection_seed=1202,
        flow_seed=1203,
    )
    # Four active atoms, so a four-atom candidate support leaves an empty
    # complement: the sampled stratum contributes nothing and the estimator
    # collapses onto the exact finite sum.
    common = _stratified_common(likelihood, mock, n_candidates=4)
    first = run_adaptive_section5(proposal_seed=1204, **common)
    second = run_adaptive_section5(proposal_seed=1205, **common)

    # No draw can move it: neither the seed nor the rung changes the answer.
    np.testing.assert_allclose(first.score, second.score, rtol=0, atol=1e-12)
    np.testing.assert_allclose(
        first.information, second.information, rtol=0, atol=1e-12
    )
    np.testing.assert_allclose(
        first.ladder_score[0], first.ladder_score[1], rtol=0, atol=1e-12
    )
    assert np.all(first.unique_counts == 0)

    h = 0.005

    def exact(g1, g2):
        return likelihood.log_likelihood(
            mock.measurements, g1, g2, object_chunk=4, atom_chunk=16
        )

    for component, plus, minus in (
        (0, (h, 0.0), (-h, 0.0)),
        (1, (0.0, h), (0.0, -h)),
    ):
        expected = (exact(*plus) - exact(*minus)) / (2.0 * h)
        np.testing.assert_allclose(
            first.score[:, component], expected, rtol=2e-3, atol=2e-3
        )


def test_stratified_mode_rejects_allocation_and_correction_combinations():
    likelihood = _torch_two_shape_likelihood()
    mock = generate_mock_catalogue(
        likelihood,
        n_detected=4,
        g1=0.0,
        g2=0.0,
        scene_seed=1301,
        detection_seed=1302,
        flow_seed=1303,
    )
    common = _stratified_common(likelihood, mock, n_candidates=2)
    with pytest.raises(ValueError, match="retained full ladder"):
        run_adaptive_section5(
            proposal_seed=1304, **{**common, "retain_full_ladder": False}
        )
    with pytest.raises(ValueError, match="retained full ladder"):
        run_adaptive_section5(
            proposal_seed=1304,
            **{**common, "allocation_method": "independent_pilot"},
        )
    with pytest.raises(ValueError, match="separate modes"):
        run_adaptive_section5(
            proposal_seed=1304,
            **{**common, "bias_correction": "richardson_1_over_m"},
        )
    with pytest.raises(ValueError, match="unknown estimator mode"):
        run_adaptive_section5(
            proposal_seed=1304, **{**common, "estimator_mode": "nonsense"}
        )


def test_stratified_complement_is_summed_once_and_normalized_by_the_draw_count():
    likelihood = _torch_two_shape_likelihood()
    mock = generate_mock_catalogue(
        likelihood,
        n_detected=8,
        g1=0.01,
        g2=0.0,
        scene_seed=1401,
        detection_seed=1402,
        flow_seed=1403,
    )
    # Half the catalogue is summed exactly and the rest is reached only by
    # prior draws.  Rebuilding the two-stratum sum by hand from the same draw
    # is the test that the complement is neither dropped nor counted twice and
    # that it carries the 1/M factor rather than a per-atom proposal weight.
    n_draws = 4096
    common = _stratified_common(likelihood, mock, n_candidates=2)
    common["draw_ladder"] = (1024, n_draws)
    result = run_adaptive_section5(proposal_seed=1404, **common)
    assert np.all(result.unique_counts > 0)

    proposal = _proposal(likelihood)
    candidates = proposal.candidates(mock.measurements, n_candidates=2)
    draw = proposal.draw_stratified(candidates, n_draws=n_draws, seed=1404)
    weights = likelihood.cache.prior.weights
    n_objects = len(mock.measurements)
    every_atom = np.tile(np.arange(len(weights)), (n_objects, 1))
    ones = np.ones(every_atom.shape, dtype=np.float64)

    def log_likelihood(g1, g2):
        log_term = likelihood.log_importance_weights_tensor(
            mock.measurements,
            g1,
            g2,
            atom_indices=every_atom,
            proposal_probability=ones,
            object_chunk=4,
            atom_chunk=16,
        )
        term = np.exp(log_term.detach().cpu().numpy().astype(np.float64))
        rows = np.arange(n_objects)[:, None]
        exact = term[rows, candidates.indices].sum(axis=1)
        retained = ~draw.local_member
        tail = np.where(
            retained, term[rows, draw.indices] / weights[draw.indices], 0.0
        ).sum(axis=1) / n_draws
        return np.log(exact + tail) - likelihood.log_population_normalization(g1, g2)

    h = 0.005
    for component, plus, minus in (
        (0, (h, 0.0), (-h, 0.0)),
        (1, (0.0, h), (0.0, -h)),
    ):
        expected = (log_likelihood(*plus) - log_likelihood(*minus)) / (2.0 * h)
        np.testing.assert_allclose(
            result.score[:, component], expected, rtol=1e-3, atol=1e-3
        )


def test_weight_diagnostics_match_a_direct_reduction_of_the_same_draws():
    """The saved ESS, peak share, and k-hat must describe the actual draws.

    Rebuilding them from the proposal draw itself, without touching the
    reduction path, is the only check that they describe the weights the
    estimator reduces rather than some neighbouring quantity.
    """

    likelihood = _torch_two_shape_likelihood()
    mock = generate_mock_catalogue(
        likelihood,
        n_detected=6,
        g1=0.01,
        g2=0.0,
        scene_seed=1301,
        detection_seed=1302,
        flow_seed=1303,
    )
    proposal = _proposal(likelihood)
    ladder = (64, 256)
    result = run_adaptive_section5(
        likelihood=likelihood,
        mock=mock,
        proposal=proposal,
        center=(0.0, 0.0),
        h=0.005,
        draw_ladder=ladder,
        n_candidates=2,
        epsilon=0.2,
        min_ess=1e9,
        max_weight_fraction=1.0,
        full_information=False,
        retain_full_ladder=True,
        proposal_seed=1304,
        object_chunk=3,
        atom_chunk=8,
    )
    assert result.weight_diagnostic_draws == ladder
    assert result.weight_ess.shape == (len(ladder), len(mock.measurements))
    # Every rung's effective count must be inside its own budget.
    for rung_index, rung in enumerate(ladder):
        assert (result.weight_ess[rung_index] > 0).all()
        assert (result.weight_ess[rung_index] <= rung + 1e-9).all()
    assert ((result.weight_max_fraction > 0) & (result.weight_max_fraction <= 1)).all()
    # With no exact stratum the relative error is the plain sampling one.
    for rung_index, rung in enumerate(ladder):
        assert np.allclose(
            result.weight_relative_error[rung_index],
            np.sqrt(1.0 / result.weight_ess[rung_index] - 1.0 / rung),
            rtol=1e-6,
        )

    candidates = proposal.candidates(mock.measurements, n_candidates=2)
    candidate_target = likelihood.log_importance_weights(
        mock.measurements,
        0.0,
        0.0,
        atom_indices=candidates.indices,
        proposal_probability=np.ones_like(candidates.indices, dtype=np.float64),
    )
    draw = proposal.draw_adapted(
        candidates,
        candidate_target,
        n_draws=ladder[-1],
        epsilon=0.2,
        seed=1304,
    )
    log_weights = likelihood.log_importance_weights(
        mock.measurements,
        0.0,
        0.0,
        atom_indices=draw.indices,
        proposal_probability=draw.probability,
    )
    # Rescale each row before exponentiating, exactly as the reduction does;
    # the diagnostics are invariant to it and the raw weights can underflow.
    weights = np.exp(log_weights - log_weights.max(axis=1, keepdims=True))
    for rung_index, rung in enumerate(ladder):
        prefix = weights[:, :rung]
        total = prefix.sum(axis=1)
        expected_ess = total**2 / (prefix**2).sum(axis=1)
        assert np.allclose(result.weight_ess[rung_index], expected_ess, rtol=1e-6)
        assert np.allclose(
            result.weight_max_fraction[rung_index],
            prefix.max(axis=1) / total,
            rtol=1e-6,
        )
        expected_k = pareto_tail_index(prefix)
        recorded = result.weight_pareto_k[rung_index]
        finite = np.isfinite(expected_k)
        assert np.array_equal(finite, np.isfinite(recorded))
        assert np.allclose(recorded[finite], expected_k[finite], rtol=1e-6)


def test_stratified_weight_diagnostics_ignore_the_exactly_summed_stratum():
    """Only the sampled complement carries variance, so only it is diagnosed."""

    likelihood = _torch_two_shape_likelihood()
    mock = generate_mock_catalogue(
        likelihood,
        n_detected=6,
        g1=0.01,
        g2=0.0,
        scene_seed=1311,
        detection_seed=1312,
        flow_seed=1313,
    )
    common = _stratified_common(likelihood, mock, 2)
    common["draw_ladder"] = (64, 256)
    result = run_adaptive_section5(**common, proposal_seed=1314)
    covered = run_adaptive_section5(
        **{**common, "n_candidates": 4, "draw_ladder": (64, 256)}, proposal_seed=1314
    )

    assert result.weight_diagnostic_draws == (64, 256)
    # Draws landing inside the exact stratum contribute nothing, so the
    # effective count is strictly below the budget.
    assert (result.weight_ess[-1] < 256).all()
    assert (result.weight_ess[-1] > 0).all()
    # With the whole catalogue in the stratum there is no sampled term left at
    # all, and the diagnostics must say so rather than invent a value.
    assert (covered.weight_ess == 0).all()
    assert (covered.weight_max_fraction == 0).all()
    assert np.isnan(covered.weight_pareto_k).all()
    # An estimator with nothing left to sample has no sampling error.
    assert (covered.weight_relative_error == 0).all()
    # The sampled share discounts the relative error below the plain sampling
    # one, which is exactly why ESS alone cannot be compared across modes.
    plain = np.sqrt(1.0 / result.weight_ess[-1] - 1.0 / 256)
    assert (result.weight_relative_error[-1] < plain).all()


def test_weight_diagnostics_are_absent_without_the_retained_ladder():
    likelihood = _torch_two_shape_likelihood()
    mock = generate_mock_catalogue(
        likelihood,
        n_detected=6,
        g1=0.01,
        g2=0.0,
        scene_seed=1321,
        detection_seed=1322,
        flow_seed=1323,
    )
    result = run_adaptive_section5(
        likelihood=likelihood,
        mock=mock,
        proposal=_proposal(likelihood),
        center=(0.0, 0.0),
        h=0.005,
        draw_ladder=(64, 256),
        n_candidates=2,
        epsilon=0.2,
        min_ess=8.0,
        max_weight_fraction=0.5,
        full_information=False,
        retain_full_ladder=False,
        proposal_seed=1324,
        object_chunk=3,
        atom_chunk=8,
    )
    assert result.weight_diagnostic_draws is None
    assert result.weight_ess is None
    assert result.weight_diagnostic_summary() is None
