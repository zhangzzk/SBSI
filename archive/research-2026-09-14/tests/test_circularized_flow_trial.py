import json

import numpy as np
import pandas as pd
import pytest
import torch

from scripts.trial_circularized_flow import (
    audit_response, fit_nll_arm, frozen_scalar_frame_gamma, verify_reconstructed_raw_targets,
)
from sbsi.flow_training import _fold_catalogue_shear_into_shape
from sbsi.flow_size_condition import circularized_radius
from sbsi.measurement_model import TargetStandardizer, build_flow, load_measurement_model
from sbsi.selection_model import TabularPreprocessor
from test_flow_physical_disk import config


def test_raw_reconstruction_allows_only_log_roundoff_and_keeps_cache_unchanged():
    names = ["measured_ngmix_g1", "measured_ngmix_g2", "measured_mag_auto", "measured_log_flux_radius"]
    cached = np.array([[.3, -.2, 24., 1.5], [.2, .1, 23., -.5]])
    original = cached.copy()
    actual = cached.copy()
    for _ in range(2):
        actual[:, 3] = np.nextafter(actual[:, 3], np.inf)
    report = verify_reconstructed_raw_targets(actual, cached, names)
    assert report[names[3]]["nonidentical"] == 2
    assert report[names[3]]["max_ulp"] == 2
    np.testing.assert_array_equal(cached, original)
    actual[:, 3] = np.nextafter(actual[:, 3], np.inf)
    with pytest.raises(AssertionError):
        verify_reconstructed_raw_targets(actual, cached, names)
    for column in range(3):
        actual = cached.copy()
        actual[0, column] = np.nextafter(actual[0, column], np.inf)
        with pytest.raises(AssertionError):
            verify_reconstructed_raw_targets(actual, cached, names)
    actual = cached.copy()
    actual[0, 3] = np.nan
    with pytest.raises(ValueError, match="nonfinite"):
        verify_reconstructed_raw_targets(actual, cached, names)


def test_fold_preserves_axis_ratio_for_invariant_size_on_both_legs():
    q0 = .4
    e0 = (1-q0)/(1+q0)
    frame = pd.DataFrame({"e1_input_rot0_p": [e0, e0], "e2_input_rot0_p": [0., 0.],
                          "gamma1_input_p": [0., .05], "gamma2_input_p": [0., .02],
                          "Re_input_p": [.8, .8], "axis_ratio_input_p": [q0, q0]})
    original = circularized_radius(frame.Re_input_p, frame.axis_ratio_input_p)
    baseline = frame[["e1_input_rot0_p", "e2_input_rot0_p"]].to_numpy().copy()
    folded = _fold_catalogue_shear_into_shape(frame.copy())
    np.testing.assert_array_equal(circularized_radius(folded.Re_input_p, folded.axis_ratio_input_p), original)
    abs_baseline = np.linalg.norm(baseline, axis=1)
    np.testing.assert_allclose(circularized_radius(folded.Re_input_p, (1-abs_baseline)/(1+abs_baseline)), original)
    abs_folded = np.hypot(folded.e1_input_rot0_p, folded.e2_input_rot0_p)
    wrong = circularized_radius(folded.Re_input_p, (1-abs_folded)/(1+abs_folded))
    assert not np.isclose(wrong[1], original[1])
    assert folded.e1_input_rot0_p.iloc[1] != e0
    np.testing.assert_array_equal(folded.gamma1_input_p, [0., 0.])


def test_scalar_response_frame_reproduces_frozen_context_precision():
    from sbsi.flow_physical_response import baseline_frame_gamma
    # Unsheared catalogue truth and decoded float32 contexts are close but
    # distinct. The invariant size uses the former; the frozen ruler uses the latter.
    frame = pd.DataFrame({"e1_input_p": [.123456789123, -.234567891234, .15, -.18],
                          "e2_input_p": [-.345678912345, .456789123456, -.32, .47],
                          "Re_input_p": [.8, 1., .8, 1.]})
    pre = TabularPreprocessor.fit(frame, list(frame))
    context = pre.transform_frame(frame)
    pairs = np.array([[0, 2], [1, 3]])
    gamma = np.array([[.04, .03], [-.03, .04]])
    # Literal reconstruction from the original frozen target builder.
    decoded = context[pairs[:, 0], :len(pre.feature_names)].astype(float)*pre.scales + pre.means
    expected = baseline_frame_gamma(gamma, decoded[:, :2])
    np.testing.assert_array_equal(frozen_scalar_frame_gamma(context, pairs, pre, gamma), expected)
    raw_frame = baseline_frame_gamma(gamma, frame.iloc[pairs[:, 0]][["e1_input_p", "e2_input_p"]].to_numpy())
    assert np.max(np.abs(raw_frame-expected)) > 1e-11


def test_nll_trial_fresh_initialization_reproducibility_and_checkpoint_selection(tmp_path):
    cfg = config()
    torch.manual_seed(87)
    context = torch.randn(24, 2)
    with torch.no_grad():
        values = build_flow(cfg)(torch.randn(24, 4), context)[0]
    frame = pd.DataFrame(context.numpy(), columns=["e1_input_p", "Re_input_p"])
    pre = TabularPreprocessor.fit(frame, list(frame), add_missing_indicators=False)
    target = TargetStandardizer(
        ["measured_ngmix_g1", "measured_ngmix_g2", "measured_flux_radius", "measured_flux_from_mag_auto"],
        np.array([0., 0., 3., 100.]), np.array([.3, .3, 2., 80.]), dtype="float64")
    recipe = dict(seed=501, learning_rate=1e-4, epochs=2, rows_per_epoch=13,
                  batch_size=6, early_stopping_patience=20)
    train, validation = (values[:18], context[:18]), (values[18:], context[18:])
    outputs = []
    for arm in ("a", "b"):
        _, result = fit_nll_arm(tmp_path / arm, arm, cfg, pre, target, train, validation, recipe,
                                dict(grid_response_weight=0., pretrained_weights_used=False))
        outputs.append(result)
        history = json.loads((tmp_path / arm / "history.json").read_text())["history"]
        assert result["best_epoch"] == min(range(3), key=lambda i: history[i]["validation"]["nll"])
        assert history[1]["train"]["rows"] == 13
        assert history[1]["train"]["optimizer_steps"] == 3
        loaded = load_measurement_model(result["best_checkpoint"])
        assert loaded.metadata["pretrained_weights_used"] is False
        assert loaded.metadata["grid_response_weight"] == 0.
    assert outputs[0]["best_validation_nll"] == outputs[1]["best_validation_nll"]
    for epoch in range(3):
        a = torch.load(tmp_path / "a" / ("epoch%03d.pt" % epoch), weights_only=False)
        b = torch.load(tmp_path / "b" / ("epoch%03d.pt" % epoch), weights_only=False)
        for name in a["state_dict"]:
            torch.testing.assert_close(a["state_dict"][name], b["state_dict"][name], rtol=0, atol=0)


def test_trial_case_audit_matches_existing_full_physical_response_estimator():
    from sbsi.flow_physical_case_validation import physical_case_sufficient_statistics
    from sbsi.flow_physical_population import PhysicalResponsePopulation, evaluate_physical_groups
    from sbsi.flow_physical_response import fit_physical_response
    rng = np.random.default_rng(51)
    n = 48
    cases = np.repeat(np.arange(12), 4)
    bins = np.tile([0, 0, 1, 1], 12)
    order = np.argsort(bins, kind="stable")
    cases, bins = cases[order], bins[order]
    gamma = rng.normal(size=(n, 2))*.05
    scalar_gamma = rng.normal(size=(n, 2))*.05
    delta = rng.normal(size=(n, 4))
    stats = fit_physical_response(delta, gamma, bins, cases, 2, scalar_gamma=scalar_gamma)
    case = physical_case_sufficient_statistics(gamma, scalar_gamma, bins, cases, delta,
                                              n_cells=2, case_labels=np.arange(12))
    case_design = dict(normal=case["normal"], measured_cross=case["cross"],
                       scalar_gamma=scalar_gamma, case_labels=case["case_labels"])
    pairs = np.arange(2*n).reshape(n, 2)
    payload = dict(pairs=pairs, gamma=gamma, bin_ids=bins, counts=stats["counts"],
                    pair_keys=np.stack([cases, np.arange(n)], axis=1))
    torch.manual_seed(17)
    model = build_flow(config())
    with torch.no_grad():
        for layer in model.base.layers:
            layer.net[-1].weight.normal_(std=.01)
    context = torch.randn(2*n, 2)
    recipe = dict(validation_groups=4, validation_draws=4, response_batch_size=7, response_seed=901)
    summary, actual, _ = audit_response(model, context, payload, case_design, stats, recipe)
    expected = evaluate_physical_groups(model, PhysicalResponsePopulation(context, pairs, bins, stats),
                                        draws=4, groups=4, batch_size=7, seed=901)
    np.testing.assert_allclose(actual["group_model_responses"], expected["group_means"], rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(summary["equal_cell"]["model_response"],
                               expected["group_means"].mean(axis=(0, 1)), rtol=1e-10, atol=1e-10)
    assert summary["draw_count"] == expected["draw_count"]
