#!/usr/bin/env python
"""Audit shape, magnitude, and size derivative losses on Flow E's held-out cases."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from sbsi.flow_coupling_target import _load_leg, coupling_bin_ids
from sbsi.flow_training import _fold_catalogue_shear_into_shape
from sbsi.measurement_model import add_measurement_target_features, load_measurement_model
from sbsi.preprocessing import rescale
from sbsi.training import build_shifted_context


KEYS = ["case", "input_index"]
FLOW_COLUMNS = [
    *KEYS,
    "detected",
    "Re_input_p",
    "Re_input_s",
    "r_input_p",
    "r_input_s",
    "sersic_n_input_p",
    "sersic_n_input_s",
    "e1_input_rot0_p",
    "e2_input_rot0_p",
    "e1_input_rot0_s",
    "e2_input_rot0_s",
    "gamma1_input_p",
    "gamma2_input_p",
    "gamma1_input_s",
    "gamma2_input_s",
    "neighbored",
    "distance",
    "polarization_angle",
    "nbr_flux_near",
    "nbr_flux_far",
    "nbr_flux_max",
    "r_blend",
    "measured_ngmix_g1",
    "measured_ngmix_g2",
    "measured_mag_auto",
    "measured_flux_radius",
]
SHEARED_COLUMNS = [
    *KEYS,
    "detected",
    "r_input_p",
    "Re_input_p",
    "r_blend",
    "neighbored",
    "distance",
    "gamma1_input_p",
    "gamma2_input_p",
    "gamma1_input_s",
    "gamma2_input_s",
    "measured_ngmix_g1",
    "measured_ngmix_g2",
    "measured_mag_auto",
    "measured_flux_radius",
]
RESCALE_KWARGS = dict(
    pixel_rms=0.312,
    pixel_size=0.2,
    zero_mag=30.0,
    psf_fwhm=0.73,
    moffat_beta=2.224,
)


def _response_matrix(delta, applied):
    normal = applied.T @ applied
    if np.linalg.cond(normal) > 1.0e8:
        raise ValueError("applied shear does not span a stable response matrix")
    return delta.T @ applied @ np.linalg.inv(normal)


def _response_matrix_outputs(delta, applied):
    """Least-squares response of every output column to the two shear columns."""

    delta = np.asarray(delta, dtype=np.float64)
    applied = np.asarray(applied, dtype=np.float64)
    if delta.ndim != 2 or applied.ndim != 2 or applied.shape[1] != 2:
        raise ValueError("delta and applied must be aligned matrices with two shear columns")
    if len(delta) != len(applied):
        raise ValueError("delta and applied must contain the same rows")
    normal = applied.T @ applied
    if np.linalg.cond(normal) > 1.0e8:
        raise ValueError("applied shear does not span a stable response matrix")
    return delta.T @ applied @ np.linalg.inv(normal)


def _bin_ids(frame, target):
    shape = tuple(int(value) for value in target["counts"].shape)
    fi = np.clip(np.digitize(frame["r_input_p"], target["edges_flux"]) - 1, 0, shape[0] - 1)
    si = np.clip(np.digitize(frame["Re_input_p"], target["edges_size"]) - 1, 0, shape[1] - 1)
    ci = np.clip(np.digitize(frame["r_blend"], target["edges_crowd"]) - 1, 0, shape[2] - 1)
    return (fi * shape[1] + si) * shape[2] + ci


def _nll_usable_mask(frame, primary_mag_max, primary_re_min):
    """Exact pre-rescale row domain used by the four-output NLL loader."""

    true_domain = (
        (frame["r_input_p"] > 18.0)
        & (frame["r_input_p"] < primary_mag_max)
        & (frame["Re_input_p"] > primary_re_min)
        & (frame["Re_input_p"] < 1.5)
        & (
            ((frame["distance"] > 0.0) & (frame["distance"] < 5.0))
            | (~frame["neighbored"].astype(bool))
        )
    )
    targets = [
        "measured_ngmix_g1", "measured_ngmix_g2",
        "measured_mag_auto", "measured_flux_radius",
    ]
    usable = true_domain & frame["detected"].astype(bool)
    usable &= np.isfinite(frame[targets]).all(axis=1)
    usable &= frame["measured_flux_radius"] > 0.0
    return true_domain, usable


def _model_derivatives(bundle, frame, delta, batch_size, difference="central"):
    if difference not in {"forward", "central"}:
        raise ValueError("difference must be 'forward' or 'central'")
    contexts = [
        build_shifted_context(
            frame,
            direction,
            sign * delta,
            bundle.condition_preprocessor,
            bundle.metadata["condition_features"],
            RESCALE_KWARGS,
        )
        for direction, sign in (
            (((1.0, 0.0), 0.0), ((1.0, 0.0), 1.0), ((1.0, 0.0), -1.0),
             ((0.0, 1.0), 1.0), ((0.0, 1.0), -1.0))
            if difference == "central"
            else (((1.0, 0.0), 0.0), ((1.0, 0.0), 1.0),
                  ((0.0, 1.0), 1.0))
        )
    ]
    scales = np.asarray(bundle.target_transform.scales, dtype=np.float64)
    derivatives = np.empty((len(frame), 4, 2), dtype=np.float64)
    bundle.model.eval()
    with torch.no_grad():
        for start in range(0, len(frame), batch_size):
            stop = min(start + batch_size, len(frame))
            tensors = [
                torch.as_tensor(context[start:stop], dtype=torch.float32, device=bundle.device)
                for context in contexts
            ]
            shifts = [bundle.model._shift(value) for value in tensors]
            if difference == "central":
                _, p1, m1, p2, m2 = shifts
                denominator = 2.0 * delta
                pairs = ((m1, p1), (m2, p2))
            else:
                zero, p1, p2 = shifts
                denominator = delta
                pairs = ((zero, p1), (zero, p2))
            for component, (minus, plus) in enumerate(pairs):
                derivatives[start:stop, :, component] = (
                    (plus - minus).double().cpu().numpy() * scales / denominator
                )
    return derivatives


def _sample_pair_delta(bundle, first, second, n_samples, batch_size, seed):
    """Return E_q[y|second]-E_q[y|first] with paired flow base draws."""

    if len(first) != len(second):
        raise ValueError("paired contexts must have the same number of rows")
    scales = np.asarray(bundle.target_transform.scales, dtype=np.float64)
    result = np.empty((len(first), len(scales)), dtype=np.float64)
    bundle.model.eval()
    with torch.no_grad():
        for start in range(0, len(first), batch_size):
            stop = min(start + batch_size, len(first))
            a = torch.as_tensor(first[start:stop], dtype=torch.float32, device=bundle.device)
            b = torch.as_tensor(second[start:stop], dtype=torch.float32, device=bundle.device)
            # A distinct deterministic stream per batch, reset identically for the two
            # conditions. This is the CRN contract; without it the O(1) draw noise is
            # divided by the small shear step.
            pair_seed = int(seed + start)
            torch.manual_seed(pair_seed)
            draw_a = bundle.model.sample(a, n_samples=n_samples, qmc=True).mean(dim=1)
            torch.manual_seed(pair_seed)
            draw_b = bundle.model.sample(b, n_samples=n_samples, qmc=True).mean(dim=1)
            result[start:stop] = (
                (draw_b - draw_a).double().cpu().numpy() * scales
            )
    return result


def _full_model_derivatives(
    bundle, frame, delta, batch_size, n_samples, seed, difference="central"
):
    """Finite-difference derivatives of the complete sampled conditional flow."""

    if difference not in {"forward", "central"}:
        raise ValueError("difference must be 'forward' or 'central'")
    directions = ((1.0, 0.0), (0.0, 1.0))
    derivatives = np.empty(
        (len(frame), bundle.target_transform.dim, len(directions)), dtype=np.float64
    )
    for component, direction in enumerate(directions):
        first_sign = -1.0 if difference == "central" else 0.0
        first = build_shifted_context(
            frame,
            direction,
            first_sign * delta,
            bundle.condition_preprocessor,
            bundle.metadata["condition_features"],
            RESCALE_KWARGS,
        )
        second = build_shifted_context(
            frame,
            direction,
            delta,
            bundle.condition_preprocessor,
            bundle.metadata["condition_features"],
            RESCALE_KWARGS,
        )
        denominator = (1.0 - first_sign) * delta
        derivatives[:, :, component] = _sample_pair_delta(
            bundle,
            first,
            second,
            n_samples,
            batch_size,
            seed + component * 1_000_000,
        ) / denominator
    return derivatives


def _prepared_flow_frame(frame):
    prepared = rescale(frame.copy(), **RESCALE_KWARGS)
    return add_measurement_target_features(prepared)


def _mean_head_pair_delta(bundle, first, second, batch_size):
    scales = np.asarray(bundle.target_transform.scales, dtype=np.float64)
    result = np.empty((len(first), len(scales)), dtype=np.float64)
    bundle.model.eval()
    with torch.no_grad():
        for start in range(0, len(first), batch_size):
            stop = min(start + batch_size, len(first))
            a = torch.as_tensor(first[start:stop], dtype=torch.float32, device=bundle.device)
            b = torch.as_tensor(second[start:stop], dtype=torch.float32, device=bundle.device)
            result[start:stop] = (
                (bundle.model._shift(b) - bundle.model._shift(a)).double().cpu().numpy()
                * scales
            )
    return result


def _population_derivative_summary(derivatives, frame, response_grid,
                                   coupling_mag, coupling_size, min_cell_count=100,
                                   coupling_edges_e_abs=None):
    """Summarize one derivative estimator against the held-out simulation rulers."""

    shape_model = derivatives[:, :2, :]
    mag_model = derivatives[:, 2, :]
    size_model = derivatives[:, 3, :]
    bin_id = _bin_ids(frame, response_grid)
    n_cells = int(np.prod(response_grid["counts"].shape))
    applied = frame[["gamma1_input_p_sheared", "gamma2_input_p_sheared"]].to_numpy(float)
    shape_delta = frame[
        ["measured_ngmix_g1_sheared", "measured_ngmix_g2_sheared"]
    ].to_numpy(float)
    shape_delta -= frame[["measured_ngmix_g1", "measured_ngmix_g2"]].to_numpy(float)
    squared_sum = 0.0
    count_sum = 0
    used_cells = 0
    for index in range(n_cells):
        selected = bin_id == index
        count = int(selected.sum())
        if count < min_cell_count:
            continue
        simulation = _response_matrix(shape_delta[selected], applied[selected])
        model = shape_model[selected].mean(axis=0)
        squared_sum += count * float(np.square(model - simulation).sum())
        count_sum += count
        used_cells += 1
    coupling_ids = coupling_bin_ids(frame, response_grid["edges_flux"],
                                   response_grid["edges_size"], response_grid["edges_crowd"],
                                   coupling_edges_e_abs)
    b_mag = coupling_mag[coupling_ids]
    b_size = coupling_size[coupling_ids]
    e1 = frame["e1_input_rot0_p"].to_numpy(float)
    e2 = frame["e2_input_rot0_p"].to_numpy(float)
    size_loss = float((
        np.square(size_model[:, 0] - b_size * e1)
        + np.square(size_model[:, 1] - b_size * e2)
    ).mean())
    magnitude_loss = float((
        np.square(mag_model[:, 0] - b_mag * e1)
        + np.square(mag_model[:, 1] - b_mag * e2)
    ).mean())
    return {
        "mean_response_all_outputs_by_shear": derivatives.mean(axis=0).tolist(),
        "shape_response_loss": float(squared_sum / (2.0 * count_sum)),
        "magnitude_derivative_loss": magnitude_loss,
        "log_size_derivative_loss": size_loss,
        "photometric_derivative_loss": magnitude_loss + size_loss,
        "shape_cells_used": used_cells,
        "rows_used": int(len(frame)),
    }


def _log_prob(bundle, frame, batch_size):
    prepared = rescale(frame.copy(), **RESCALE_KWARGS)
    prepared = add_measurement_target_features(prepared)
    return bundle.log_prob(prepared, batch_size=batch_size).astype(np.float64, copy=False)


def audit(model_path, g0_catalogue, sheared_catalogue, response_target, coupling_target,
          output, *, device="cuda", delta=0.02, batch_size=65536,
          reference_model_path=None, sample_draws=64, sample_rows=200000,
          sample_seed=7301):
    with np.load(response_target, allow_pickle=False) as rt:
        validation_cases = np.asarray(rt["validation_cases"], dtype=np.int64)
        training_cases = np.asarray(rt["train_cases"], dtype=np.int64)
        primary_mag_max = float(rt["primary_mag_max"])
        primary_re_min = float(rt["primary_re_min"])
        target_g0_catalogue = str(rt["g0_catalogue"].item())
        target_sheared_catalogue = str(rt["sheared_catalogue"].item())
        target_training_rows = int(np.asarray(rt["counts"]).sum())
        response_grid = {key: np.asarray(rt[key]) for key in (
            "edges_flux", "edges_size", "edges_crowd", "counts"
        )}
    with np.load(coupling_target, allow_pickle=False) as ct:
        if str(ct["case_role"].item()) != "validation":
            raise ValueError("coupling audit target must use case_role='validation'")
        for key in ("edges_flux", "edges_size", "edges_crowd"):
            np.testing.assert_array_equal(ct[key], response_grid[key])
        coupling_mag = np.asarray(ct["coupling_mag"], dtype=np.float64).reshape(-1)
        coupling_size = np.asarray(ct["coupling_size"], dtype=np.float64).reshape(-1)
        coupling_edges_e_abs = np.asarray(ct["edges_e_abs"], float) if "edges_e_abs" in ct else None

    zero = _load_leg(g0_catalogue, FLOW_COLUMNS, validation_cases)
    sheared = _load_leg(sheared_catalogue, SHEARED_COLUMNS, validation_cases)
    before_zero, before_sheared = len(zero), len(sheared)
    zero_true, zero_usable_independent = _nll_usable_mask(
        zero, primary_mag_max, primary_re_min
    )
    sheared_true, sheared_usable_independent = _nll_usable_mask(
        sheared, primary_mag_max, primary_re_min
    )
    zero_keys = zero.loc[zero_usable_independent, KEYS]
    sheared_keys = sheared.loc[sheared_usable_independent, KEYS]
    usable_union = zero_keys.merge(
        sheared_keys, on=KEYS, how="outer", indicator=True, validate="one_to_one"
    )
    independent_domain_counts = {
        "g0_identities_before_true_cuts": int(len(zero)),
        "sheared_identities_before_true_cuts": int(len(sheared)),
        "g0_identities_after_true_cuts": int(zero_true.sum()),
        "sheared_identities_after_true_cuts": int(sheared_true.sum()),
        "g0_nll_usable_after_true_cuts": int(zero_usable_independent.sum()),
        "sheared_nll_usable_after_true_cuts": int(sheared_usable_independent.sum()),
        "both_legs_nll_usable_after_true_cuts": int(
            (usable_union["_merge"] == "both").sum()
        ),
        "g0_only_nll_usable_after_true_cuts": int(
            (usable_union["_merge"] == "left_only").sum()
        ),
        "sheared_only_nll_usable_after_true_cuts": int(
            (usable_union["_merge"] == "right_only").sum()
        ),
    }
    renamed = sheared.rename(columns={
        name: f"{name}_sheared" for name in SHEARED_COLUMNS if name not in KEYS
    })
    frame = zero.merge(renamed, on=KEYS, how="inner", validate="one_to_one")
    matched = len(frame)
    true_domain = (
        (frame["r_input_p"] > 18.0)
        & (frame["r_input_p"] < primary_mag_max)
        & (frame["Re_input_p"] > primary_re_min)
        & (frame["Re_input_p"] < 1.5)
        & (
            ((frame["distance"] > 0.0) & (frame["distance"] < 5.0))
            | (~frame["neighbored"].astype(bool))
        )
    )
    zero_target_columns = [
        "measured_ngmix_g1", "measured_ngmix_g2",
        "measured_mag_auto", "measured_flux_radius",
    ]
    sheared_target_columns = [f"{name}_sheared" for name in zero_target_columns]
    zero_usable = true_domain & frame["detected"].astype(bool)
    zero_usable &= np.isfinite(frame[zero_target_columns]).all(axis=1)
    zero_usable &= frame["measured_flux_radius"] > 0.0
    sheared_usable = true_domain & frame["detected_sheared"].astype(bool)
    sheared_usable &= np.isfinite(frame[sheared_target_columns]).all(axis=1)
    sheared_usable &= frame["measured_flux_radius_sheared"] > 0.0
    matched_domain_counts = {
        "matched_identities_before_true_cuts": int(len(frame)),
        "matched_identities_after_true_cuts": int(true_domain.sum()),
        "g0_nll_usable_after_true_cuts": int(zero_usable.sum()),
        "sheared_nll_usable_after_true_cuts": int(sheared_usable.sum()),
        "both_legs_nll_usable_after_true_cuts": int((zero_usable & sheared_usable).sum()),
        "g0_only_nll_usable_after_true_cuts": int((zero_usable & ~sheared_usable).sum()),
        "sheared_only_nll_usable_after_true_cuts": int((~zero_usable & sheared_usable).sum()),
        "neither_leg_nll_usable_after_true_cuts": int((~zero_usable & ~sheared_usable & true_domain).sum()),
    }
    keep = true_domain
    finite_columns = [
        "r_input_p", "Re_input_p", "r_blend", "e1_input_rot0_p", "e2_input_rot0_p",
        "gamma1_input_p_sheared", "gamma2_input_p_sheared",
        "measured_ngmix_g1", "measured_ngmix_g2",
        "measured_ngmix_g1_sheared", "measured_ngmix_g2_sheared",
        "measured_mag_auto", "measured_mag_auto_sheared",
        "measured_flux_radius", "measured_flux_radius_sheared",
    ]
    keep &= np.isfinite(frame[finite_columns]).all(axis=1)
    keep &= (frame["measured_flux_radius"] > 0.0) & (frame["measured_flux_radius_sheared"] > 0.0)
    frame = frame.loc[keep].reset_index(drop=True)

    bundle = load_measurement_model(str(model_path), device=device)
    has_mean_head = hasattr(bundle.model, "_shift")
    bin_id = _bin_ids(frame, response_grid)
    n_cells = int(np.prod(response_grid["counts"].shape))

    applied = frame[["gamma1_input_p_sheared", "gamma2_input_p_sheared"]].to_numpy(float)
    shape_delta = frame[["measured_ngmix_g1_sheared", "measured_ngmix_g2_sheared"]].to_numpy(float)
    shape_delta -= frame[["measured_ngmix_g1", "measured_ngmix_g2"]].to_numpy(float)
    # Preserve the historical exact mean-head audit when that architectural
    # component exists. Plain flows have no _shift: their response losses are
    # filled below from the complete-flow CRN finite difference instead.
    if has_mean_head:
        derivatives = _model_derivatives(
            bundle, frame, delta, batch_size, difference="central"
        )
        legacy_summary = _population_derivative_summary(
            derivatives, frame, response_grid, coupling_mag, coupling_size, 100,
            coupling_edges_e_abs=coupling_edges_e_abs
        )
        shape_loss = legacy_summary["shape_response_loss"]
        magnitude_loss = legacy_summary["magnitude_derivative_loss"]
        size_loss = legacy_summary["log_size_derivative_loss"]
        theta_loss = legacy_summary["photometric_derivative_loss"]
        used_cells = legacy_summary["shape_cells_used"]
        derivative_estimator = "exact_explicit_mean_head"
        derivative_rows = int(len(frame))
    else:
        shape_loss = magnitude_loss = size_loss = theta_loss = None
        used_cells = None
        derivative_estimator = "complete_flow_crn_central"
        derivative_rows = None

    zero_nll_frame = frame[FLOW_COLUMNS].copy()
    sheared_nll_frame = frame[FLOW_COLUMNS].copy()
    for name in SHEARED_COLUMNS:
        if name not in KEYS:
            sheared_nll_frame[name] = frame[f"{name}_sheared"].to_numpy()
    sheared_nll_frame = _fold_catalogue_shear_into_shape(sheared_nll_frame)
    logp_zero = _log_prob(bundle, zero_nll_frame, batch_size)
    logp_sheared = _log_prob(bundle, sheared_nll_frame, batch_size)
    nll_zero = float(-logp_zero.mean())
    nll_sheared = float(-logp_sheared.mean())

    # Complete-flow response audit. The analytic path changes exactly the
    # likelihood's registered shear coordinates. The empirical-leg path feeds
    # every condition from its own simulation leg; this is deliberately shown
    # separately because measured conditioning variables then move too.
    if sample_rows <= 0:
        raise ValueError("sample_rows must be positive")
    if sample_draws <= 0 or sample_draws & (sample_draws - 1):
        raise ValueError("sample_draws must be a positive power of two for Sobol QMC")
    if len(frame) > sample_rows:
        rng = np.random.default_rng(sample_seed)
        sample_index = np.sort(rng.choice(len(frame), size=sample_rows, replace=False))
    else:
        sample_index = np.arange(len(frame))
    sampled = frame.iloc[sample_index].reset_index(drop=True)
    zero_sample = zero_nll_frame.iloc[sample_index].reset_index(drop=True)
    sheared_sample = sheared_nll_frame.iloc[sample_index].reset_index(drop=True)

    full_forward = _full_model_derivatives(
        bundle, sampled, delta, batch_size, sample_draws, sample_seed,
        difference="forward",
    )
    full_central = _full_model_derivatives(
        bundle, sampled, delta, batch_size, sample_draws, sample_seed,
        difference="central",
    )
    derivative_summaries = {
        "complete_flow_forward": _population_derivative_summary(
            full_forward, sampled, response_grid, coupling_mag, coupling_size, 20,
            coupling_edges_e_abs=coupling_edges_e_abs
        ),
        "complete_flow_central": _population_derivative_summary(
            full_central, sampled, response_grid, coupling_mag, coupling_size, 20,
            coupling_edges_e_abs=coupling_edges_e_abs
        ),
    }
    if has_mean_head:
        mean_forward = _model_derivatives(
            bundle, sampled, delta, batch_size, difference="forward"
        )
        mean_central = _model_derivatives(
            bundle, sampled, delta, batch_size, difference="central"
        )
        derivative_summaries.update({
            "mean_head_forward": _population_derivative_summary(
                mean_forward, sampled, response_grid, coupling_mag, coupling_size, 20,
                coupling_edges_e_abs=coupling_edges_e_abs
            ),
            "mean_head_central": _population_derivative_summary(
                mean_central, sampled, response_grid, coupling_mag, coupling_size, 20,
                coupling_edges_e_abs=coupling_edges_e_abs
            ),
        })
    else:
        central_summary = derivative_summaries["complete_flow_central"]
        shape_loss = central_summary["shape_response_loss"]
        magnitude_loss = central_summary["magnitude_derivative_loss"]
        size_loss = central_summary["log_size_derivative_loss"]
        theta_loss = central_summary["photometric_derivative_loss"]
        used_cells = central_summary["shape_cells_used"]
        derivative_rows = int(len(sampled))

    zero_prepared = _prepared_flow_frame(zero_sample)
    sheared_prepared = _prepared_flow_frame(sheared_sample)
    zero_context = bundle.condition_preprocessor.transform_frame(zero_prepared)
    sheared_context = bundle.condition_preprocessor.transform_frame(sheared_prepared)
    empirical_full_delta = _sample_pair_delta(
        bundle, zero_context, sheared_context, sample_draws, batch_size, sample_seed
    )
    target_names = list(bundle.target_transform.target_names)
    observed_delta = (
        sheared_prepared[target_names].to_numpy(float)
        - zero_prepared[target_names].to_numpy(float)
    )
    sampled_applied = sampled[
        ["gamma1_input_p_sheared", "gamma2_input_p_sheared"]
    ].to_numpy(float)
    empirical_condition_response = {
        "measured_outputs": _response_matrix_outputs(
            observed_delta, sampled_applied
        ).tolist(),
        "complete_flow": _response_matrix_outputs(
            empirical_full_delta, sampled_applied
        ).tolist(),
    }
    if has_mean_head:
        empirical_mean_delta = _mean_head_pair_delta(
            bundle, zero_context, sheared_context, batch_size
        )
        empirical_condition_response.update({
            "mean_head": _response_matrix_outputs(
                empirical_mean_delta, sampled_applied
            ).tolist(),
            "residual_flow_contribution": _response_matrix_outputs(
                empirical_full_delta - empirical_mean_delta, sampled_applied
            ).tolist(),
        })
    condition_names = list(bundle.condition_preprocessor.feature_names)
    context_delta = sheared_context - zero_context
    condition_motion = {
        name: {
            "mean_abs_standardized_delta": float(np.mean(np.abs(context_delta[:, index]))),
            "fraction_changed_gt_1e-7": float(np.mean(np.abs(context_delta[:, index]) > 1.0e-7)),
        }
        for index, name in enumerate(condition_names)
    }
    keep_indices = set(
        int(index) for index in bundle.model.keep_indices.detach().cpu().numpy().tolist()
    ) if hasattr(bundle.model, "keep_indices") else set(range(len(condition_names)))
    residual_blind_features = [
        name for index, name in enumerate(condition_names) if index not in keep_indices
    ]

    result = {
        "model": str(model_path),
        "g0_catalogue": str(g0_catalogue),
        "sheared_catalogue": str(sheared_catalogue),
        "response_target": str(response_target),
        "coupling_target": str(coupling_target),
        "population": "Flow-E 40 grouped validation cases; matched selected finite rows",
        "data_domain_audit": {
            "nll_population": (
                "Each leg independently: true primary cuts, detected=True, finite four-output "
                "measurement; the training job then priority-samples 2,000,000 rows per leg."
            ),
            "response_target_population": (
                "Inner-matched (case,input_index) identities with finite outputs in both "
                "the g=0 and forward g=0.05 legs, after the same true primary cuts."
            ),
            "nll_and_target_populations_identical": False,
            "population_mismatch": (
                "The target conditions on D0=Dg=1, while NLL conditions on detection in "
                "the evaluated leg only; one-leg detections enter NLL but not the target."
            ),
            "response_target_stencil": "forward 0 -> +0.05 with random applied directions",
            "historical_training_stencil": "central -0.02 -> +0.02 analytic shape shear",
            "stencils_identical": False,
            "target_g0_catalogue_matches_argument": (
                Path(target_g0_catalogue).resolve() == Path(g0_catalogue).resolve()
            ),
            "target_sheared_catalogue_matches_argument": (
                Path(target_sheared_catalogue).resolve() == Path(sheared_catalogue).resolve()
            ),
            "training_cases": training_cases.tolist(),
            "validation_cases": validation_cases.tolist(),
            "target_training_rows_after_match_cuts_finite": target_training_rows,
            "validation_population_counts_independent_legs": independent_domain_counts,
            "validation_population_counts_after_identity_match": matched_domain_counts,
            "condition_features": condition_names,
            "residual_flow_blind_features": residual_blind_features,
            "registered_inference_shear_path": (
                "Apply S_g to e1_input_p/e2_input_p; hold every other condition fixed."
            ),
            "empirical_leg_condition_motion": condition_motion,
        },
        "validation_cases": validation_cases.tolist(),
        "rows_g0_before_match": before_zero,
        "rows_sheared_before_match": before_sheared,
        "matched_rows_before_cuts": matched,
        "unmatched_g0_rows": before_zero - matched,
        "unmatched_sheared_rows": before_sheared - matched,
        "n_rows": int(len(frame)),
        "has_explicit_mean_head": has_mean_head,
        "derivative_loss_estimator": derivative_estimator,
        "derivative_loss_rows": derivative_rows,
        "shape_cells_used": used_cells,
        "shape_response_loss": float(shape_loss),
        "magnitude_derivative_loss": magnitude_loss,
        "log_size_derivative_loss": size_loss,
        "photometric_derivative_loss": theta_loss,
        "nll_g0": nll_zero,
        "nll_sheared": nll_sheared,
        "nll_balanced": 0.5 * (nll_zero + nll_sheared),
        "weighted_objective_r450_t500": 0.5 * (nll_zero + nll_sheared)
        + 450.0 * shape_loss + 500.0 * theta_loss,
        "weighted_objective_r500_t500": 0.5 * (nll_zero + nll_sheared)
        + 500.0 * shape_loss + 500.0 * theta_loss,
        "complete_flow_response_audit": {
            "sample_rows": int(len(sampled)),
            "qmc_draws_per_row": int(sample_draws),
            "common_random_seed": int(sample_seed),
            "target_names": target_names,
            "analytic_registered_shear_path": derivative_summaries,
            "empirical_full_leg_condition_path": empirical_condition_response,
            "interpretation_warning": (
                "The empirical full-leg path moves measured conditioning variables as well "
                "as sheared intrinsic shape, so it diagnoses NLL leg consistency but is not "
                "the configured inference derivative, which holds non-shape conditions fixed."
            ),
        },
    }
    if reference_model_path is not None:
        reference_bundle = load_measurement_model(str(reference_model_path), device=device)
        if bundle.condition_preprocessor.to_state()["feature_names"] != (
            reference_bundle.condition_preprocessor.to_state()["feature_names"]
        ):
            raise ValueError("reference model condition features differ")
        if bundle.target_transform.target_names != reference_bundle.target_transform.target_names:
            raise ValueError("reference model targets differ")
        reference_zero = _log_prob(reference_bundle, zero_nll_frame, batch_size)
        reference_sheared = _log_prob(reference_bundle, sheared_nll_frame, batch_size)
        delta_logp = np.concatenate(
            [logp_zero - reference_zero, logp_sheared - reference_sheared]
        )
        result.update({
            "preservation_reference_model": str(reference_model_path),
            "preservation_mse_log_prob": float(np.mean(np.square(delta_logp))),
            "preservation_rms_log_prob": float(np.sqrt(np.mean(np.square(delta_logp)))),
            "preservation_mean_delta_log_prob": float(np.mean(delta_logp)),
            "preservation_q95_abs_delta_log_prob": float(
                np.quantile(np.abs(delta_logp), 0.95)
            ),
            "preservation_q99_abs_delta_log_prob": float(
                np.quantile(np.abs(delta_logp), 0.99)
            ),
        })
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--g0-catalogue", type=Path, required=True)
    parser.add_argument("--sheared-catalogue", type=Path, required=True)
    parser.add_argument("--response-target", type=Path, required=True)
    parser.add_argument("--coupling-target", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--delta", type=float, default=0.02)
    parser.add_argument("--batch-size", type=int, default=65536)
    parser.add_argument("--reference-model", type=Path, default=None)
    parser.add_argument("--sample-draws", type=int, default=64)
    parser.add_argument("--sample-rows", type=int, default=200000)
    parser.add_argument("--sample-seed", type=int, default=7301)
    args = parser.parse_args(argv)
    audit(
        args.model,
        args.g0_catalogue,
        args.sheared_catalogue,
        args.response_target,
        args.coupling_target,
        args.output,
        device=args.device,
        delta=args.delta,
        batch_size=args.batch_size,
        reference_model_path=args.reference_model,
        sample_draws=args.sample_draws,
        sample_rows=args.sample_rows,
        sample_seed=args.sample_seed,
    )


if __name__ == "__main__":
    main()
