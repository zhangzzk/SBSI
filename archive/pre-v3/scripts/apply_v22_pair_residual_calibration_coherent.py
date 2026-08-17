"""Fit panel-B pair residuals and transfer the correction to coherent anchors.

The calibration target is the case-balanced half-shear curve

    delta(p) = mean(label - prediction | prediction = p).

The primary model is a continuous three-parameter hinge,

    delta(p) = a + b_neg min(p, 0) + b_pos max(p, 0),

and the corrected pair prediction is ``p + delta(p)``.  A four-parameter
two-sided saturating hinge, an exact natural cubic spline, and two fit-free
uses of the original panel-B curve (straight-line interpolation and
piecewise-constant bin lookup) are carried as sensitivity checks.  All
corrections are fixed only from the saved half-shear panel-B points.
Coherent-anchor truth is opened only after they are frozen.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd
from scipy import stats
from scipy.interpolate import CubicSpline
from scipy.optimize import least_squares


KEY = ["case", "input_index"]
PRIMARY_MODEL = "hinge"
SENSITIVITY_MODEL = "saturating_hinge"
SPLINE_MODEL = "natural_cubic_spline"
BIN_LOOKUP_MODEL = "binned_lookup"
LINEAR_INTERP_MODEL = "linear_interpolation"


def finite_stat(values: np.ndarray) -> dict[str, float | int]:
    """Case-level mean, spread, SEM, and two-sided zero test."""
    values = np.asarray(values, float)
    if values.ndim != 1 or values.size < 2 or not np.isfinite(values).all():
        raise ValueError("statistic needs at least two finite case values")
    sd = float(values.std(ddof=1))
    sem = sd / np.sqrt(values.size)
    t_value = float(values.mean() / sem) if sem > 0 else None
    p_value = (
        float(2.0 * stats.t.sf(abs(t_value), values.size - 1))
        if t_value is not None else None
    )
    return {
        "mean": float(values.mean()),
        "case_sd": sd,
        "case_sem": sem,
        "n_cases": int(values.size),
        "t": t_value,
        "p": p_value,
    }


def hinge_residual(prediction: np.ndarray, parameters: np.ndarray) -> np.ndarray:
    """Continuous piecewise-linear residual with a shared intercept."""
    a, b_negative, b_positive = np.asarray(parameters, float)
    prediction = np.asarray(prediction, float)
    return (
        a
        + b_negative * np.minimum(prediction, 0.0)
        + b_positive * np.maximum(prediction, 0.0)
    )


def saturating_hinge_residual(
    prediction: np.ndarray, parameters: np.ndarray,
) -> np.ndarray:
    """Two-sided hinge with one shared saturation scale."""
    a, b_negative, b_positive, scale = np.asarray(parameters, float)
    prediction = np.asarray(prediction, float)
    negative = np.minimum(prediction, 0.0)
    positive = np.maximum(prediction, 0.0)
    return (
        a
        + b_negative * negative / (1.0 + np.abs(negative) / scale)
        + b_positive * positive / (1.0 + positive / scale)
    )


def natural_cubic_spline_residual(
    prediction: np.ndarray, parameters: np.ndarray,
) -> np.ndarray:
    """Evaluate a natural cubic interpolator with endpoint clamping.

    ``parameters`` concatenates the strictly increasing prediction knots and
    their residual values.  Clamping is explicit because unconstrained cubic
    extrapolation outside the outer bin centers is not identified by panel B.
    """
    parameters = np.asarray(parameters, float)
    if parameters.ndim != 1 or parameters.size < 8 or parameters.size % 2:
        raise ValueError("spline parameters must contain equal knot/value halves")
    split = parameters.size // 2
    knots = parameters[:split]
    values = parameters[split:]
    if np.any(np.diff(knots) <= 0.0):
        raise ValueError("spline knots must be strictly increasing")
    prediction = np.asarray(prediction, float)
    spline = CubicSpline(knots, values, bc_type="natural", extrapolate=False)
    return spline(np.clip(prediction, knots[0], knots[-1]))


def binned_lookup_residual(
    prediction: np.ndarray, parameters: np.ndarray,
) -> np.ndarray:
    """Return the measured residual of the containing panel-B bin.

    ``parameters`` concatenates ``n + 1`` strictly increasing bin edges and
    ``n`` residual means.  Values outside the outer edges retain the first or
    last bin value; this rule is explicit even though the coherent-pair range
    is contained by the half-shear calibration range in the present transfer.
    """
    parameters = np.asarray(parameters, float)
    if parameters.ndim != 1 or parameters.size < 5 or not parameters.size % 2:
        raise ValueError("lookup parameters must contain n+1 edges and n values")
    n_bins = (parameters.size - 1) // 2
    edges = parameters[:n_bins + 1]
    values = parameters[n_bins + 1:]
    if values.size != n_bins or np.any(np.diff(edges) <= 0.0):
        raise ValueError("lookup bin edges must be strictly increasing")
    prediction = np.asarray(prediction, float)
    bin_index = np.searchsorted(edges[1:-1], prediction, side="right")
    return values[bin_index]


def linear_interpolation_residual(
    prediction: np.ndarray, parameters: np.ndarray,
) -> np.ndarray:
    """Interpolate the plotted panel-B points linearly with endpoint clamping."""
    parameters = np.asarray(parameters, float)
    if parameters.ndim != 1 or parameters.size < 4 or parameters.size % 2:
        raise ValueError("linear parameters must contain equal knot/value halves")
    split = parameters.size // 2
    knots = parameters[:split]
    values = parameters[split:]
    if np.any(np.diff(knots) <= 0.0):
        raise ValueError("linear interpolation knots must be strictly increasing")
    return np.interp(
        np.asarray(prediction, float), knots, values,
        left=values[0], right=values[-1],
    )


def _fit_summary(
    name: str,
    parameters: np.ndarray,
    covariance: np.ndarray,
    prediction: np.ndarray,
    residual: np.ndarray,
    sem: np.ndarray,
    pair_fraction: np.ndarray,
    function: Callable[[np.ndarray, np.ndarray], np.ndarray],
) -> dict[str, Any]:
    fitted = function(prediction, parameters)
    standardized = (residual - fitted) / sem
    dof = int(len(residual) - len(parameters))
    chi2 = float(np.square(standardized).sum())
    return {
        "name": name,
        "parameters": [float(item) for item in parameters],
        "parameter_case_sem_scaled": [
            float(item) for item in np.sqrt(np.diag(covariance))
        ],
        "chi2": chi2,
        "degrees_of_freedom": dof,
        "reduced_chi2": chi2 / dof,
        "aic_gaussian_known_sem": chi2 + 2.0 * len(parameters),
        "unweighted_rmse": float(np.sqrt(np.mean(np.square(residual - fitted)))),
        "maximum_absolute_standardized_residual": float(
            np.max(np.abs(standardized))
        ),
        "observed_pair_weighted_mean_residual": float(
            np.sum(pair_fraction * residual)
        ),
        "fitted_pair_weighted_mean_residual": float(
            np.sum(pair_fraction * fitted)
        ),
        "fitted_residual_at_bin_means": fitted.tolist(),
    }


def fit_hinge_curve(calibration: pd.DataFrame) -> dict[str, Any]:
    """Fit the primary inverse-case-variance weighted hinge."""
    prediction = calibration.prediction_mean.to_numpy(float)
    residual = calibration.label_minus_prediction_mean.to_numpy(float)
    sem = calibration.label_minus_prediction_case_sem.to_numpy(float)
    pair_fraction = calibration.pair_fraction.to_numpy(float)
    design = np.column_stack([
        np.ones(len(prediction)),
        np.minimum(prediction, 0.0),
        np.maximum(prediction, 0.0),
    ])
    weighted_design = design / sem[:, None]
    weighted_residual = residual / sem
    parameters, _, rank, _ = np.linalg.lstsq(
        weighted_design, weighted_residual, rcond=None,
    )
    if rank != design.shape[1]:
        raise RuntimeError("hinge calibration design is rank deficient")
    fitted = hinge_residual(prediction, parameters)
    dof = len(residual) - len(parameters)
    reduced_chi2 = np.square((residual - fitted) / sem).sum() / dof
    covariance = np.linalg.inv(weighted_design.T @ weighted_design) * reduced_chi2
    return _fit_summary(
        PRIMARY_MODEL, parameters, covariance, prediction, residual, sem,
        pair_fraction, hinge_residual,
    )


def fit_saturating_curve(calibration: pd.DataFrame) -> dict[str, Any]:
    """Fit the predeclared shared-scale saturating sensitivity model."""
    prediction = calibration.prediction_mean.to_numpy(float)
    residual = calibration.label_minus_prediction_mean.to_numpy(float)
    sem = calibration.label_minus_prediction_case_sem.to_numpy(float)
    pair_fraction = calibration.pair_fraction.to_numpy(float)
    result = least_squares(
        lambda parameters: (
            saturating_hinge_residual(prediction, parameters) - residual
        ) / sem,
        x0=np.asarray([1.0e-3, 3.0, 3.0, 2.0e-3]),
        bounds=(
            np.asarray([-0.1, 0.0, 0.0, 1.0e-6]),
            np.asarray([+0.1, 100.0, 100.0, 1.0]),
        ),
        max_nfev=100_000,
    )
    if not result.success:
        raise RuntimeError(f"saturating calibration fit failed: {result.message}")
    dof = len(residual) - len(result.x)
    reduced_chi2 = np.square(result.fun).sum() / dof
    covariance = np.linalg.pinv(result.jac.T @ result.jac) * reduced_chi2
    return _fit_summary(
        SENSITIVITY_MODEL, result.x, covariance, prediction, residual, sem,
        pair_fraction, saturating_hinge_residual,
    )


def fit_natural_cubic_spline(calibration: pd.DataFrame) -> dict[str, Any]:
    """Freeze an exact raw-prediction natural cubic interpolation."""
    prediction = calibration.prediction_mean.to_numpy(float)
    residual = calibration.label_minus_prediction_mean.to_numpy(float)
    sem = calibration.label_minus_prediction_case_sem.to_numpy(float)
    pair_fraction = calibration.pair_fraction.to_numpy(float)
    parameters = np.r_[prediction, residual]
    fitted = natural_cubic_spline_residual(prediction, parameters)
    # Resolve every interval densely so spline overshoot is measured rather
    # than hidden by evaluation only at its interpolation knots.
    grid = np.concatenate([
        np.linspace(prediction[index], prediction[index + 1], 1000,
                    endpoint=False)
        for index in range(len(prediction) - 1)
    ] + [prediction[-1:]])
    grid_values = natural_cubic_spline_residual(grid, parameters)
    standardized = (residual - fitted) / sem
    return {
        "name": SPLINE_MODEL,
        "parameters": parameters.tolist(),
        "n_knots": int(len(prediction)),
        "prediction_knots": prediction.tolist(),
        "residual_at_knots": residual.tolist(),
        "boundary_condition": "natural (zero second derivative at both ends)",
        "extrapolation": "prediction clamped to outer knot before evaluation",
        "chi2_at_interpolation_knots": float(np.square(standardized).sum()),
        "degrees_of_freedom": 0,
        "reduced_chi2": None,
        "aic_gaussian_known_sem": None,
        "unweighted_rmse_at_interpolation_knots": float(
            np.sqrt(np.mean(np.square(residual - fitted)))
        ),
        "observed_residual_range": [
            float(residual.min()), float(residual.max()),
        ],
        "between_knot_spline_range": [
            float(grid_values.min()), float(grid_values.max()),
        ],
        "maximum_overshoot_beyond_observed_range": float(max(
            grid_values.max() - residual.max(),
            residual.min() - grid_values.min(),
            0.0,
        )),
        "observed_pair_weighted_mean_residual": float(
            np.sum(pair_fraction * residual)
        ),
        "fitted_pair_weighted_mean_residual_at_knots": float(
            np.sum(pair_fraction * fitted)
        ),
        "fitted_residual_at_bin_means": fitted.tolist(),
    }


def make_binned_lookup(calibration: pd.DataFrame) -> dict[str, Any]:
    """Freeze the original panel-B bins as a fit-free step correction."""
    edges = np.r_[
        calibration.lower.to_numpy(float),
        calibration.upper.to_numpy(float)[-1],
    ]
    residual = calibration.label_minus_prediction_mean.to_numpy(float)
    pair_fraction = calibration.pair_fraction.to_numpy(float)
    prediction = calibration.prediction_mean.to_numpy(float)
    parameters = np.r_[edges, residual]
    direct = binned_lookup_residual(prediction, parameters)
    if not np.array_equal(direct, residual):
        raise RuntimeError("panel-B bin means do not map back to their own bins")
    return {
        "name": BIN_LOOKUP_MODEL,
        "parameters": parameters.tolist(),
        "n_bins": int(len(residual)),
        "prediction_edges": edges.tolist(),
        "residual_by_bin": residual.tolist(),
        "estimation": "direct saved bin means; no fitted parameters",
        "outside_edge_rule": "retain the nearest outer-bin residual",
        "observed_pair_weighted_mean_residual": float(
            np.sum(pair_fraction * residual)
        ),
        "fitted_residual_at_bin_means": direct.tolist(),
    }


def make_linear_interpolation(calibration: pd.DataFrame) -> dict[str, Any]:
    """Freeze the displayed straight segments through panel-B bin means."""
    prediction = calibration.prediction_mean.to_numpy(float)
    residual = calibration.label_minus_prediction_mean.to_numpy(float)
    pair_fraction = calibration.pair_fraction.to_numpy(float)
    parameters = np.r_[prediction, residual]
    direct = linear_interpolation_residual(prediction, parameters)
    if not np.array_equal(direct, residual):
        raise RuntimeError("linear curve does not reproduce its panel-B knots")
    return {
        "name": LINEAR_INTERP_MODEL,
        "parameters": parameters.tolist(),
        "n_knots": int(len(prediction)),
        "prediction_knots": prediction.tolist(),
        "residual_at_knots": residual.tolist(),
        "estimation": "straight segments through saved bin means; no fit",
        "outside_knot_rule": "retain the nearest endpoint residual",
        "observed_pair_weighted_mean_residual_at_knots": float(
            np.sum(pair_fraction * residual)
        ),
        "fitted_residual_at_bin_means": direct.tolist(),
    }


def calibration_components(
    prediction: np.ndarray, model: str, parameters: np.ndarray,
) -> dict[str, np.ndarray]:
    """Return additive pair-correction terms for one frozen model."""
    prediction = np.asarray(prediction, float)
    if model == PRIMARY_MODEL:
        a, b_negative, b_positive = np.asarray(parameters, float)
        negative = b_negative * np.minimum(prediction, 0.0)
        positive = b_positive * np.maximum(prediction, 0.0)
    elif model == SENSITIVITY_MODEL:
        a, b_negative, b_positive, scale = np.asarray(parameters, float)
        raw_negative = np.minimum(prediction, 0.0)
        raw_positive = np.maximum(prediction, 0.0)
        negative = (
            b_negative * raw_negative
            / (1.0 + np.abs(raw_negative) / scale)
        )
        positive = (
            b_positive * raw_positive
            / (1.0 + raw_positive / scale)
        )
    elif model in (SPLINE_MODEL, BIN_LOOKUP_MODEL, LINEAR_INTERP_MODEL):
        function = {
            SPLINE_MODEL: natural_cubic_spline_residual,
            BIN_LOOKUP_MODEL: binned_lookup_residual,
            LINEAR_INTERP_MODEL: linear_interpolation_residual,
        }[model]
        total = function(prediction, parameters)
        intercept = np.zeros(prediction.shape, dtype=float)
        negative = np.where(prediction < 0.0, total, 0.0)
        positive = np.where(prediction >= 0.0, total, 0.0)
        return {
            "intercept": intercept,
            "negative_branch": negative,
            "positive_branch": positive,
            "total": total,
        }
    else:
        raise ValueError(f"unknown calibration model {model}")
    intercept = np.full(prediction.shape, a, dtype=float)
    return {
        "intercept": intercept,
        "negative_branch": negative,
        "positive_branch": positive,
        "total": intercept + negative + positive,
    }


def validate_calibration(calibration: pd.DataFrame) -> None:
    required = {
        "bin", "lower", "upper", "prediction_mean", "label_minus_prediction_mean",
        "label_minus_prediction_case_sem", "pair_fraction",
    }
    if missing := required - set(calibration):
        raise KeyError(f"calibration CSV lacks {sorted(missing)}")
    values = calibration[list(required - {"bin"})].to_numpy(float)
    if len(calibration) < 6 or not np.isfinite(values).all():
        raise ValueError("calibration curve is too short or non-finite")
    if (calibration.label_minus_prediction_case_sem <= 0).any():
        raise ValueError("calibration SEM must be positive")
    if not np.isclose(calibration.pair_fraction.sum(), 1.0, atol=2e-8):
        raise ValueError("calibration pair fractions do not sum to one")
    lower = calibration.lower.to_numpy(float)
    upper = calibration.upper.to_numpy(float)
    prediction = calibration.prediction_mean.to_numpy(float)
    if np.any(lower >= upper) or not np.allclose(
        lower[1:], upper[:-1], rtol=1.0e-12, atol=1.0e-15,
    ):
        raise ValueError("calibration bins are not contiguous and increasing")
    if np.any((prediction < lower) | (prediction > upper)):
        raise ValueError("a calibration prediction mean lies outside its bin")


def discover_pair_paths(
    design_paths: list[str], case_min: int, case_max: int,
) -> tuple[dict[int, str], list[dict[str, Any]]]:
    """Resolve exact renderer-pair files from their saved design records."""
    paths: dict[int, str] = {}
    designs = []
    for design_path in design_paths:
        with open(design_path, encoding="utf-8") as handle:
            design = json.load(handle)
        if design.get("catalogue_source") != "rendered":
            raise RuntimeError(f"{design_path}: pair catalogue is not renderer-exact")
        if design.get("tag") != "lsst_r_extnbr_v22":
            raise RuntimeError(f"{design_path}: unexpected model tag")
        manifest = Path(design["manifest_dir"])
        prefix = design["pair_prefix"]
        for item in design["per_case"]:
            case = int(item["case"])
            if case in paths:
                raise RuntimeError(f"duplicate pair design for case {case}")
            paths[case] = str(manifest / f"{prefix}_case{case}.feather")
        designs.append({
            "path": os.path.abspath(design_path),
            "manifest_dir": str(manifest),
            "pair_prefix": prefix,
            "cases": [
                int(design["per_case"][0]["case"]),
                int(design["per_case"][-1]["case"]),
            ],
        })
    expected = set(range(case_min, case_max + 1))
    missing = sorted(expected - set(paths))
    extra = sorted(set(paths) - expected)
    if missing or extra:
        raise RuntimeError(
            f"pair-design case mismatch: missing={missing[:8]}, extra={extra[:8]}"
        )
    absent = [path for path in paths.values() if not os.path.isfile(path)]
    if absent:
        raise FileNotFoundError(f"missing {len(absent)} pair files")
    return paths, designs


def read_references(
    reference_paths: list[str], case_min: int, case_max: int,
) -> pd.DataFrame:
    columns = [
        *KEY, "R_blend_truth", "R_blend_lsst_r_extnbr_v22",
    ]
    frame = pd.concat([
        pd.read_feather(path, columns=columns) for path in reference_paths
    ], ignore_index=True)
    frame = frame.loc[frame.case.between(case_min, case_max)].copy()
    if frame.empty or frame.duplicated(KEY).any():
        raise RuntimeError("coherent reference is empty or has duplicate keys")
    observed_cases = set(frame.case.astype(int).unique())
    expected_cases = set(range(case_min, case_max + 1))
    if observed_cases != expected_cases:
        raise RuntimeError("coherent reference does not cover the exact case window")
    values = frame[["R_blend_truth", "R_blend_lsst_r_extnbr_v22"]].to_numpy(float)
    if not np.isfinite(values).all():
        raise RuntimeError("coherent reference contains non-finite responses")
    return frame


def apply_to_coherent(
    reference: pd.DataFrame,
    pair_paths: dict[int, str],
    fits: dict[str, dict[str, Any]],
    replay_tolerance: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply both frozen pair corrections and return one row per anchor."""
    parts = []
    maximum_replay = 0.0
    manifest_pair_count = 0
    pair_count = 0
    negative_count = 0
    response_min = np.inf
    response_max = -np.inf
    for progress, case in enumerate(sorted(pair_paths), start=1):
        local_reference = reference.loc[reference.case == case].copy()
        pairs = pd.read_feather(
            pair_paths[case],
            columns=["case", "anchor_index", "secondary_index", "response"],
        )
        if len(pairs) and not pairs.case.eq(case).all():
            raise RuntimeError(f"case {case}: pair file contains another case")
        if pairs.duplicated(["anchor_index", "secondary_index"]).any():
            raise RuntimeError(f"case {case}: duplicate deployed pair")
        manifest_pair_count += len(pairs)
        pairs = pairs.loc[
            pairs.anchor_index.isin(local_reference.input_index)
        ].copy()
        response = pairs.response.to_numpy(float)
        if not np.isfinite(response).all():
            raise RuntimeError(f"case {case}: non-finite pair prediction")
        pair_count += len(response)
        negative_count += int((response < 0).sum())
        if len(response):
            response_min = min(response_min, float(response.min()))
            response_max = max(response_max, float(response.max()))

        pair_columns: dict[str, np.ndarray] = {
            "raw_pair": response,
            "n_pairs": np.ones(len(response), dtype=np.int16),
        }
        for model, fit in fits.items():
            components = calibration_components(
                response, model, np.asarray(fit["parameters"], float),
            )
            for component, values in components.items():
                pair_columns[f"{model}_{component}"] = values
        pair_frame = pd.DataFrame({"input_index": pairs.anchor_index, **pair_columns})
        aggregate = pair_frame.groupby("input_index", sort=False).sum()
        aggregate.index.name = "input_index"
        local = local_reference.set_index("input_index").join(
            aggregate, how="left",
        )
        aggregate_columns = list(pair_columns)
        local[aggregate_columns] = local[aggregate_columns].fillna(0.0)
        replay = float(np.max(np.abs(
            local.raw_pair.to_numpy(float)
            - local.R_blend_lsst_r_extnbr_v22.to_numpy(float)
        )))
        maximum_replay = max(maximum_replay, replay)
        if replay > replay_tolerance:
            raise RuntimeError(f"case {case}: raw pair replay mismatch {replay:.3e}")
        local["prediction_raw"] = local.raw_pair
        local["gap_truth_minus_raw"] = (
            local.R_blend_truth - local.prediction_raw
        )
        for model in fits:
            local[f"prediction_{model}"] = (
                local.prediction_raw + local[f"{model}_total"]
            )
            local[f"gap_truth_minus_{model}"] = (
                local.R_blend_truth - local[f"prediction_{model}"]
            )
        parts.append(local.reset_index())
        if progress % 25 == 0 or progress == len(pair_paths):
            print(
                f"applied frozen calibration to {progress}/{len(pair_paths)} cases",
                flush=True,
            )
    anchors = pd.concat(parts, ignore_index=True)
    support = {
        "n_manifest_pairs_before_response_join": int(manifest_pair_count),
        "n_scored_pairs": int(pair_count),
        "n_anchors": int(len(anchors)),
        "mean_scored_pairs_per_anchor_pooled": float(
            pair_count / len(anchors)
        ),
        "negative_scored_pair_fraction": float(negative_count / pair_count),
        "scored_pair_prediction_minimum": float(response_min),
        "scored_pair_prediction_maximum": float(response_max),
        "maximum_raw_scene_replay_error": float(maximum_replay),
        "raw_scene_replay_tolerance": float(replay_tolerance),
    }
    return anchors, support


def summarize_population(frame: pd.DataFrame, fits: dict[str, Any]) -> dict[str, Any]:
    columns = [
        "R_blend_truth", "prediction_raw", "gap_truth_minus_raw", "n_pairs",
    ]
    for model in fits:
        columns.extend([
            f"prediction_{model}", f"{model}_total",
            f"gap_truth_minus_{model}",
        ])
        for component in ("intercept", "negative_branch", "positive_branch"):
            columns.append(f"{model}_{component}")
    case = frame.groupby("case", sort=True)[columns].mean()
    result: dict[str, Any] = {
        "n_anchors": int(len(frame)),
        "n_cases": int(len(case)),
        "truth": finite_stat(case.R_blend_truth.to_numpy(float)),
        "raw_prediction": finite_stat(case.prediction_raw.to_numpy(float)),
        "raw_gap_truth_minus_prediction": finite_stat(
            case.gap_truth_minus_raw.to_numpy(float)
        ),
        "mean_pairs_per_anchor": finite_stat(case.n_pairs.to_numpy(float)),
    }
    raw_mean = result["raw_gap_truth_minus_prediction"]["mean"]
    result["models"] = {}
    for model in fits:
        correction = finite_stat(case[f"{model}_total"].to_numpy(float))
        remaining = finite_stat(
            case[f"gap_truth_minus_{model}"].to_numpy(float)
        )
        model_result = {
            "corrected_prediction": finite_stat(
                case[f"prediction_{model}"].to_numpy(float)
            ),
            "additive_correction": correction,
            "remaining_gap_truth_minus_prediction": remaining,
            "fraction_of_raw_gap_removed": (
                float(correction["mean"] / raw_mean) if raw_mean != 0 else None
            ),
            "remaining_gap_fraction": (
                float(remaining["mean"] / raw_mean) if raw_mean != 0 else None
            ),
            "component_decomposition": {
                component: finite_stat(
                    case[f"{model}_{component}"].to_numpy(float)
                )
                for component in (
                    "intercept", "negative_branch", "positive_branch"
                )
            },
        }
        result["models"][model] = model_result
    return result


def population_summaries(
    anchors: pd.DataFrame, fits: dict[str, Any], case_min: int, case_max: int,
) -> dict[str, Any]:
    groups: dict[str, tuple[int, int]] = {
        "all": (case_min, case_max),
    }
    if case_min <= 400 and case_max >= 899:
        groups.update({
            "c400_599": (400, 599),
            "c600_699": (600, 699),
            "c700_899": (700, 899),
        })
    start = case_min
    while start <= case_max:
        stop = min(start + 99, case_max)
        groups[f"c{start}_{stop}"] = (start, stop)
        start = stop + 1
    output = {}
    for name, (lo, hi) in groups.items():
        local = anchors.loc[anchors.case.between(lo, hi)]
        if local.case.nunique() < 2:
            continue
        output[name] = {
            "case_window": [int(lo), int(hi)],
            **summarize_population(local, fits),
        }
    return output


def conditional_deciles(
    anchors: pd.DataFrame, fits: dict[str, Any], n_bins: int = 10,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    prediction = anchors.prediction_raw.to_numpy(float)
    edges = np.unique(np.quantile(prediction, np.linspace(0, 1, n_bins + 1)))
    if len(edges) != n_bins + 1:
        raise RuntimeError("coherent scene-prediction decile edges collapsed")
    bin_index = np.searchsorted(edges[1:-1], prediction, side="right")
    rows = []
    for index in range(n_bins):
        local = anchors.loc[bin_index == index]
        columns = ["prediction_raw", "gap_truth_minus_raw"] + [
            f"gap_truth_minus_{model}" for model in fits
        ]
        case = local.groupby("case", sort=True)[columns].mean()
        rows.append({
            "bin": int(index),
            "lower": float(edges[index]),
            "upper": float(edges[index + 1]),
            "n_anchors": int(len(local)),
            "n_cases_with_anchors": int(len(case)),
            "raw_scene_prediction": finite_stat(
                case.prediction_raw.to_numpy(float)
            ),
            "raw_gap_truth_minus_prediction": finite_stat(
                case.gap_truth_minus_raw.to_numpy(float)
            ),
            "models": {
                model: finite_stat(
                    case[f"gap_truth_minus_{model}"].to_numpy(float)
                ) for model in fits
            },
        })
    return rows, edges


def conditional_metrics(
    deciles: list[dict[str, Any]], fits: dict[str, Any],
) -> dict[str, Any]:
    """Summarize unweighted conditional mean error over equal-count deciles."""
    raw = np.asarray([
        item["raw_gap_truth_minus_prediction"]["mean"] for item in deciles
    ])
    output: dict[str, Any] = {
        "raw": {
            "decile_rmse": float(np.sqrt(np.mean(np.square(raw)))),
            "maximum_absolute_decile_gap": float(np.max(np.abs(raw))),
        },
        "models": {},
    }
    for model in fits:
        values = np.asarray([
            item["models"][model]["mean"] for item in deciles
        ])
        output["models"][model] = {
            "decile_rmse": float(np.sqrt(np.mean(np.square(values)))),
            "maximum_absolute_decile_gap": float(np.max(np.abs(values))),
            "rmse_ratio_to_raw": float(
                np.sqrt(np.mean(np.square(values)))
                / output["raw"]["decile_rmse"]
            ),
        }
    return output


def configure_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 8.0,
        "axes.labelsize": 8.5,
        "axes.titlesize": 8.5,
        "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.0,
        "legend.fontsize": 7.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": "white",
    })


def plot_results(
    calibration: pd.DataFrame,
    fits: dict[str, dict[str, Any]],
    populations: dict[str, Any],
    deciles: list[dict[str, Any]],
    output_prefix: str,
) -> None:
    configure_style()
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.45))
    x = calibration.prediction_mean.to_numpy(float)
    y = calibration.label_minus_prediction_mean.to_numpy(float)
    yerr = calibration.label_minus_prediction_case_sem.to_numpy(float)
    axes[0].errorbar(
        x, y, yerr=yerr, color="#D55E00", marker="o", markersize=3.7,
        linewidth=1.0, capsize=1.8, label="Panel-B means",
    )
    negative = -np.geomspace(1.0e-6, max(abs(x.min()), 1.0e-6), 240)[::-1]
    positive = np.geomspace(1.0e-6, max(x.max(), 1.0e-6), 240)
    grid = np.r_[negative, 0.0, positive]
    axes[0].plot(
        grid,
        hinge_residual(grid, np.asarray(fits[PRIMARY_MODEL]["parameters"])),
        color="#0072B2", linewidth=1.5, label="Hinge fit (primary)",
    )
    axes[0].plot(
        grid,
        saturating_hinge_residual(
            grid, np.asarray(fits[SENSITIVITY_MODEL]["parameters"])
        ),
        color="0.25", linestyle="--", linewidth=1.15,
        label="Saturating sensitivity",
    )
    axes[0].plot(
        grid,
        natural_cubic_spline_residual(
            grid, np.asarray(fits[SPLINE_MODEL]["parameters"])
        ),
        color="#CC79A7", linestyle=":", linewidth=1.35,
        label="Natural cubic spline",
    )
    axes[0].plot(
        grid,
        linear_interpolation_residual(
            grid, np.asarray(fits[LINEAR_INTERP_MODEL]["parameters"])
        ),
        color="#56B4E9", linewidth=1.25, label="Direct linear curve",
    )
    lookup_edges = np.asarray(
        fits[BIN_LOOKUP_MODEL]["prediction_edges"], float,
    )
    axes[0].stairs(
        y, lookup_edges, color="#009E73", linestyle="-.", linewidth=1.15,
        label="Direct bin lookup",
    )
    axes[0].axhline(0.0, color="0.55", linestyle=":", linewidth=0.8)
    axes[0].set_xscale("symlog", linthresh=1.0e-3, linscale=1.0)
    axes[0].set_xlim(1.12 * x.min(), 1.08 * x.max())
    axes[0].set_xlabel(r"V2.2 pair prediction $p$")
    axes[0].set_ylabel(r"Mean label $-$ prediction")
    axes[0].set_title("Half-shear calibration curves")
    axes[0].legend(frameon=False, loc="best")

    group_order = [
        name for name in ("all", "c400_599", "c600_699", "c700_899")
        if name in populations
    ]
    labels = {
        "all": "All\n400-899", "c400_599": "400-599",
        "c600_699": "600-699", "c700_899": "700-899",
    }
    positions = np.arange(len(group_order), dtype=float)
    raw = np.asarray([
        populations[name]["raw_gap_truth_minus_prediction"]["mean"]
        for name in group_order
    ])
    raw_sem = np.asarray([
        populations[name]["raw_gap_truth_minus_prediction"]["case_sem"]
        for name in group_order
    ])
    corrected = np.asarray([
        populations[name]["models"][PRIMARY_MODEL][
            "remaining_gap_truth_minus_prediction"
        ]["mean"] for name in group_order
    ])
    corrected_sem = np.asarray([
        populations[name]["models"][PRIMARY_MODEL][
            "remaining_gap_truth_minus_prediction"
        ]["case_sem"] for name in group_order
    ])
    axes[1].errorbar(
        positions - 0.375, raw, yerr=raw_sem, color="#0072B2", marker="o",
        linestyle="none", capsize=2.2, label="Raw V2.2",
    )
    axes[1].errorbar(
        positions - 0.225, corrected, yerr=corrected_sem, color="#D55E00",
        marker="s", linestyle="none", capsize=2.2, label="Hinge-corrected",
    )
    sensitivity = np.asarray([
        populations[name]["models"][SENSITIVITY_MODEL][
            "remaining_gap_truth_minus_prediction"
        ]["mean"] for name in group_order
    ])
    sensitivity_sem = np.asarray([
        populations[name]["models"][SENSITIVITY_MODEL][
            "remaining_gap_truth_minus_prediction"
        ]["case_sem"] for name in group_order
    ])
    axes[1].errorbar(
        positions - 0.075, sensitivity, yerr=sensitivity_sem, color="0.25",
        marker="D", markerfacecolor="white", linestyle="none", capsize=2.2,
        label="Saturating sensitivity",
    )
    spline = np.asarray([
        populations[name]["models"][SPLINE_MODEL][
            "remaining_gap_truth_minus_prediction"
        ]["mean"] for name in group_order
    ])
    spline_sem = np.asarray([
        populations[name]["models"][SPLINE_MODEL][
            "remaining_gap_truth_minus_prediction"
        ]["case_sem"] for name in group_order
    ])
    axes[1].errorbar(
        positions + 0.075, spline, yerr=spline_sem, color="#CC79A7",
        marker="^", markerfacecolor="white", linestyle="none", capsize=2.2,
        label="Natural cubic spline",
    )
    linear = np.asarray([
        populations[name]["models"][LINEAR_INTERP_MODEL][
            "remaining_gap_truth_minus_prediction"
        ]["mean"] for name in group_order
    ])
    linear_sem = np.asarray([
        populations[name]["models"][LINEAR_INTERP_MODEL][
            "remaining_gap_truth_minus_prediction"
        ]["case_sem"] for name in group_order
    ])
    axes[1].errorbar(
        positions + 0.225, linear, yerr=linear_sem, color="#56B4E9",
        marker="P", markerfacecolor="white", linestyle="none", capsize=2.2,
        label="Direct linear curve",
    )
    lookup = np.asarray([
        populations[name]["models"][BIN_LOOKUP_MODEL][
            "remaining_gap_truth_minus_prediction"
        ]["mean"] for name in group_order
    ])
    lookup_sem = np.asarray([
        populations[name]["models"][BIN_LOOKUP_MODEL][
            "remaining_gap_truth_minus_prediction"
        ]["case_sem"] for name in group_order
    ])
    axes[1].errorbar(
        positions + 0.375, lookup, yerr=lookup_sem, color="#009E73",
        marker="v", markerfacecolor="white", linestyle="none", capsize=2.2,
        label="Direct bin lookup",
    )
    axes[1].axhline(0.0, color="0.45", linestyle="--", linewidth=0.8)
    axes[1].set_xticks(positions, [labels[name] for name in group_order])
    axes[1].set_ylabel(r"Coherent truth $-$ prediction")
    axes[1].set_title("Coherent-anchor gap")
    axes[1].legend(frameon=False)

    decile_x = np.asarray([
        item["raw_scene_prediction"]["mean"] for item in deciles
    ])
    for field, color, marker, label in (
        ("raw_gap_truth_minus_prediction", "#0072B2", "o", "Raw V2.2"),
        (PRIMARY_MODEL, "#D55E00", "s", "Hinge-corrected"),
        (
            SENSITIVITY_MODEL, "0.25", "D",
            "Saturating sensitivity",
        ),
        (SPLINE_MODEL, "#CC79A7", "^", "Natural cubic spline"),
        (LINEAR_INTERP_MODEL, "#56B4E9", "P", "Direct linear curve"),
        (BIN_LOOKUP_MODEL, "#009E73", "v", "Direct bin lookup"),
    ):
        if field in fits:
            mean = np.asarray([item["models"][field]["mean"] for item in deciles])
            sem = np.asarray([
                item["models"][field]["case_sem"] for item in deciles
            ])
        else:
            mean = np.asarray([item[field]["mean"] for item in deciles])
            sem = np.asarray([item[field]["case_sem"] for item in deciles])
        axes[2].errorbar(
            decile_x, mean, yerr=sem, color=color, marker=marker,
            markerfacecolor=(
                "white" if field in (
                    SENSITIVITY_MODEL, SPLINE_MODEL, LINEAR_INTERP_MODEL,
                    BIN_LOOKUP_MODEL,
                ) else color
            ),
            markersize=3.7, linewidth=1.0,
            linestyle=(
                "--" if field == SENSITIVITY_MODEL
                else ":" if field == SPLINE_MODEL
                else "-" if field == LINEAR_INTERP_MODEL
                else "-." if field == BIN_LOOKUP_MODEL else "-"
            ),
            capsize=1.8, label=label,
        )
    axes[2].axhline(0.0, color="0.45", linestyle="--", linewidth=0.8)
    axes[2].set_xscale("symlog", linthresh=1.0e-3, linscale=1.0)
    axes[2].set_xlabel("Raw V2.2 scene prediction")
    axes[2].set_ylabel(r"Coherent truth $-$ prediction")
    axes[2].set_title("Conditional coherent gap")
    axes[2].legend(frameon=False)

    for index, axis in enumerate(axes):
        axis.spines[["top", "right"]].set_visible(False)
        axis.text(
            -0.14, 1.04, chr(ord("A") + index), transform=axis.transAxes,
            fontsize=10, fontweight="bold", va="top",
        )
    fig.suptitle(
        "Panel-B pair calibration transferred without fitting to coherent anchors\n"
        "Error bars are one SEM across rendered cases",
        fontsize=9.5, y=0.99,
    )
    fig.subplots_adjust(
        left=0.075, right=0.99, bottom=0.19, top=0.77, wspace=0.34,
    )
    fig.savefig(f"{output_prefix}.pdf", bbox_inches="tight")
    fig.savefig(f"{output_prefix}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def flatten_deciles(deciles: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for item in deciles:
        row = {
            "bin": item["bin"], "lower": item["lower"],
            "upper": item["upper"], "n_anchors": item["n_anchors"],
            "scene_prediction_mean": item["raw_scene_prediction"]["mean"],
            "raw_gap_mean": item["raw_gap_truth_minus_prediction"]["mean"],
            "raw_gap_case_sem": item[
                "raw_gap_truth_minus_prediction"
            ]["case_sem"],
        }
        for model, value in item["models"].items():
            row[f"{model}_gap_mean"] = value["mean"]
            row[f"{model}_gap_case_sem"] = value["case_sem"]
        rows.append(row)
    return pd.DataFrame(rows)


def markdown_report(payload: dict[str, Any]) -> str:
    fit = payload["calibration_fits"][PRIMARY_MODEL]
    sensitivity = payload["calibration_fits"][SENSITIVITY_MODEL]
    spline_fit = payload["calibration_fits"][SPLINE_MODEL]
    lookup_curve = payload["calibration_fits"][BIN_LOOKUP_MODEL]
    linear_curve = payload["calibration_fits"][LINEAR_INTERP_MODEL]
    all_result = payload["coherent_populations"]["all"]
    raw = all_result["raw_gap_truth_minus_prediction"]
    corrected = all_result["models"][PRIMARY_MODEL]
    remaining = corrected["remaining_gap_truth_minus_prediction"]
    alternative = all_result["models"][SENSITIVITY_MODEL]
    spline = all_result["models"][SPLINE_MODEL]
    lookup = all_result["models"][BIN_LOOKUP_MODEL]
    linear = all_result["models"][LINEAR_INTERP_MODEL]
    conditional = payload["coherent_conditional_metrics"]
    component = corrected["component_decomposition"]
    parameters = fit["parameters"]
    return "\n".join([
        "# Panel-B pair corrections on coherent anchors",
        "",
        "The primary correction is fitted only to the saved half-shear panel-B "
        "curve: `delta(p) = a + b_neg min(p,0) + b_pos max(p,0)`. "
        "No coherent-anchor label enters the fit or model choice.",
        "",
        f"- Hinge parameters `(a, b_neg, b_pos)`: "
        f"`({parameters[0]:+.8f}, {parameters[1]:+.6f}, "
        f"{parameters[2]:+.6f})`.",
        f"- Hinge fit reduced chi-square: `{fit['reduced_chi2']:.3f}` "
        f"for `{fit['degrees_of_freedom']}` degrees of freedom.",
        f"- Saturating sensitivity reduced chi-square: "
        f"`{sensitivity['reduced_chi2']:.3f}`.",
        f"- Natural cubic spline interpolates all `{spline_fit['n_knots']}` "
        f"bin means exactly; its between-knot range is "
        f"`[{spline_fit['between_knot_spline_range'][0]:+.6f}, "
        f"{spline_fit['between_knot_spline_range'][1]:+.6f}]`, versus the "
        f"observed range `[{spline_fit['observed_residual_range'][0]:+.6f}, "
        f"{spline_fit['observed_residual_range'][1]:+.6f}]`.",
        f"- Direct lookup uses the original `{lookup_curve['n_bins']}` bin "
        "boundaries and residual means with no fitting or interpolation.",
        f"- Direct linear interpolation joins the original "
        f"`{linear_curve['n_knots']}` plotted bin means with straight segments; "
        "it also has no fitted parameters.",
        f"- Coherent cases: `{payload['case_window'][0]}--"
        f"{payload['case_window'][1]}`; anchors: "
        f"`{all_result['n_anchors']:,}`; mean deployed pairs per anchor: "
        f"`{all_result['mean_pairs_per_anchor']['mean']:.3f}`.",
        f"- Raw coherent truth minus V2.2: `{raw['mean']:+.6f} +- "
        f"{raw['case_sem']:.6f}`.",
        f"- Hinge additive correction: "
        f"`{corrected['additive_correction']['mean']:+.6f} +- "
        f"{corrected['additive_correction']['case_sem']:.6f}`.",
        f"- Hinge remaining gap: `{remaining['mean']:+.6f} +- "
        f"{remaining['case_sem']:.6f}`; removed "
        f"`{100.0 * corrected['fraction_of_raw_gap_removed']:.1f}%` of the "
        "raw mean gap.",
        f"- Saturating sensitivity removes "
        f"`{100.0 * alternative['fraction_of_raw_gap_removed']:.1f}%`, leaving "
        f"`{alternative['remaining_gap_truth_minus_prediction']['mean']:+.6f} "
        f"+- {alternative['remaining_gap_truth_minus_prediction']['case_sem']:.6f}`.",
        f"- Natural cubic spline removes "
        f"`{100.0 * spline['fraction_of_raw_gap_removed']:.1f}%`, leaving "
        f"`{spline['remaining_gap_truth_minus_prediction']['mean']:+.6f} "
        f"+- {spline['remaining_gap_truth_minus_prediction']['case_sem']:.6f}`.",
        f"- Direct bin lookup removes "
        f"`{100.0 * lookup['fraction_of_raw_gap_removed']:.1f}%`, leaving "
        f"`{lookup['remaining_gap_truth_minus_prediction']['mean']:+.6f} "
        f"+- {lookup['remaining_gap_truth_minus_prediction']['case_sem']:.6f}`.",
        f"- Direct linear interpolation removes "
        f"`{100.0 * linear['fraction_of_raw_gap_removed']:.1f}%`, leaving "
        f"`{linear['remaining_gap_truth_minus_prediction']['mean']:+.6f} "
        f"+- {linear['remaining_gap_truth_minus_prediction']['case_sem']:.6f}`.",
        f"- Equal-count coherent scene-decile RMSE: raw "
        f"`{conditional['raw']['decile_rmse']:.6f}`, hinge "
        f"`{conditional['models'][PRIMARY_MODEL]['decile_rmse']:.6f}`, "
        f"saturating `{conditional['models'][SENSITIVITY_MODEL]['decile_rmse']:.6f}`, "
        f"spline `{conditional['models'][SPLINE_MODEL]['decile_rmse']:.6f}`, "
        f"direct linear "
        f"`{conditional['models'][LINEAR_INTERP_MODEL]['decile_rmse']:.6f}`, "
        f"direct lookup "
        f"`{conditional['models'][BIN_LOOKUP_MODEL]['decile_rmse']:.6f}`.",
        f"- Hinge correction decomposition: count/intercept "
        f"`{component['intercept']['mean']:+.6f}`, negative branch "
        f"`{component['negative_branch']['mean']:+.6f}`, positive branch "
        f"`{component['positive_branch']['mean']:+.6f}`.",
        "",
        "Uncertainties on coherent gaps and corrections are one SEM across "
        "rendered cases. The parametric fit chi-square uses the plotted case SEMs as "
        "diagonal weights; it is descriptive because cross-bin case covariance "
        "was not stored in the original panel payload. The direct lookup does "
        "not use those errors to alter the saved bin means; neither does the "
        "direct linear interpolation.",
        "",
    ])


def json_clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: json_clean(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_clean(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_clean(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-csv", required=True)
    parser.add_argument("--calibration-json", required=True)
    parser.add_argument("--reference", nargs="+", required=True)
    parser.add_argument("--pair-design", nargs="+", required=True)
    parser.add_argument("--case-min", type=int, default=400)
    parser.add_argument("--case-max", type=int, default=899)
    parser.add_argument("--replay-tolerance", type=float, default=3.0e-7)
    parser.add_argument("--output-prefix", required=True)
    args = parser.parse_args()
    if args.case_min > args.case_max or args.replay_tolerance <= 0:
        raise ValueError("invalid case window or replay tolerance")
    outputs = [
        f"{args.output_prefix}.{suffix}"
        for suffix in ("json", "md", "png", "pdf")
    ] + [
        f"{args.output_prefix}_{suffix}.csv"
        for suffix in ("cases", "calibration_bins", "anchor_deciles")
    ]
    existing = [path for path in outputs if os.path.exists(path)]
    if existing:
        raise FileExistsError(f"refusing existing outputs: {existing}")
    Path(args.output_prefix).parent.mkdir(parents=True, exist_ok=True)

    calibration = pd.read_csv(args.calibration_csv)
    validate_calibration(calibration)
    with open(args.calibration_json, encoding="utf-8") as handle:
        calibration_payload = json.load(handle)
    if calibration_payload["n_pairs"] != int(calibration.n_pairs.sum()):
        raise RuntimeError("calibration CSV/JSON pair counts disagree")
    fits = {
        PRIMARY_MODEL: fit_hinge_curve(calibration),
        SENSITIVITY_MODEL: fit_saturating_curve(calibration),
        SPLINE_MODEL: fit_natural_cubic_spline(calibration),
        BIN_LOOKUP_MODEL: make_binned_lookup(calibration),
        LINEAR_INTERP_MODEL: make_linear_interpolation(calibration),
    }
    print(
        "froze half-shear calibration curves before opening anchor truth",
        flush=True,
    )
    print(json.dumps({name: fit["parameters"] for name, fit in fits.items()}), flush=True)

    pair_paths, design_provenance = discover_pair_paths(
        args.pair_design, args.case_min, args.case_max,
    )
    reference = read_references(args.reference, args.case_min, args.case_max)
    anchors, support = apply_to_coherent(
        reference, pair_paths, fits, args.replay_tolerance,
    )
    populations = population_summaries(
        anchors, fits, args.case_min, args.case_max,
    )
    deciles, decile_edges = conditional_deciles(anchors, fits)
    conditional_summary = conditional_metrics(deciles, fits)

    case_columns = [
        "R_blend_truth", "prediction_raw", "gap_truth_minus_raw", "n_pairs",
    ]
    for model in fits:
        case_columns.extend([
            f"prediction_{model}", f"{model}_total",
            f"gap_truth_minus_{model}", f"{model}_intercept",
            f"{model}_negative_branch", f"{model}_positive_branch",
        ])
    case_table = anchors.groupby("case", sort=True)[case_columns].mean().reset_index()
    case_table.to_csv(f"{args.output_prefix}_cases.csv", index=False)
    calibration_output = calibration.copy()
    for model, fit in fits.items():
        calibration_output[f"{model}_fitted_residual"] = fit[
            "fitted_residual_at_bin_means"
        ]
    calibration_output.to_csv(
        f"{args.output_prefix}_calibration_bins.csv", index=False,
    )
    flatten_deciles(deciles).to_csv(
        f"{args.output_prefix}_anchor_deciles.csv", index=False,
    )

    payload = json_clean({
        "title": "Panel-B pair residual calibration transferred to coherent anchors",
        "case_window": [args.case_min, args.case_max],
        "fit_firewall": (
            "both parametric models, the spline, the direct linear curve, and "
            "the direct binned lookup were frozen from the half-shear panel-B "
            "curve before coherent-anchor truth was opened"
        ),
        "calibration_source": {
            "csv": os.path.abspath(args.calibration_csv),
            "json": os.path.abspath(args.calibration_json),
            "n_pairs": calibration_payload["n_pairs"],
            "case_window": calibration_payload["case_window"],
            "evaluation_status": calibration_payload["evaluation_status"],
            "uncertainty_unit": calibration_payload["uncertainty_unit"],
        },
        "calibration_fits": fits,
        "coherent_reference_paths": [
            os.path.abspath(path) for path in args.reference
        ],
        "pair_designs": design_provenance,
        "coherent_support": support,
        "coherent_populations": populations,
        "coherent_scene_prediction_decile_edges": decile_edges,
        "coherent_scene_prediction_deciles": deciles,
        "coherent_conditional_metrics": conditional_summary,
        "interpretation_limits": [
            "half-shear calibration is in-sample for the deployed V2.2 emulator",
            "the three-parameter hinge is visibly misspecified at the curve level",
            "the exact raw-coordinate natural cubic spline overshoots between unevenly spaced knots",
            "the direct lookup is piecewise constant at the original half-shear quantile boundaries",
            "the direct linear curve joins the raw-coordinate bin means and clamps beyond the endpoint knots",
            "fit weights omit cross-bin covariance because the saved panel payload has no case-by-bin matrix",
            "coherent-anchor labels are evaluation only and do not alter the correction",
        ],
        "constgold_opened": False,
        "deployable_emulator_written": False,
    })
    with open(f"{args.output_prefix}.json", "x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.write("\n")
    Path(f"{args.output_prefix}.md").write_text(
        markdown_report(payload), encoding="utf-8",
    )
    plot_results(calibration, fits, populations, deciles, args.output_prefix)
    all_result = populations["all"]
    print(json.dumps({
        "raw_gap": all_result["raw_gap_truth_minus_prediction"],
        "hinge": all_result["models"][PRIMARY_MODEL],
        "saturating_hinge": all_result["models"][SENSITIVITY_MODEL],
        "natural_cubic_spline": all_result["models"][SPLINE_MODEL],
        "binned_lookup": all_result["models"][BIN_LOOKUP_MODEL],
        "linear_interpolation": all_result["models"][LINEAR_INTERP_MODEL],
        "support": support,
    }, indent=2, allow_nan=False), flush=True)
    print("V22_PAIR_RESIDUAL_CALIBRATION_COHERENT_DONE", flush=True)


if __name__ == "__main__":
    main()
