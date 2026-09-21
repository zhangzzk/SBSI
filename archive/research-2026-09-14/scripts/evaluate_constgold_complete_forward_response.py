#!/usr/bin/env python
"""Forward ConstGold response with a frozen event probability and shape flow.

For every requested ConstGold case, the model starts from the complete
truth-level population on the registered finite-prior support.  At each of the
two applied shears it computes

The default factorization predicts usable detection and integrates the measured
cut through the flow.  The optional direct-selected factorization instead uses
the exact selected-event probability and conditions the flow shape on that cut.

where C is the production measured-output cut.  Thus neither detection
membership nor measured selection is inherited from the image simulation.
The image-side comparator uses the actual per-leg cross-match and applies the
same measured-output cut to the measured catalogue.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.special import stdtrit

from sbsi.catalogue_likelihood import OutputCut, predict_detection_probability
from sbsi.coordinates import ellipticity_from_axis_ratio_angle
from sbsi.flow_size_condition import CIRCULARIZED_FEATURE, circularized_radius
from sbsi.measurement_model import load_measurement_model
from sbsi.models import ModelPaths, load_emulator
from sbsi.scene_prior import ScenePrior
from sbsi.selection_model import load_selection_model
from sbsi.shear_map import apply_shear_to_ellipticity


CONDITIONS = {
    "pixel_size": 0.2,
    "zero_point": 30.0,
    "psf_fwhm": 0.73,
    "moffat_beta": 2.224,
    "pixel_rms": 0.312,
}
TARGET_NAMES = (
    "measured_ngmix_g1",
    "measured_ngmix_g2",
    "measured_mag_auto",
    "measured_log_flux_radius",
)
CROWD_COLUMNS = ("nbr_flux_near", "nbr_flux_far", "nbr_flux_max")
RADIUS_MIN_ARCSEC = 0.75
ABS_SHAPE_MAX = 0.6
MAG_MAX = 25.8


def _qmc_confidence_multiplier(seed_count: int, confidence: float = 0.95) -> float | None:
    """Student-t multiplier for a two-sided QMC-replicate mean interval."""
    if seed_count < 1 or not 0.0 < confidence < 1.0:
        raise ValueError("invalid QMC seed count or confidence level")
    if seed_count == 1:
        return None
    return float(stdtrit(seed_count - 1, 0.5 + confidence / 2.0))


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _detector_feature_names(detector) -> tuple[str, ...]:
    if hasattr(detector, "preprocessor"):
        return tuple(detector.preprocessor.feature_names)
    return tuple(detector.bst_cla.feature_names)


class _SelectionProbabilityEnsemble:
    """Equal-probability wrapper compatible with older frozen SBSI sources."""

    def __init__(self, members):
        self.members = tuple(members)
        if not self.members:
            raise ValueError("selection-model ensemble cannot be empty")
        features = _detector_feature_names(self.members[0])
        if any(_detector_feature_names(member) != features for member in self.members[1:]):
            raise ValueError("selection-model ensemble feature contracts differ")
        self.preprocessor = self.members[0].preprocessor
        self.metadata = {
            "aggregation": "arithmetic_mean_probability",
            "n_members": len(self.members),
            "members": [dict(member.metadata) for member in self.members],
        }

    def predict_proba(self, frame, batch_size=65536):
        return np.mean(
            np.stack(
                [
                    member.predict_proba(frame, batch_size=batch_size)
                    for member in self.members
                ],
                axis=0,
            ),
            axis=0,
        )


def _load_selection_model_ensemble(paths, *, device):
    return _SelectionProbabilityEnsemble(
        [load_selection_model(path, device=device) for path in paths]
    )


def _with_flow_conditions(
    frame: pd.DataFrame,
    bundles,
    *,
    rblend: np.ndarray,
) -> pd.DataFrame:
    """Attach shared external conditions and reject an incomplete flow frame."""
    required = {
        name
        for bundle in bundles
        for name in bundle.condition_preprocessor.feature_names
    }
    result = frame
    if "R_blend" in required:
        values = np.asarray(rblend, dtype=np.float64)
        if values.shape != (len(frame),) or not np.isfinite(values).all():
            raise ValueError("R_blend must be one finite value per flow row")
        result = frame.copy()
        result["R_blend"] = values
    missing = sorted(required - set(result.columns))
    if missing:
        raise KeyError(f"flow inputs cannot be built: {missing}")
    return result


def _flow_seed_sem(responses: np.ndarray) -> list[float] | None:
    """Return ensemble-seed SEM, or null when only one checkpoint is present."""
    if len(responses) < 2:
        return None
    return (responses.std(axis=0, ddof=1) / np.sqrt(len(responses))).tolist()


def _prediction_run_diagnostics(
    responses: np.ndarray,
    *,
    model_count: int,
    sampling_seed_count: int,
) -> dict:
    """Keep model and QMC variation distinct while averaging all run pairs."""
    values = np.asarray(responses, dtype=np.float64)
    if values.shape != (model_count * sampling_seed_count, 2):
        raise ValueError("prediction responses do not match model-by-QMC run count")
    grid = values.reshape(model_count, sampling_seed_count, 2)
    by_model = grid.mean(axis=1)
    by_sampling_seed = grid.mean(axis=0)
    return {
        "response_by_prediction_run": values.tolist(),
        "response_by_flow_seed": by_model.tolist(),
        "flow_seed_sem": _flow_seed_sem(by_model),
        "response_by_sampling_seed": by_sampling_seed.tolist(),
        "sampling_seed_sem": _flow_seed_sem(by_sampling_seed),
    }


def _direct_selected_support_summary(
    predicted: list[dict],
    *,
    cases: tuple[int, ...],
    h: float,
    component_index: int,
) -> dict:
    """Pool and conservatively bound zero-acceptance direct-selected rows."""
    if not predicted or not cases or h <= 0.0 or component_index not in (0, 1):
        raise ValueError("invalid direct-selected support inputs")
    per_run = []
    per_case_run_bounds = np.zeros((len(cases), len(predicted)), dtype=np.float64)
    for run_index, run in enumerate(predicted):
        leg_mean_bounds = []
        for direction in (+1, -1):
            total_mass = 0.0
            missing_bound = np.zeros(2, dtype=np.float64)
            for case_index, case in enumerate(cases):
                support = run[direction][case]["direct_selected_conditional_support"]
                case_total = float(support["total_event_probability_mass"])
                case_bound = np.asarray(
                    support["missing_numerator_abs_bound"], dtype=np.float64
                )
                if case_total <= 0.0 or case_bound.shape != (2,):
                    raise ValueError("invalid direct-selected support sufficient statistic")
                total_mass += case_total
                missing_bound += case_bound
                per_case_run_bounds[case_index, run_index] += (
                    case_bound[component_index] / case_total / (2.0 * h)
                )
            leg_mean_bounds.append(missing_bound / total_mass)
        leg_mean_bounds = np.asarray(leg_mean_bounds)
        response_bound = (
            leg_mean_bounds[0] + leg_mean_bounds[1]
        ) / (2.0 * h)
        per_run.append(
            {
                "leg_mean_abs_bound": leg_mean_bounds.tolist(),
                "response_abs_bound": response_bound.tolist(),
            }
        )
    ensemble_response_bound = np.mean(
        [item["response_abs_bound"] for item in per_run], axis=0
    )
    maximum_case_ensemble_bound = float(
        per_case_run_bounds.mean(axis=1).max()
    )
    return {
        "definition": (
            "worst-case contribution of direct-event probability mass on rows "
            "with zero accepted conditional-flow draws; physical disk bound, "
            "zero midpoint imputation"
        ),
        "per_prediction_run": per_run,
        "pooled_ensemble_response_abs_bound": ensemble_response_bound.tolist(),
        "maximum_case_ensemble_response_abs_bound": maximum_case_ensemble_bound,
    }


def _raw_catalogue_root(root: Path, case: int, sign: float) -> Path:
    token = f"{float(sign):.2f}"
    return root / f"case{case}_{token}" / "real0" / "catalogues"


def _read_truth(path: Path) -> pd.DataFrame:
    raw = pd.read_feather(path)
    rename = {
        "index_input": "index",
        "RA_input": "RA",
        "DEC_input": "DEC",
        "redshift_input": "redshift",
        "Re_input": "Re",
        "axis_ratio_input": "axis_ratio",
        "position_angle_input": "position_angle",
        "sersic_n_input": "sersic_n",
        "r_input": "r",
    }
    missing = sorted(set(rename) - set(raw))
    if missing:
        raise KeyError(f"{path} lacks truth columns: {missing}")
    frame = raw.rename(columns=rename).loc[:, list(rename.values())].copy()
    if frame["index"].duplicated().any():
        raise ValueError(f"{path} contains duplicate input identities")
    return frame


def _active_rows(truth: pd.DataFrame, support: pd.DataFrame) -> np.ndarray:
    position = pd.Series(
        np.arange(len(truth), dtype=np.int64), index=truth["index"].to_numpy()
    )
    rows = position.reindex(support["input_index"].to_numpy()).to_numpy()
    if pd.isna(rows).any():
        raise RuntimeError("finite-prior support contains identities absent from truth")
    rows = rows.astype(np.int64)
    if len(np.unique(rows)) != len(rows):
        raise RuntimeError("finite-prior support does not map one-to-one onto truth")
    return rows


def _with_circularized_condition(frame: pd.DataFrame, baseline_truth: pd.DataFrame) -> pd.DataFrame:
    """Supply the invariant size from aligned PRE-shear truth for either leg."""
    if (len(frame) != len(baseline_truth)
            or not np.array_equal(frame["input_index"].to_numpy(), baseline_truth["index"].to_numpy())
            or baseline_truth["index"].duplicated().any()):
        raise ValueError("flow frame and baseline truth identities must be aligned and unique")
    # Match training's reconstruction from its unfurled baseline ellipticity.
    e1, e2 = ellipticity_from_axis_ratio_angle(
        baseline_truth["axis_ratio"].to_numpy(float),
        baseline_truth["position_angle"].to_numpy(float),
    )
    eabs = np.hypot(e1, e2)
    q0 = (1.0 - eabs) / (1.0 + eabs)
    result = frame.copy()
    result[CIRCULARIZED_FEATURE] = circularized_radius(baseline_truth["Re"].to_numpy(float), q0)
    return result


def _with_truth_crowding(
    support: pd.DataFrame,
    *,
    prior: ScenePrior,
    active: np.ndarray,
) -> pd.DataFrame:
    """Derive the three shear-invariant crowding inputs from the truth scene."""
    crowding = prior.shear(0.0, 0.0).object_view(
        neighbour_radius_arcsec=7.0,
        conditions=CONDITIONS,
        include_crowding=True,
        primary_indices=active,
    )
    if not np.array_equal(
        crowding["input_index"].to_numpy(), support["input_index"].to_numpy()
    ):
        raise RuntimeError("truth crowding/support alignment failed")
    result = support.copy()
    for name in CROWD_COLUMNS:
        values = crowding[name].to_numpy(float)
        if not np.isfinite(values).all():
            raise RuntimeError(f"truth-derived crowding feature {name!r} is nonfinite")
        result[name] = values
    return result


def _measured_leg(
    root: Path,
    *,
    case: int,
    sign: float,
    active_ids: np.ndarray,
    abs_shape_max: float | None,
    mag_max: float | None = MAG_MAX,
    radius_min_arcsec: float | None = RADIUS_MIN_ARCSEC,
) -> tuple[dict, dict, dict, np.ndarray, dict]:
    catalogue_root = _raw_catalogue_root(root, case, sign)
    cross_path = (
        catalogue_root / "CrossMatch" / "tile180.0_-0.5_rot0_matched.feather"
    )
    shape_path = (
        catalogue_root
        / "Shapes"
        / "shape_catalogue_detect_position_all_tile180.0_-0.5.feather"
    )
    cross = pd.read_feather(cross_path, columns=["id_detec", "id_input"])
    shape = pd.read_feather(
        shape_path,
        columns=["NUMBER", "NGMIX_G1", "NGMIX_G2", "MAG_AUTO", "FLUX_RADIUS"],
    )
    if cross["id_input"].duplicated().any() or cross["id_detec"].duplicated().any():
        raise ValueError(f"{cross_path} is not one-to-one")
    if shape["NUMBER"].duplicated().any():
        raise ValueError(f"{shape_path} contains duplicate detection identities")
    detected_all = cross.merge(
        shape,
        left_on="id_detec",
        right_on="NUMBER",
        how="inner",
        validate="one_to_one",
        sort=False,
    )
    cross_without_shape = int(len(cross) - len(detected_all))
    shape_without_cross = int(len(shape) - len(detected_all))
    active_mask = detected_all["id_input"].isin(active_ids)
    detected_outside_support = int((~active_mask).sum())
    detected = detected_all.loc[active_mask].copy()
    values = np.column_stack(
        (
            detected["NGMIX_G1"].to_numpy(float),
            detected["NGMIX_G2"].to_numpy(float),
            detected["MAG_AUTO"].to_numpy(float),
            np.log(detected["FLUX_RADIUS"].to_numpy(float)),
        )
    )
    bounds = []
    if mag_max is not None:
        bounds.append(("measured_mag_auto", None, mag_max))
    if radius_min_arcsec is not None:
        bounds.append(
            (
                "measured_log_flux_radius",
                np.log(radius_min_arcsec / CONDITIONS["pixel_size"]),
                None,
            )
        )
    cut = (
        None
        if abs_shape_max is None and not bounds
        else OutputCut(TARGET_NAMES, abs_shape=abs_shape_max, bounds=tuple(bounds))
    )
    finite = np.isfinite(values).all(axis=1)
    valid_shape = finite & ~(
        (values[:, 0] == -1.0)
        | ((values[:, 0] == 0.0) & (values[:, 1] == 0.0))
    )
    passed = (
        np.ones(len(values), dtype=bool)
        if cut is None
        else np.asarray(cut(values), dtype=bool)
    )
    selected = valid_shape & passed
    shape_values = values[selected, :2]
    selected_sufficient = {
        "count": int(selected.sum()),
        "sum": shape_values.sum(axis=0, dtype=np.float64),
    }
    detected_sufficient = {
        "count": int(valid_shape.sum()),
        "sum": values[valid_shape, :2].sum(axis=0, dtype=np.float64),
    }
    report = {
        "truth_population_count": int(len(active_ids)),
        "crossmatch_rows_without_shape": cross_without_shape,
        "shape_rows_without_crossmatch": shape_without_cross,
        "crossmatched_rows_outside_finite_prior_support": detected_outside_support,
        "actual_detected_count": int(len(detected)),
        "actual_detection_fraction": float(len(detected) / len(active_ids)),
        "actual_usable_shape_count": int(valid_shape.sum()),
        "actual_selected_count": int(selected.sum()),
        "actual_selected_fraction": float(selected.sum() / len(active_ids)),
        "dropped_nonfinite_measurements": int((~finite).sum()),
        "dropped_invalid_shape": int((finite & ~valid_shape).sum()),
        "dropped_measured_cut": int(
            (valid_shape & ~passed).sum()
        ),
    }
    identities = detected["id_input"].to_numpy(np.int64)
    populations = {
        "all_detected_ids": identities,
        "selected_ids": identities[selected],
        "selected_values": values[selected, :2],
        "detected_ids": identities[valid_shape],
        "detected_values": values[valid_shape, :2],
    }
    return (
        selected_sufficient,
        detected_sufficient,
        report,
        values[selected],
        populations,
    )


def _prediction_sufficient(
    bundle,
    frame: pd.DataFrame,
    *,
    detection_probability: np.ndarray,
    blend_shift: np.ndarray,
    draws: int,
    seed: int,
    batch_size: int,
    abs_shape_max: float | None,
    mag_max: float | None = MAG_MAX,
    radius_min_arcsec: float | None = RADIUS_MIN_ARCSEC,
    truth_shape: np.ndarray | None = None,
    fixed_population_masks: dict[str, np.ndarray] | None = None,
    plot_seed: int | None = None,
    probability_factorization: str = "usable_then_cut",
) -> tuple[dict, np.ndarray | None]:
    physical_names=("measured_ngmix_g1","measured_ngmix_g2",
                    "measured_flux_radius","measured_flux_from_mag_auto")
    names=tuple(bundle.target_transform.target_names)
    if names not in (TARGET_NAMES,physical_names):
        raise ValueError("measurement checkpoint target order is not v3.2-like")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    samples = bundle.sample(frame, n_samples=draws, batch_size=batch_size, qmc=True)
    samples = np.asarray(samples, dtype=np.float64)
    physical_diagnostics = None
    if names == physical_names:
        if not np.isfinite(samples).all() or np.any(samples[...,2:]<0):
            raise ValueError("physical samples must be finite and nonnegative; no clipping or discarding")
        physical_diagnostics = {
            "zero_radius_flux_draws": (samples[..., 2:] == 0).sum(axis=(0, 1)).tolist(),
            "minima": samples.min(axis=(0, 1)).tolist(),
            "maxima": samples.max(axis=(0, 1)).tolist(),
            "draw_count": int(np.prod(samples.shape[:2])),
        }
        # Exact unit conversion for the existing MAG_AUTO/radius cuts and plots.
        # Physical samples are generated directly by the model's bijection.
        # Numerical softplus zeros map to mag=+inf or ln(radius)=-inf. Retain
        # their shape draws in uncut means; the corresponding finite cuts fail.
        # No output is floored, dropped from its parent, or resampled.
        with np.errstate(divide="ignore"):
            samples=np.stack((samples[...,0],samples[...,1],
                              30.-2.5*np.log10(samples[...,3]),np.log(samples[...,2])),axis=-1)
    if probability_factorization not in ("usable_then_cut", "direct_selected"):
        raise ValueError(f"unknown probability factorization: {probability_factorization}")
    if probability_factorization == "direct_selected" and (
        names != physical_names
        or getattr(getattr(bundle, "model", None).__class__, "__name__", None)
        != "ConditionalPhysicalDiskFlow"
    ):
        raise ValueError(
            "direct-selected conditioning requires the physical disk flow so "
            "zero-acceptance shape support can be bounded"
        )
    samples[..., :2] += blend_shift[:, None, :]
    bounds = []
    if mag_max is not None:
        bounds.append(("measured_mag_auto", None, mag_max))
    if radius_min_arcsec is not None:
        bounds.append(
            (
                "measured_log_flux_radius",
                np.log(radius_min_arcsec / CONDITIONS["pixel_size"]),
                None,
            )
        )
    cut = (
        None
        if abs_shape_max is None and not bounds
        else OutputCut(TARGET_NAMES, abs_shape=abs_shape_max, bounds=tuple(bounds))
    )
    passed = (
        np.ones(samples.shape[:2], dtype=bool)
        if cut is None
        else np.asarray(cut(samples), dtype=bool)
    )
    p_det = np.asarray(detection_probability, dtype=np.float64)
    if (
        p_det.shape != (len(frame),)
        or not np.isfinite(p_det).all()
        or np.any((p_det < 0.0) | (p_det > 1.0))
    ):
        raise ValueError("event probabilities must be finite and in [0,1]")
    pass_count = passed.sum(axis=1, dtype=np.float64)
    pass_shape_sum = (samples[..., :2] * passed[..., None]).sum(
        axis=1, dtype=np.float64
    )
    shape_sum = samples[..., :2].sum(axis=1, dtype=np.float64)
    scale = 1.0 / float(draws)

    conditional_support = None
    if probability_factorization == "direct_selected":
        supported = pass_count > 0.0
        conditional_shape_mean = np.zeros_like(pass_shape_sum)
        np.divide(
            pass_shape_sum,
            pass_count[:, None],
            out=conditional_shape_mean,
            where=supported[:, None],
        )
        complete_denominator = float(np.sum(p_det))
        if complete_denominator <= 0.0:
            raise ValueError("direct selected-event probability has zero total mass")
        complete_numerator = np.sum(
            p_det[:, None] * conditional_shape_mean,
            axis=0,
            dtype=np.float64,
        )
        missing = ~supported
        # The physical flow maps shape into the open unit disk before the
        # deterministic R_blend shift.  Imputing zero for a row with no passing
        # draw therefore has the following component-wise worst-case bound.
        component_bound = 1.0 + np.abs(blend_shift)
        missing_numerator_bound = np.sum(
            p_det[missing, None] * component_bound[missing],
            axis=0,
            dtype=np.float64,
        )
        conditional_support = {
            "zero_pass_row_count": int(missing.sum()),
            "total_row_count": int(len(frame)),
            "total_event_probability_mass": complete_denominator,
            "zero_pass_event_probability_mass": float(p_det[missing].sum()),
            "zero_pass_event_probability_fraction": float(
                p_det[missing].sum() / complete_denominator
            ),
            "missing_numerator_abs_bound": missing_numerator_bound,
            "leg_mean_abs_bound": missing_numerator_bound / complete_denominator,
            "imputation": "zero_with_physical_disk_worst_case_bound",
        }
    else:
        complete_denominator = float(np.sum(p_det * pass_count) * scale)
        complete_numerator = np.sum(
            p_det[:, None] * pass_shape_sum, axis=0, dtype=np.float64
        ) * scale
    detection_denominator = float(np.sum(p_det))
    detection_numerator = np.sum(
        p_det[:, None] * shape_sum, axis=0, dtype=np.float64
    ) * scale
    selection_denominator = float(np.sum(pass_count) * scale)
    selection_numerator = pass_shape_sum.sum(axis=0, dtype=np.float64) * scale
    unweighted_denominator = float(len(frame))
    unweighted_numerator = shape_sum.sum(axis=0, dtype=np.float64) * scale
    sufficient = {
        "complete": {
            "denominator": complete_denominator,
            "numerator": complete_numerator,
        },
        "detection_only": {
            "denominator": detection_denominator,
            "numerator": detection_numerator,
        },
        "selection_only": {
            "denominator": selection_denominator,
            "numerator": selection_numerator,
        },
        "unweighted": {
            "denominator": unweighted_denominator,
            "numerator": unweighted_numerator,
        },
        "mean_pass_probability": float(pass_count.mean() * scale),
        "mean_detection_probability": float(p_det.mean()),
        "mean_detected_and_selected_probability": float(
            p_det.mean()
            if probability_factorization == "direct_selected"
            else np.mean(p_det * pass_count * scale)
        ),
    }
    if conditional_support is not None:
        sufficient["direct_selected_conditional_support"] = conditional_support
    if physical_diagnostics is not None:
        sufficient["physical_sample_diagnostics"] = physical_diagnostics
    if truth_shape is not None:
        truth_shape = np.asarray(truth_shape, dtype=np.float64)
        if truth_shape.shape != (len(frame), 2) or not np.isfinite(truth_shape).all():
            raise ValueError("truth_shape must be a finite (n_object, 2) array")
        sufficient["truth_detection"] = {
            "denominator": detection_denominator,
            "numerator": np.sum(
                p_det[:, None] * truth_shape, axis=0, dtype=np.float64
            ),
        }
        selection_weight = (
            p_det
            if probability_factorization == "direct_selected"
            else p_det * pass_count * scale
        )
        sufficient["truth_complete"] = {
            "denominator": complete_denominator,
            "numerator": np.sum(
                selection_weight[:, None] * truth_shape,
                axis=0,
                dtype=np.float64,
            ),
        }
    if fixed_population_masks:
        for name, mask in fixed_population_masks.items():
            mask = np.asarray(mask, dtype=bool)
            if mask.shape != (len(frame),) or not mask.any():
                raise ValueError(f"invalid or empty fixed-population mask: {name}")
            if name == "matched_detected_selected":
                sufficient[name] = {
                    "denominator": float(pass_count[mask].sum() * scale),
                    "numerator": pass_shape_sum[mask].sum(axis=0, dtype=np.float64) * scale,
                }
            else:
                sufficient[name] = {
                    "denominator": float(mask.sum()),
                    "numerator": shape_sum[mask].sum(axis=0, dtype=np.float64) * scale,
                }
    plot_values = None
    if plot_seed is not None:
        if probability_factorization == "direct_selected":
            raise ValueError(
                "plot thinning is not implemented for conditional selected-flow draws"
            )
        # One randomly shifted Sobol point per atom is marginally a valid flow
        # draw.  Bernoulli thinning by p_det therefore produces an unweighted
        # sample from the complete detected-and-selected distribution without
        # inheriting any image-simulation membership.
        rng = np.random.default_rng(plot_seed)
        detected = rng.random(len(frame)) < p_det
        retained = detected & passed[:, 0]
        plot_values = samples[retained, 0, :].copy()
    return sufficient, plot_values


def _mean(item: dict) -> np.ndarray:
    return np.asarray(item["numerator"], dtype=np.float64) / float(item["denominator"])


def _combine(items: list[dict], branch: str) -> dict:
    return {
        "denominator": float(sum(item[branch]["denominator"] for item in items)),
        "numerator": sum(
            (np.asarray(item[branch]["numerator"], dtype=np.float64) for item in items),
            start=np.zeros(2, dtype=np.float64),
        ),
    }


def _response_summary(plus: dict, minus: dict, h: float) -> dict:
    plus_mean = _mean(plus)
    minus_mean = _mean(minus)
    response = (plus_mean - minus_mean) / (2.0 * h)
    return {
        "plus_mean": plus_mean.tolist(),
        "minus_mean": minus_mean.tolist(),
        "response_first_column": response.tolist(),
    }


def _measured_summary(plus: dict, minus: dict, h: float) -> dict:
    converted_plus = {
        "denominator": plus["count"],
        "numerator": plus["sum"],
    }
    converted_minus = {
        "denominator": minus["count"],
        "numerator": minus["sum"],
    }
    summary = _response_summary(converted_plus, converted_minus, h)
    summary["plus_count"] = int(plus["count"])
    summary["minus_count"] = int(minus["count"])
    return summary


def _serialize_sufficient(item):
    if isinstance(item, np.ndarray):
        return item.tolist()
    if isinstance(item, dict):
        return {key: _serialize_sufficient(value) for key, value in item.items()}
    if isinstance(item, (list, tuple)):
        return [_serialize_sufficient(value) for value in item]
    return item


def _raw_case_response_sufficient(
    *,
    case: int,
    measured_matched_selected: dict,
    measured_matched_detected: dict,
    measured_matched_detected_selected: dict,
    measured_complete: dict,
    measured_detected: dict,
    truth_actual_detected: dict,
    truth_actual_selected: dict,
    predicted: list[dict],
) -> tuple[dict, dict]:
    """Preserve every model-by-QMC prediction run for exact block aggregation."""
    measured = {
        branch: {str(direction): values[direction][case] for direction in (+1, -1)}
        for branch, values in (
            ("matched_selected", measured_matched_selected),
            ("matched_detected", measured_matched_detected),
            ("matched_detected_selected", measured_matched_detected_selected),
            ("complete", measured_complete),
            ("detection_only", measured_detected),
        )
    }
    predicted_all = [
        {str(direction): run[direction][case] for direction in (+1, -1)}
        for run in predicted
    ]
    matched = {
        "measured": {
            str(direction): measured_matched_detected_selected[direction][case]
            for direction in (+1, -1)
        },
        "predicted": [
            {
                str(direction): run[direction][case]["matched_detected_selected"]
                for direction in (+1, -1)
            }
            for run in predicted
        ],
    }
    return {
        "measured": measured,
        "truth_actual": {
            "detection": {
                str(direction): truth_actual_detected[direction][case]
                for direction in (+1, -1)
            },
            "detection_and_selection": {
                str(direction): truth_actual_selected[direction][case]
                for direction in (+1, -1)
            },
        },
        "predicted": predicted_all,
    }, matched


def _prediction_run_metadata(prediction_runs: list[tuple]) -> list[dict]:
    """Serialize the canonical (bundle, path, QMC-seed) run ordering."""
    return [
        {
            "path": str(path),
            "sha256": _sha256(path),
            "training_seed": bundle.metadata.get("seed"),
            "sampling_seed": int(sampling_seed),
        }
        for bundle, path, sampling_seed in prediction_runs
    ]


def _experiment_comparison(
    *,
    measured: dict,
    predicted: dict,
    bootstrap: dict,
    definition: str,
    model_components: tuple[str, ...],
) -> dict:
    """Build one compact measured-versus-predicted response comparison."""
    measured_response = np.asarray(
        measured["response_first_column"], dtype=np.float64
    )
    predicted_response = np.asarray(
        predicted["response_first_column"], dtype=np.float64
    )
    return {
        "definition": definition,
        "model_components": list(model_components),
        "measured": measured,
        "predicted": predicted,
        "predicted_minus_measured": {
            "response_first_column": (
                predicted_response - measured_response
            ).tolist(),
            "case_bootstrap_standard_error": bootstrap["standard_error"],
            "case_bootstrap_ci95": bootstrap["ci95"],
        },
    }


def _combine_measured(items: list[dict]) -> dict:
    return {
        "count": int(sum(item["count"] for item in items)),
        "sum": sum(
            (np.asarray(item["sum"], dtype=np.float64) for item in items),
            start=np.zeros(2, dtype=np.float64),
        ),
    }


def _restrict_measured_population(population: dict, common_ids: np.ndarray) -> dict:
    ids = np.asarray(population["ids"], dtype=np.int64)
    values = np.asarray(population["values"], dtype=np.float64)
    keep = np.isin(ids, common_ids, assume_unique=True)
    if not keep.any():
        raise RuntimeError("common measured population is empty")
    return {
        "count": int(keep.sum()),
        "sum": values[keep].sum(axis=0, dtype=np.float64),
    }


def _truth_shape_sufficient(
    active_ids: np.ndarray,
    population_ids: np.ndarray,
    truth_shape: np.ndarray,
) -> dict:
    """Sufficient statistics of truth shapes for an observed-ID population."""
    active_ids = np.asarray(active_ids, dtype=np.int64)
    population_ids = np.asarray(population_ids, dtype=np.int64)
    truth_shape = np.asarray(truth_shape, dtype=np.float64)
    if truth_shape.shape != (len(active_ids), 2):
        raise ValueError("truth_shape and active_ids are not aligned")
    if len(np.unique(population_ids)) != len(population_ids):
        raise ValueError("observed population contains duplicate truth identities")
    keep = np.isin(active_ids, population_ids, assume_unique=True)
    if int(keep.sum()) != len(population_ids):
        raise RuntimeError("observed population contains identities outside support")
    if not keep.any():
        raise RuntimeError("observed truth-shape population is empty")
    return {
        "count": int(keep.sum()),
        "sum": truth_shape[keep].sum(axis=0, dtype=np.float64),
    }


def _truth_shape_case_bootstrap(
    *,
    cases: tuple[int, ...],
    actual_detected: dict,
    actual_selected: dict,
    predicted: list[dict],
    h: float,
    replicates: int,
    seed: int,
    include_detection: bool = True,
) -> dict:
    """Case-cluster bootstrap for truth-shape population-response closure."""
    if replicates <= 0:
        raise ValueError("bootstrap replicates must be positive")
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(cases), size=(replicates, len(cases)))
    offsets = len(cases) * np.arange(replicates, dtype=np.int64)[:, None]
    weights = np.bincount(
        (sampled + offsets).ravel(), minlength=replicates * len(cases)
    ).reshape(replicates, len(cases))
    case_order = tuple(int(case) for case in cases)
    branch_map = {"detection_and_selection": (actual_selected, "truth_complete")}
    if include_detection:
        branch_map["detection"] = (actual_detected, "truth_detection")
    residual = {}
    for name, (actual, predicted_branch) in branch_map.items():
        actual_counts = np.asarray(
            [[actual[d][case]["count"] for case in case_order] for d in (+1, -1)],
            dtype=np.float64,
        )
        actual_sums = np.asarray(
            [[actual[d][case]["sum"] for case in case_order] for d in (+1, -1)],
            dtype=np.float64,
        )
        actual_denominators = np.einsum("bc,dc->bd", weights, actual_counts)
        actual_numerators = np.einsum("bc,dcj->bdj", weights, actual_sums)
        if (actual_denominators <= 0).any():
            raise ValueError("empty actual truth-shape leg in case bootstrap")
        actual_means = actual_numerators / actual_denominators[:, :, None]
        actual_response = (actual_means[:, 0] - actual_means[:, 1]) / (2.0 * h)

        model_denominators = np.asarray(
            [
                [
                    [
                        model[d][case][predicted_branch]["denominator"]
                        for case in case_order
                    ]
                    for d in (+1, -1)
                ]
                for model in predicted
            ],
            dtype=np.float64,
        )
        model_numerators = np.asarray(
            [
                [
                    [
                        model[d][case][predicted_branch]["numerator"]
                        for case in case_order
                    ]
                    for d in (+1, -1)
                ]
                for model in predicted
            ],
            dtype=np.float64,
        )
        pooled_denominators = np.einsum(
            "bc,mdc->bmd", weights, model_denominators
        )
        pooled_numerators = np.einsum(
            "bc,mdcj->bmdj", weights, model_numerators
        )
        if (pooled_denominators <= 0).any():
            raise ValueError("empty predicted truth-shape leg in case bootstrap")
        model_means = pooled_numerators / pooled_denominators[:, :, :, None]
        model_response = (
            (model_means[:, :, 0] - model_means[:, :, 1]) / (2.0 * h)
        ).mean(axis=1)
        residual[name] = model_response - actual_response

    def summarize(values):
        return {
            "standard_error": values.std(axis=0, ddof=1).tolist(),
            "ci95": np.quantile(values, [0.025, 0.975], axis=0).tolist(),
        }

    result = {
        "unit": "ConstGold case",
        "replicates": int(replicates),
        "seed": int(seed),
        "detection_and_selection_predicted_minus_actual": summarize(
            residual["detection_and_selection"]
        ),
    }
    if include_detection:
        result["detection_predicted_minus_actual"] = summarize(
            residual["detection"]
        )
    return result


def _case_bootstrap(
    *,
    cases: tuple[int, ...],
    measured_selected: dict,
    measured_detected: dict,
    measured_matched_selected: dict,
    measured_matched_detected: dict,
    predicted: list[dict],
    h: float,
    replicates: int,
    seed: int,
    component_index: int = 0,
    include_detection: bool = True,
) -> dict:
    if replicates <= 0:
        raise ValueError("bootstrap replicates must be positive")
    if component_index not in (0, 1):
        raise ValueError("component_index must be 0 or 1")
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(cases), size=(replicates, len(cases)))
    offsets = len(cases) * np.arange(replicates, dtype=np.int64)[:, None]
    weights = np.bincount(
        (sampled + offsets).ravel(), minlength=replicates * len(cases)
    ).reshape(replicates, len(cases))
    branches = ("matched_selected", "matched_detected", "complete")
    if include_detection:
        branches = (*branches[:-1], "detection_only", branches[-1])
    measured_response = {}
    predicted_response = {}
    residual = {}
    multiplicative = {}
    case_order = tuple(int(case) for case in cases)
    sources = [
        ("matched_selected", measured_matched_selected),
        ("matched_detected", measured_matched_detected),
        ("complete", measured_selected),
    ]
    if include_detection:
        sources.insert(2, ("detection_only", measured_detected))
    for branch, source in sources:
        counts = np.asarray(
            [[source[d][case]["count"] for case in case_order] for d in (+1, -1)],
            dtype=np.float64,
        )
        sums = np.asarray(
            [[source[d][case]["sum"] for case in case_order] for d in (+1, -1)],
            dtype=np.float64,
        )
        denominators = np.einsum("bc,dc->bd", weights, counts)
        numerators = np.einsum("bc,dcj->bdj", weights, sums)
        if (denominators <= 0).any():
            raise ValueError("empty measured leg in case bootstrap")
        means = numerators / denominators[:, :, None]
        measured_response[branch] = (means[:, 0] - means[:, 1]) / (2.0 * h)

        model_denominators = np.asarray(
            [
                [
                    [model[d][case][branch]["denominator"] for case in case_order]
                    for d in (+1, -1)
                ]
                for model in predicted
            ],
            dtype=np.float64,
        )
        model_numerators = np.asarray(
            [
                [
                    [model[d][case][branch]["numerator"] for case in case_order]
                    for d in (+1, -1)
                ]
                for model in predicted
            ],
            dtype=np.float64,
        )
        pooled_denominators = np.einsum(
            "bc,mdc->bmd", weights, model_denominators
        )
        pooled_numerators = np.einsum(
            "bc,mdcj->bmdj", weights, model_numerators
        )
        if (pooled_denominators <= 0).any():
            raise ValueError("empty predicted leg in case bootstrap")
        model_means = pooled_numerators / pooled_denominators[:, :, :, None]
        predicted_response[branch] = (
            (model_means[:, :, 0] - model_means[:, :, 1]) / (2.0 * h)
        ).mean(axis=1)
        residual[branch] = predicted_response[branch] - measured_response[branch]
        denominator = predicted_response[branch][:, component_index]
        if (denominator <= 0).any():
            raise ValueError("nonpositive diagonal model response in case bootstrap")
        multiplicative[branch] = 100.0 * (
            measured_response[branch][:, component_index] / denominator - 1.0
        )
    def summarize(values):
        return {
            "standard_error": values.std(axis=0, ddof=1).tolist(),
            "ci95": np.quantile(values, [0.025, 0.975], axis=0).tolist(),
        }

    def summarize_scalar(values):
        return {
            "standard_error": float(values.std(ddof=1)),
            "ci95": np.quantile(values, [0.025, 0.975]).tolist(),
        }

    result = {
        "unit": "ConstGold case",
        "replicates": int(replicates),
        "seed": int(seed),
        "matched_selected_predicted_minus_measured": summarize(
            residual["matched_selected"]
        ),
        "matched_detected_predicted_minus_measured": summarize(
            residual["matched_detected"]
        ),
        "complete_predicted_minus_measured": summarize(residual["complete"]),
        "multiplicative_bias_percent": {
            branch: summarize_scalar(multiplicative[branch]) for branch in branches
        },
    }
    if include_detection:
        selection_residual = (
            predicted_response["complete"]
            - predicted_response["detection_only"]
            - measured_response["complete"]
            + measured_response["detection_only"]
        )
        result["detection_only_predicted_minus_measured"] = summarize(
            residual["detection_only"]
        )
        result["selection_shift_predicted_minus_measured"] = summarize(
            selection_residual
        )
    return result


def _matched_detection_selection_comparison(
    *, cases, measured, predicted, h, replicates, seed, component_index=0
):
    """Pooled response and paired-case ratio bootstrap on a fixed detected parent.

    Each leg retains its OWN measurement selection and denominator. Model
    predictions use their own passing draws, without classifier weights.
    """
    branch = "matched_detected_selected"
    if not cases or not predicted or h <= 0 or replicates < 2:
        raise ValueError("nonempty cases/models, positive h and >=2 replicates required")
    if component_index not in (0, 1):
        raise ValueError("component_index must be 0 or 1")
    cases = tuple(cases)
    md = np.asarray([[measured[d][c]["count"] for c in cases] for d in (+1, -1)], float)
    mn = np.asarray([[measured[d][c]["sum"] for c in cases] for d in (+1, -1)], float)
    pd = np.asarray([[[m[d][c][branch]["denominator"] for c in cases] for d in (+1,-1)] for m in predicted], float)
    pn = np.asarray([[[m[d][c][branch]["numerator"] for c in cases] for d in (+1,-1)] for m in predicted], float)
    if not all(np.isfinite(v).all() for v in (md,mn,pd,pn)):
        raise ValueError("nonfinite matched-detection selected sufficient statistics")
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0,len(cases),size=(replicates,len(cases)))
    weights = np.stack([np.bincount(v,minlength=len(cases)) for v in sampled])
    weights = np.vstack((np.ones(len(cases)),weights))
    mden = np.einsum('bc,dc->bd',weights,md)
    pden = np.einsum('bc,mdc->bmd',weights,pd)
    if (mden<=0).any() or (pden<=0).any():
        raise ValueError("empty selected leg in pooled or bootstrap sample")
    mm = np.einsum('bc,dcj->bdj',weights,mn)/mden[:,:,None]
    pm = np.einsum('bc,mdcj->bmdj',weights,pn)/pden[:,:,:,None]
    mr = (mm[:,0]-mm[:,1])/(2*h)
    pr = ((pm[:,:,0]-pm[:,:,1])/(2*h)).mean(axis=1)
    if (pr[:, component_index] <= 0).any():
        raise ValueError("nonpositive diagonal model response in pooled or bootstrap sample")
    m = 100 * (
        mr[:, component_index] / pr[:, component_index] - 1
    )
    residual=pr-mr
    label = "R11" if component_index == 0 else "R22"
    result = {
        "definition": f"Fixed identities usable and detected in both legs; separate measured cuts per leg. Model uses its own cut draws on those identities, without detection weights. Diagnostic m=100*({label}_sim/{label}_model-1), not complete-population m.",
        "response_component": label,
        "R_sim": float(mr[0, component_index]),
        "R_model": float(pr[0, component_index]),
        "m_percent":float(m[0]),"m_case_se_percent":float(m[1:].std(ddof=1)),
        "m_case_ci95_percent":np.quantile(m[1:],[.025,.975]).tolist(),
        "measured_selected_counts":md.sum(axis=1).astype(int).tolist(),
        "predicted_selected_weights_by_model":pd.sum(axis=2).tolist(),
        "predicted_minus_measured":residual[0].tolist(),
        "residual_case_se":residual[1:].std(axis=0,ddof=1).tolist(),
        "case_count":len(cases),"bootstrap_replicates":replicates,"bootstrap_seed":seed,
    }
    if component_index == 0:
        result.update(R11_sim=result["R_sim"], R11_model=result["R_model"])
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--constgold-root", type=Path, required=True)
    parser.add_argument(
        "--crowd-lookup",
        type=Path,
        help=(
            "optional precomputed crowding table; when omitted, derive the three "
            "shear-invariant crowding inputs directly from each truth scene"
        ),
    )
    parser.add_argument(
        "--rblend-lookup", type=Path, action="append", required=True
    )
    parser.add_argument("--measurement-model", type=Path, action="append", required=True)
    parser.add_argument("--emulator-model", type=Path)
    parser.add_argument("--emulator-metadata", type=Path)
    parser.add_argument(
        "--detection-classifier",
        type=Path,
        action="append",
        help=(
            "explicit SBSI detection-classifier checkpoint; repeat for an "
            "equal-probability ensemble replacing the bundled classifier"
        ),
    )
    parser.add_argument(
        "--probability-factorization",
        choices=("usable_then_cut", "direct_selected"),
        default="usable_then_cut",
        help=(
            "usable_then_cut uses p(U|x) times the flow cut probability; "
            "direct_selected treats the classifier as p(S|x) for the exact "
            "selected event and uses the flow only for E[e|S,x]"
        ),
    )
    parser.add_argument(
        "--shear-component",
        choices=("g1", "g2"),
        default="g1",
        help="constant-shear axis represented by the +/-h catalogues",
    )
    parser.add_argument(
        "--detection-neighbour-selection",
        choices=("nearest", "impact"),
        default="nearest",
    )
    parser.add_argument("--detection-impact-exponent", type=float, default=2.0)
    parser.add_argument("--case", type=int, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--h", type=float, default=0.02)
    parser.add_argument("--draws", type=int, default=64)
    parser.add_argument(
        "--sampling-seed",
        type=int,
        action="append",
        help="QMC seed; repeat for independent equal-weight draw groups (default: 7301)",
    )
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-boot", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260902)
    parser.add_argument("--plot-sample-size", type=int, default=0)
    parser.add_argument("--plot-sample-output", type=Path)
    parser.add_argument("--plot-sample-seed", type=int, default=20260902)
    parser.add_argument(
        "--disable-shape-cut",
        action="store_true",
        help="omit the measured ellipticity-norm cut while retaining mag and size cuts",
    )
    parser.add_argument(
        "--disable-mag-cut",
        action="store_true",
        help="omit the measured MAG_AUTO cut while retaining shape and size cuts",
    )
    parser.add_argument(
        "--disable-radius-cut",
        action="store_true",
        help="omit the measured radius cut while retaining shape and magnitude cuts",
    )
    parser.add_argument(
        "--analysis-primary-re-min",
        type=float,
        help=(
            "optionally restrict reported primary objects to Re_input above this "
            "value while retaining the complete truth scene for neighbour context"
        ),
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    cases = tuple(dict.fromkeys(int(case) for case in args.case))
    sampling_seeds = tuple(dict.fromkeys(args.sampling_seed or (7301,)))
    if not cases or args.h <= 0 or args.draws <= 0 or args.batch_size <= 0:
        raise ValueError("invalid cases, h, draws, or batch size")
    if args.detection_classifier is None and (
        args.emulator_model is None or args.emulator_metadata is None
    ):
        raise ValueError(
            "the bundled detector requires --emulator-model and --emulator-metadata"
        )
    if args.probability_factorization == "direct_selected" and args.plot_sample_size:
        raise ValueError("direct-selected mode does not support plot-sample thinning")
    abs_shape_max = None if args.disable_shape_cut else ABS_SHAPE_MAX
    mag_max = None if args.disable_mag_cut else MAG_MAX
    radius_min_arcsec = (
        None if args.disable_radius_cut else RADIUS_MIN_ARCSEC
    )

    support_all = pd.concat(
        [
            pd.read_feather(path, columns=["case", "input_index", "R_blend"])
            for path in args.rblend_lookup
        ],
        ignore_index=True,
    )
    support_all = support_all.loc[support_all["case"].isin(cases)].copy()
    if support_all["R_blend"].isna().any():
        raise RuntimeError("finite-prior support has missing R_blend values")
    if args.crowd_lookup is not None:
        crowd_all = pd.read_feather(
            args.crowd_lookup,
            columns=["case", "input_index", *CROWD_COLUMNS],
        )
        crowd_all = crowd_all.loc[crowd_all["case"].isin(cases)].copy()
        support_all = support_all.merge(
            crowd_all,
            on=["case", "input_index"],
            how="left",
            validate="one_to_one",
            sort=False,
        )
        if support_all[list(CROWD_COLUMNS)].isna().any().any():
            raise RuntimeError("finite-prior support has missing crowding values")

    classifier_paths = tuple(args.detection_classifier or ())
    if classifier_paths:
        detector = (
            load_selection_model(classifier_paths[0], device=args.device)
            if len(classifier_paths) == 1
            else _load_selection_model_ensemble(classifier_paths, device=args.device)
        )
        detector_source = (
            "explicit_sbsi_classifier"
            if len(classifier_paths) == 1
            else "explicit_sbsi_equal_probability_ensemble"
        )
    else:
        detector = load_emulator(
            ModelPaths(
                flow_checkpoints=(args.measurement_model[0],),
                emulator_model=args.emulator_model,
                emulator_metadata=args.emulator_metadata,
            ),
            conditions=CONDITIONS,
            device=args.device,
        )
        detector_source = "blendemu_release_classifier"
    detection_features = _detector_feature_names(detector)
    models = [
        load_measurement_model(str(path), device=args.device)
        for path in args.measurement_model
    ]
    prediction_runs = [
        (bundle, path, seed)
        for bundle, path in zip(models, args.measurement_model)
        for seed in sampling_seeds
    ]
    needs_circularized = (
        CIRCULARIZED_FEATURE in detection_features
        or any(
            CIRCULARIZED_FEATURE in bundle.condition_preprocessor.feature_names
            for bundle in models
        )
    )

    per_case_raw = {}
    measured_sufficient = {+1: {}, -1: {}}
    measured_detected_sufficient = {+1: {}, -1: {}}
    measured_matched_selected_sufficient = {+1: {}, -1: {}}
    measured_matched_detected_sufficient = {+1: {}, -1: {}}
    measured_matched_detected_selected_sufficient = {+1: {}, -1: {}}
    truth_actual_detected_sufficient = {+1: {}, -1: {}}
    truth_actual_selected_sufficient = {+1: {}, -1: {}}
    predicted_sufficient = [{+1: {}, -1: {}} for _ in prediction_runs]
    measured_plot_candidates = []
    predicted_plot_candidates = []
    for case in cases:
        print(f"case {case}: building complete truth scene", flush=True)
        support = support_all.loc[support_all["case"] == case].copy()
        support = support.sort_values("input_index", kind="stable").reset_index(drop=True)
        if support.empty or support["input_index"].duplicated().any():
            raise RuntimeError(f"case {case} has empty or duplicate finite-prior support")
        plus_root = _raw_catalogue_root(args.constgold_root, case, +args.h)
        truth_path = plus_root / "input" / "gals_info_tile180.0_-0.5.feather"
        truth = _read_truth(truth_path)
        active = _active_rows(truth, support)
        if args.analysis_primary_re_min is not None:
            keep_primary = (
                truth.iloc[active]["Re"].to_numpy(float)
                > args.analysis_primary_re_min
            )
            support = support.loc[keep_primary].reset_index(drop=True)
            active = active[keep_primary]
        active_ids = support["input_index"].to_numpy(np.int64)
        if not np.array_equal(truth.iloc[active]["index"].to_numpy(), active_ids):
            raise RuntimeError("active truth/support alignment failed")
        prior = ScenePrior.from_catalogue(
            truth,
            guard_radius_arcsec=7.0,
            duplicate_policy="error",
        )
        if args.crowd_lookup is None:
            support = _with_truth_crowding(support, prior=prior, active=active)

        measured_reports = {}
        measured_populations = {}
        frames = {}
        detection_probabilities = {}
        shifts = {}
        truth_shapes = {}
        intrinsic_e1, intrinsic_e2 = ellipticity_from_axis_ratio_angle(
            truth.iloc[active]["axis_ratio"].to_numpy(float),
            truth.iloc[active]["position_angle"].to_numpy(float),
        )
        rblend = support["R_blend"].to_numpy(float)
        for direction, sign in ((+1, +args.h), (-1, -args.h)):
            (
                measured_sufficient[direction][case],
                measured_detected_sufficient[direction][case],
                measured_reports[direction],
                measured_plot_values,
                measured_populations[direction],
            ) = _measured_leg(
                args.constgold_root,
                case=case,
                sign=sign,
                active_ids=active_ids,
                abs_shape_max=abs_shape_max,
                mag_max=mag_max,
                radius_min_arcsec=radius_min_arcsec,
            )
            if direction == +1 and args.plot_sample_size:
                measured_plot_candidates.append(measured_plot_values)
            shear = (sign, 0.0) if args.shear_component == "g1" else (0.0, sign)
            sheared = prior.shear(*shear)
            detection_frame = sheared.detection_view(
                conditions=CONDITIONS,
                radius_arcsec=3.0,
                neighbour_selection=args.detection_neighbour_selection,
                impact_exponent=args.detection_impact_exponent,
                primary_indices=active,
            )
            if not np.array_equal(
                detection_frame["input_index"].to_numpy(), active_ids
            ):
                raise RuntimeError("detection view/support alignment failed")
            frame = sheared.object_view(
                neighbour_radius_arcsec=7.0,
                conditions=CONDITIONS,
                include_crowding=False,
                primary_indices=active,
            )
            if not np.array_equal(frame["input_index"].to_numpy(), active_ids):
                raise RuntimeError("flow view/support alignment failed")
            for name in CROWD_COLUMNS:
                frame[name] = support[name].to_numpy(float)
            if needs_circularized:
                frame = _with_circularized_condition(frame, truth.iloc[active])
            frame = _with_flow_conditions(frame, models, rblend=rblend)
            if "R_blend" in detection_features:
                detection_frame["R_blend"] = rblend
            for name in detection_features:
                if name not in detection_frame and name in frame:
                    detection_frame[name] = frame[name]
            missing_detection = sorted(set(detection_features) - set(detection_frame))
            if missing_detection:
                raise KeyError(
                    f"detector inputs cannot be built: {missing_detection}"
                )
            detection_probabilities[direction] = predict_detection_probability(
                detector, detection_frame
            )
            if _detector_feature_names(detector) != detection_features:
                raise RuntimeError("detection model feature identity changed during run")
            frames[direction] = frame
            sheared_e1, sheared_e2 = apply_shear_to_ellipticity(
                intrinsic_e1, intrinsic_e2, *shear
            )
            truth_shapes[direction] = np.column_stack((sheared_e1, sheared_e2))
            shifts[direction] = rblend[:, None] * np.column_stack(
                (sheared_e1 - intrinsic_e1, sheared_e2 - intrinsic_e2)
            )

            truth_actual_detected_sufficient[direction][case] = (
                _truth_shape_sufficient(
                    active_ids,
                    measured_populations[direction]["all_detected_ids"],
                    truth_shapes[direction],
                )
            )
            truth_actual_selected_sufficient[direction][case] = (
                _truth_shape_sufficient(
                    active_ids,
                    measured_populations[direction]["selected_ids"],
                    truth_shapes[direction],
                )
            )

        common_selected_ids = np.intersect1d(
            measured_populations[+1]["selected_ids"],
            measured_populations[-1]["selected_ids"],
            assume_unique=True,
        )
        common_detected_ids = np.intersect1d(
            measured_populations[+1]["detected_ids"],
            measured_populations[-1]["detected_ids"],
            assume_unique=True,
        )
        for direction in (+1, -1):
            measured_matched_detected_selected_sufficient[direction][case] = (
                _restrict_measured_population(
                    {"ids": measured_populations[direction]["selected_ids"],
                     "values": measured_populations[direction]["selected_values"]},
                    common_detected_ids,
                )
            )
            measured_matched_selected_sufficient[direction][case] = (
                _restrict_measured_population(
                    {
                        "ids": measured_populations[direction]["selected_ids"],
                        "values": measured_populations[direction]["selected_values"],
                    },
                    common_selected_ids,
                )
            )
            measured_matched_detected_sufficient[direction][case] = (
                _restrict_measured_population(
                    {
                        "ids": measured_populations[direction]["detected_ids"],
                        "values": measured_populations[direction]["detected_values"],
                    },
                    common_detected_ids,
                )
            )
        fixed_population_masks = {
            "matched_selected": np.isin(
                active_ids, common_selected_ids, assume_unique=True
            ),
            "matched_detected": np.isin(
                active_ids, common_detected_ids, assume_unique=True
            ),
            "matched_detected_selected": np.isin(
                active_ids, common_detected_ids, assume_unique=True
            ),
        }

        model_case = []
        for model_index, (bundle, _, sampling_seed) in enumerate(prediction_runs):
            print(
                f"case {case}: full-flow selection expectation "
                f"run {model_index + 1}/{len(prediction_runs)} seed={sampling_seed}",
                flush=True,
            )
            for direction in (+1, -1):
                sufficient, plot_values = _prediction_sufficient(
                    bundle,
                    frames[direction],
                    detection_probability=detection_probabilities[direction],
                    blend_shift=shifts[direction],
                    draws=args.draws,
                    seed=sampling_seed,
                    batch_size=args.batch_size,
                    abs_shape_max=abs_shape_max,
                    mag_max=mag_max,
                    radius_min_arcsec=radius_min_arcsec,
                    truth_shape=truth_shapes[direction],
                    fixed_population_masks=fixed_population_masks,
                    probability_factorization=args.probability_factorization,
                    plot_seed=(
                        args.plot_sample_seed
                        + 1009 * case
                        + 100_003 * model_index
                        if direction == +1 and args.plot_sample_size
                        else None
                    ),
                )
                predicted_sufficient[model_index][direction][case] = sufficient
                if plot_values is not None:
                    predicted_plot_candidates.append(plot_values)
            model_case.append(
                {
                    branch: _response_summary(
                        predicted_sufficient[model_index][+1][case][branch],
                        predicted_sufficient[model_index][-1][case][branch],
                        args.h,
                    )
                    for branch in (
                        "matched_selected",
                        "matched_detected",
                        "matched_detected_selected",
                        "complete",
                        "detection_only",
                        "selection_only",
                        "unweighted",
                        "truth_detection",
                        "truth_complete",
                    )
                }
            )

        ensemble_case = {}
        for branch in (
            "matched_selected",
            "matched_detected",
            "matched_detected_selected",
            "complete",
            "detection_only",
            "selection_only",
            "unweighted",
            "truth_detection",
            "truth_complete",
        ):
            responses = np.asarray(
                [record[branch]["response_first_column"] for record in model_case]
            )
            plus_means = np.asarray([record[branch]["plus_mean"] for record in model_case])
            minus_means = np.asarray([record[branch]["minus_mean"] for record in model_case])
            ensemble_case[branch] = {
                "plus_mean": plus_means.mean(axis=0).tolist(),
                "minus_mean": minus_means.mean(axis=0).tolist(),
                "response_first_column": responses.mean(axis=0).tolist(),
                **_prediction_run_diagnostics(
                    responses,
                    model_count=len(models),
                    sampling_seed_count=len(sampling_seeds),
                ),
            }
        complete_r = np.asarray(ensemble_case["complete"]["response_first_column"])
        detect_r = np.asarray(ensemble_case["detection_only"]["response_first_column"])
        unweighted_r = np.asarray(ensemble_case["unweighted"]["response_first_column"])
        ensemble_case["ordered_contributions"] = (
            {
                "selected_event_weighting_with_uncut_flow_shape": (
                    detect_r - unweighted_r
                ).tolist(),
                "conditional_selected_shape_correction": (
                    complete_r - detect_r
                ).tolist(),
                "note": "diagnostic algebra, not a causal decomposition",
            }
            if args.probability_factorization == "direct_selected"
            else {
                "detection_weighting": (detect_r - unweighted_r).tolist(),
                "measured_selection_after_detection": (complete_r - detect_r).tolist(),
            }
        )
        measured_complete = _measured_summary(
            measured_sufficient[+1][case], measured_sufficient[-1][case], args.h
        )
        measured_detected = _measured_summary(
            measured_detected_sufficient[+1][case],
            measured_detected_sufficient[-1][case],
            args.h,
        )
        measured_matched_selected = _measured_summary(
            measured_matched_selected_sufficient[+1][case],
            measured_matched_selected_sufficient[-1][case],
            args.h,
        )
        measured_matched_detected = _measured_summary(
            measured_matched_detected_sufficient[+1][case],
            measured_matched_detected_sufficient[-1][case],
            args.h,
        )
        measured_matched_detected_selected = _measured_summary(
            measured_matched_detected_selected_sufficient[+1][case],
            measured_matched_detected_selected_sufficient[-1][case], args.h,
        )
        truth_actual_detected = _measured_summary(
            truth_actual_detected_sufficient[+1][case],
            truth_actual_detected_sufficient[-1][case],
            args.h,
        )
        truth_actual_selected = _measured_summary(
            truth_actual_selected_sufficient[+1][case],
            truth_actual_selected_sufficient[-1][case],
            args.h,
        )
        response_sufficient, matched_separate_sufficient = (
            _raw_case_response_sufficient(
                case=case,
                measured_matched_selected=measured_matched_selected_sufficient,
                measured_matched_detected=measured_matched_detected_sufficient,
                measured_matched_detected_selected=(
                    measured_matched_detected_selected_sufficient
                ),
                measured_complete=measured_sufficient,
                measured_detected=measured_detected_sufficient,
                truth_actual_detected=truth_actual_detected_sufficient,
                truth_actual_selected=truth_actual_selected_sufficient,
                predicted=predicted_sufficient,
            )
        )
        per_case_raw[str(case)] = {
            "response_sufficient": response_sufficient,
            "matched_detection_separate_selection_sufficient": (
                matched_separate_sufficient
            ),
            "measured": {
                "matched_selected": measured_matched_selected,
                "matched_detected": measured_matched_detected,
                "matched_detected_selected": measured_matched_detected_selected,
                "complete": measured_complete,
                "detection_only": measured_detected,
                "measured_selection_after_detection": (
                    np.asarray(measured_complete["response_first_column"])
                    - np.asarray(measured_detected["response_first_column"])
                ).tolist(),
            },
            "measured_population": {
                "plus": measured_reports[+1],
                "minus": measured_reports[-1],
                "matched_selected_count": int(len(common_selected_ids)),
                "matched_detected_count": int(len(common_detected_ids)),
            },
            "predicted": ensemble_case,
            "truth_shape_population": {
                "actual_detection_and_selection": truth_actual_selected,
                "predicted_detection_and_selection": ensemble_case[
                    "truth_complete"
                ],
                **(
                    {
                        "actual_detection": truth_actual_detected,
                        "predicted_detection": ensemble_case["truth_detection"],
                    }
                    if args.probability_factorization == "usable_then_cut"
                    else {}
                ),
            },
            "predicted_population": {
                "classifier_probability_role": (
                    "exact_selected_event_probability_p_S"
                    if args.probability_factorization == "direct_selected"
                    else "usable_detection_probability_p_U"
                ),
                "legacy_detection_probability_field_note": (
                    "the mean_detection_probability* keys below contain p_S, "
                    "not p_U, in direct_selected mode"
                    if args.probability_factorization == "direct_selected"
                    else None
                ),
                "mean_classifier_probability": float(
                    np.mean(
                        [
                            detection_probabilities[+1].mean(),
                            detection_probabilities[-1].mean(),
                        ]
                    )
                ),
                "mean_detection_probability": float(
                    np.mean(
                        [
                            detection_probabilities[+1].mean(),
                            detection_probabilities[-1].mean(),
                        ]
                    )
                ),
                "mean_detection_probability_plus": float(
                    detection_probabilities[+1].mean()
                ),
                "mean_detection_probability_minus": float(
                    detection_probabilities[-1].mean()
                ),
                "detection_probability_min": float(
                    min(
                        detection_probabilities[+1].min(),
                        detection_probabilities[-1].min(),
                    )
                ),
                "detection_probability_max": float(
                    max(
                        detection_probabilities[+1].max(),
                        detection_probabilities[-1].max(),
                    )
                ),
                "detection_probability_plus_minus_max_abs_difference": float(
                    np.max(
                        np.abs(
                            detection_probabilities[+1]
                            - detection_probabilities[-1]
                        )
                    )
                ),
                "complete_selected_fraction_plus_by_prediction_run": [
                    predicted_sufficient[i][+1][case][
                        "mean_detected_and_selected_probability"
                    ]
                    for i in range(len(prediction_runs))
                ],
                "complete_selected_fraction_minus_by_prediction_run": [
                    predicted_sufficient[i][-1][case][
                        "mean_detected_and_selected_probability"
                    ]
                    for i in range(len(prediction_runs))
                ],
            },
        }
        del prior, detection_frame, detection_probabilities, frames, shifts, truth_shapes
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    measured_plus_combined = {
        "count": sum(measured_sufficient[+1][case]["count"] for case in cases),
        "sum": sum(
            (measured_sufficient[+1][case]["sum"] for case in cases),
            start=np.zeros(2),
        ),
    }
    measured_minus_combined = {
        "count": sum(measured_sufficient[-1][case]["count"] for case in cases),
        "sum": sum(
            (measured_sufficient[-1][case]["sum"] for case in cases),
            start=np.zeros(2),
        ),
    }
    measured_detected_plus_combined = {
        "count": sum(
            measured_detected_sufficient[+1][case]["count"] for case in cases
        ),
        "sum": sum(
            (measured_detected_sufficient[+1][case]["sum"] for case in cases),
            start=np.zeros(2),
        ),
    }
    measured_detected_minus_combined = {
        "count": sum(
            measured_detected_sufficient[-1][case]["count"] for case in cases
        ),
        "sum": sum(
            (measured_detected_sufficient[-1][case]["sum"] for case in cases),
            start=np.zeros(2),
        ),
    }
    measured_matched_selected_plus_combined = _combine_measured(
        [measured_matched_selected_sufficient[+1][case] for case in cases]
    )
    measured_matched_selected_minus_combined = _combine_measured(
        [measured_matched_selected_sufficient[-1][case] for case in cases]
    )
    measured_matched_detected_plus_combined = _combine_measured(
        [measured_matched_detected_sufficient[+1][case] for case in cases]
    )
    measured_matched_detected_minus_combined = _combine_measured(
        [measured_matched_detected_sufficient[-1][case] for case in cases]
    )
    truth_actual_detected_plus_combined = _combine_measured(
        [truth_actual_detected_sufficient[+1][case] for case in cases]
    )
    truth_actual_detected_minus_combined = _combine_measured(
        [truth_actual_detected_sufficient[-1][case] for case in cases]
    )
    truth_actual_selected_plus_combined = _combine_measured(
        [truth_actual_selected_sufficient[+1][case] for case in cases]
    )
    truth_actual_selected_minus_combined = _combine_measured(
        [truth_actual_selected_sufficient[-1][case] for case in cases]
    )
    reported_branches = (
        "matched_selected",
        "matched_detected",
        "matched_detected_selected",
        "complete",
        "detection_only",
        "selection_only",
        "unweighted",
        "truth_detection",
        "truth_complete",
    )
    pooled_models = []
    for model_index in range(len(prediction_runs)):
        pooled_models.append(
            {
                branch: _response_summary(
                    _combine(
                        [predicted_sufficient[model_index][+1][case] for case in cases],
                        branch,
                    ),
                    _combine(
                        [predicted_sufficient[model_index][-1][case] for case in cases],
                        branch,
                    ),
                    args.h,
                )
                for branch in reported_branches
            }
        )
    pooled_prediction = {}
    for branch in reported_branches:
        responses = np.asarray(
            [record[branch]["response_first_column"] for record in pooled_models]
        )
        pooled_prediction[branch] = {
            "plus_mean": np.mean(
                [record[branch]["plus_mean"] for record in pooled_models], axis=0
            ).tolist(),
            "minus_mean": np.mean(
                [record[branch]["minus_mean"] for record in pooled_models], axis=0
            ).tolist(),
            "response_first_column": responses.mean(axis=0).tolist(),
            **_prediction_run_diagnostics(
                responses,
                model_count=len(models),
                sampling_seed_count=len(sampling_seeds),
            ),
        }
    complete_r = np.asarray(pooled_prediction["complete"]["response_first_column"])
    detect_r = np.asarray(pooled_prediction["detection_only"]["response_first_column"])
    unweighted_r = np.asarray(pooled_prediction["unweighted"]["response_first_column"])
    pooled_prediction["ordered_contributions"] = (
        {
            "selected_event_weighting_with_uncut_flow_shape": (
                detect_r - unweighted_r
            ).tolist(),
            "conditional_selected_shape_correction": (
                complete_r - detect_r
            ).tolist(),
            "note": "diagnostic algebra, not a causal decomposition",
        }
        if args.probability_factorization == "direct_selected"
        else {
            "detection_weighting": (detect_r - unweighted_r).tolist(),
            "measured_selection_after_detection": (complete_r - detect_r).tolist(),
        }
    )

    measured_complete = _measured_summary(
        measured_plus_combined, measured_minus_combined, args.h
    )
    measured_detected = _measured_summary(
        measured_detected_plus_combined, measured_detected_minus_combined, args.h
    )
    measured_matched_selected = _measured_summary(
        measured_matched_selected_plus_combined,
        measured_matched_selected_minus_combined,
        args.h,
    )
    measured_matched_detected = _measured_summary(
        measured_matched_detected_plus_combined,
        measured_matched_detected_minus_combined,
        args.h,
    )
    measured_matched_detected_selected = _measured_summary(
        _combine_measured([measured_matched_detected_selected_sufficient[+1][case] for case in cases]),
        _combine_measured([measured_matched_detected_selected_sufficient[-1][case] for case in cases]), args.h,
    )
    diagonal_index = 0 if args.shear_component == "g1" else 1
    matched_detection_selection = _matched_detection_selection_comparison(
        cases=cases, measured=measured_matched_detected_selected_sufficient,
        predicted=predicted_sufficient, h=args.h, replicates=args.n_boot, seed=args.bootstrap_seed,
        component_index=0 if args.shear_component == "g1" else 1,
    )
    if not np.allclose(
        [matched_detection_selection['R_sim'], matched_detection_selection['R_model']],
        [measured_matched_detected_selected['response_first_column'][diagonal_index],
         pooled_prediction['matched_detected_selected']['response_first_column'][diagonal_index]],
        rtol=1e-12, atol=1e-12,
    ):
        raise RuntimeError('matched-detection selection pooled/bootstrap conventions disagree')
    truth_actual_detected = _measured_summary(
        truth_actual_detected_plus_combined,
        truth_actual_detected_minus_combined,
        args.h,
    )
    truth_actual_selected = _measured_summary(
        truth_actual_selected_plus_combined,
        truth_actual_selected_minus_combined,
        args.h,
    )
    bootstrap = _case_bootstrap(
        cases=cases,
        measured_selected=measured_sufficient,
        measured_detected=measured_detected_sufficient,
        measured_matched_selected=measured_matched_selected_sufficient,
        measured_matched_detected=measured_matched_detected_sufficient,
        predicted=predicted_sufficient,
        h=args.h,
        replicates=args.n_boot,
        seed=args.bootstrap_seed,
        component_index=diagonal_index,
        include_detection=args.probability_factorization == "usable_then_cut",
    )
    truth_shape_bootstrap = _truth_shape_case_bootstrap(
        cases=cases,
        actual_detected=truth_actual_detected_sufficient,
        actual_selected=truth_actual_selected_sufficient,
        predicted=predicted_sufficient,
        h=args.h,
        replicates=args.n_boot,
        seed=args.bootstrap_seed,
        include_detection=args.probability_factorization == "usable_then_cut",
    )
    measured_diagonal = float(
        measured_complete["response_first_column"][diagonal_index]
    )
    predicted_diagonal = float(
        pooled_prediction["complete"]["response_first_column"][diagonal_index]
    )
    if predicted_diagonal <= 0:
        raise ValueError("nonpositive complete diagonal model response")
    complete_m = 100.0 * (measured_diagonal / predicted_diagonal - 1.0)
    complete_m_bootstrap = bootstrap["multiplicative_bias_percent"]["complete"]
    direct_support = None
    support_m_bound = 0.0
    if args.probability_factorization == "direct_selected":
        direct_support = _direct_selected_support_summary(
            predicted_sufficient,
            cases=cases,
            h=args.h,
            component_index=diagonal_index,
        )
        response_bound = direct_support[
            "maximum_case_ensemble_response_abs_bound"
        ]
        if predicted_diagonal <= response_bound:
            raise ValueError("direct-selected support bound reaches nonpositive response")
        support_m_bound = max(
            abs(
                100.0 * (measured_diagonal / value - 1.0) - complete_m
            )
            for value in (
                predicted_diagonal - response_bound,
                predicted_diagonal + response_bound,
            )
        )
        direct_support["maximum_case_m_abs_bound_percent"] = support_m_bound
    qmc_responses = np.asarray(
        pooled_prediction["complete"]["response_by_sampling_seed"],
        dtype=np.float64,
    )[:, diagonal_index]
    if np.any(qmc_responses <= 0.0):
        raise ValueError("nonpositive complete QMC-seed response")
    qmc_m_values = 100.0 * (measured_diagonal / qmc_responses - 1.0)
    qmc_m_se = (
        float(qmc_m_values.std(ddof=1) / np.sqrt(len(qmc_m_values)))
        if len(qmc_m_values) > 1
        else None
    )
    qmc_confidence = 0.95
    qmc_multiplier = _qmc_confidence_multiplier(
        len(qmc_m_values), confidence=qmc_confidence
    )
    qmc_half_width = (
        None if qmc_m_se is None else qmc_multiplier * qmc_m_se
    )
    numerical_margin = support_m_bound + (
        0.0 if qmc_half_width is None else qmc_half_width
    )
    expanded_ci = [
        float(complete_m_bootstrap["ci95"][0] - numerical_margin),
        float(complete_m_bootstrap["ci95"][1] + numerical_margin),
    ]
    response_acceptance = {
        "response_component": "R11" if diagonal_index == 0 else "R22",
        "measured_response": measured_diagonal,
        "model_response": predicted_diagonal,
        "m_percent": complete_m,
        "case_bootstrap_standard_error_percent": complete_m_bootstrap[
            "standard_error"
        ],
        "case_bootstrap_ci95_percent": complete_m_bootstrap["ci95"],
        "qmc_seed_standard_error_percent": qmc_m_se,
        "qmc_seed_confidence_level": qmc_confidence,
        "qmc_seed_degrees_of_freedom": (
            None if qmc_multiplier is None else len(qmc_m_values) - 1
        ),
        "qmc_seed_student_t_multiplier": qmc_multiplier,
        "qmc_seed_ci95_half_width_percent": qmc_half_width,
        "zero_acceptance_support_bound_percent": support_m_bound,
        "case_ci95_expanded_for_qmc_and_support_percent": expanded_ci,
        "equivalence_band_percent": [-0.3, 0.3],
        "passes_marginal_ci95_equivalence": bool(
            complete_m_bootstrap["ci95"][0] > -0.3
            and complete_m_bootstrap["ci95"][1] < 0.3
        ),
        "passes_expanded_ci95_equivalence": bool(
            expanded_ci[0] > -0.3 and expanded_ci[1] < 0.3
        ),
    }
    experiments = {
        "1_leg_matched_selected_shape": _experiment_comparison(
            measured=measured_matched_selected,
            predicted=pooled_prediction["matched_selected"],
            bootstrap=bootstrap[
                "matched_selected_predicted_minus_measured"
            ],
            definition=(
                "measured ellipticity response on the common identities that "
                "are detected, usable, and pass every measured cut in both "
                "legs; prediction uses the same fixed identities, so detection "
                "and moving-boundary selection responses are conditioned away"
            ),
            model_components=("measurement_flow", "fixed_r_blend_shift"),
        ),
        "3_sheared_intrinsic_detection_and_selection": _experiment_comparison(
            measured=truth_actual_selected,
            predicted=pooled_prediction["truth_complete"],
            bootstrap=truth_shape_bootstrap[
                "detection_and_selection_predicted_minus_actual"
            ],
            definition=(
                "per-leg mean analytically sheared intrinsic ellipticity over "
                "actual usable detections passing the magnitude and 0.75arcsec "
                "radius cuts versus the full parent weighted directly by p_S"
                if args.probability_factorization == "direct_selected"
                else "per-leg mean analytically sheared intrinsic ellipticity over "
                "actual detected identities passing every measured cut versus "
                "the full parent population weighted by classifier probability "
                "times the flow-estimated cut-pass probability"
            ),
            model_components=(
                ("direct_selected_event_classifier",)
                if args.probability_factorization == "direct_selected"
                else (
                    "transition_aware_classifier",
                    "measurement_flow_selection_probability",
                    "fixed_r_blend_shift_for_shape_cut",
                )
            ),
        ),
    }
    if args.probability_factorization == "usable_then_cut":
        experiments["2_sheared_intrinsic_detection"] = _experiment_comparison(
            measured=truth_actual_detected,
            predicted=pooled_prediction["truth_detection"],
            bootstrap=truth_shape_bootstrap[
                "detection_predicted_minus_actual"
            ],
            definition=(
                "per-leg mean analytically sheared intrinsic ellipticity over "
                "the actual detected identities versus the full parent "
                "population weighted by the independently evaluated classifier"
            ),
            model_components=("transition_aware_classifier",),
        )
    plot_sample_manifest = None
    if args.plot_sample_size:
        if args.plot_sample_output is None:
            raise ValueError("--plot-sample-size requires --plot-sample-output")
        measured_candidates = np.concatenate(measured_plot_candidates, axis=0)
        predicted_candidates = np.concatenate(predicted_plot_candidates, axis=0)
        if min(len(measured_candidates), len(predicted_candidates)) < args.plot_sample_size:
            raise RuntimeError("too few complete-likelihood contour candidates")
        rng = np.random.default_rng(args.plot_sample_seed)
        measured_index = rng.choice(
            len(measured_candidates), size=args.plot_sample_size, replace=False
        )
        predicted_index = rng.choice(
            len(predicted_candidates), size=args.plot_sample_size, replace=False
        )
        args.plot_sample_output.mkdir(parents=True, exist_ok=False)
        measured_path = args.plot_sample_output / "measured_plus_selected_100k.parquet"
        predicted_path = args.plot_sample_output / "predicted_plus_selected_100k.parquet"
        pd.DataFrame(
            measured_candidates[measured_index], columns=TARGET_NAMES
        ).to_parquet(measured_path, index=False)
        pd.DataFrame(
            predicted_candidates[predicted_index], columns=TARGET_NAMES
        ).to_parquet(predicted_path, index=False)
        plot_sample_manifest = {
            "injected_g1": float(args.h) if args.shear_component == "g1" else 0.0,
            "injected_g2": float(args.h) if args.shear_component == "g2" else 0.0,
            "sample_size_each": int(args.plot_sample_size),
            "seed": int(args.plot_sample_seed),
            "predicted_candidate_count": int(len(predicted_candidates)),
            "measured_candidate_count": int(len(measured_candidates)),
            "prediction_sampling": (
                "one marginal QMC flow draw per atom and flow checkpoint, "
                "Bernoulli-thinned by p_det, then measured-output cut"
            ),
            "predicted": str(predicted_path),
            "measured": str(measured_path),
            "output_sha256": {
                "measurements.parquet": _sha256(predicted_path),
                "measured_comparator.parquet": _sha256(measured_path),
            },
        }
        (args.plot_sample_output / "manifest.json").write_text(
            json.dumps(plot_sample_manifest, indent=2) + "\n"
        )

    result = {
        "analysis": "ConstGold complete forward likelihood response",
        "matrix_convention": (
            "response_first_column is [R11,R21]"
            if args.shear_component == "g1"
            else "legacy response_first_column field contains [R12,R22]"
        ),
        "shear_component": args.shear_component,
        "probability_factorization": args.probability_factorization,
        "response_acceptance": response_acceptance,
        "direct_selected_conditional_support": direct_support,
        "cases": list(cases),
        "h": float(args.h),
        "analysis_primary_re_min": args.analysis_primary_re_min,
        "circularized_condition": (
            "a0*sqrt(q0) from aligned pre-shear truth, held fixed in both legs"
            if needs_circularized else None
        ),
        "measured_output_cut": {
            "ellipticity_norm_max": abs_shape_max,
            "measured_mag_auto_max": mag_max,
            "measured_radius_arcsec_min": radius_min_arcsec,
            "measured_log_flux_radius_min": (
                float(np.log(radius_min_arcsec / CONDITIONS["pixel_size"]))
                if radius_min_arcsec is not None
                else None
            ),
        },
        "prediction_definition": (
            "sum_z p_S(z) E_flow[e|U,C,z,g] / sum_z p_S(z), with "
            "S=U&C and C=(MAG_AUTO<25.8, FLUX_RADIUS>=0.75arcsec); "
            "full finite-prior truth population"
            if args.probability_factorization == "direct_selected"
            else "sum_z p_det(z) E[e 1_C|z,g,det] / "
            "sum_z p_det(z) P(C|z,g,det); full finite-prior truth population"
        ),
        "response_ladder_definition": {
            "matched_selected": (
                "same identities detected and passing all measured cuts in both legs; "
                "model conditions on those identities, so only conditional shape response remains"
            ),
            "matched_detected_control": (
                "same usable detected identities in both legs with no measured-output cut; "
                "the literal both-detected construction also conditions away the classifier"
            ),
            "matched_detected_selected": (
                "same usable detected identities in both legs, then separate measured selection per leg; "
                "model applies its own draw-level selection on those identities without classifier weights"
            ),
            "detection_only": (
                "not an estimand in direct_selected mode; retained internal branch uses "
                "p_S-weighted uncut flow shape and is excluded from comparisons"
                if args.probability_factorization == "direct_selected"
                else "independent measured detected populations in each leg versus full-parent "
                "p_det-weighted prediction; conditional shape plus classifier, no measured cut"
            ),
            "complete": (
                "independent measured usable-and-selected populations in each leg versus "
                "full-parent direct p_S weighting of the flow conditional selected-shape mean"
                if args.probability_factorization == "direct_selected"
                else "independent measured detected-and-selected populations in each leg versus "
                "full-parent p_det times flow pass-probability prediction"
            ),
        },
        "detection_note": (
            "the direct selected-event ensemble is evaluated independently on the "
            "+/-h sheared scene views; its probability already includes usable "
            "measurement plus the magnitude and convolved-radius cuts"
            if args.probability_factorization == "direct_selected"
            else "the detector is evaluated independently on the +/-h sheared scene "
            "views, so shape-sensitive transition probabilities contribute to "
            "the finite-difference response"
            if classifier_paths
            else "the BlendEMU detection classifier is evaluated independently "
            "on the +/-h views; its spin-0 features make the probabilities equal"
        ),
        "truth_shape_population_definition": {
            "shape": (
                "true intrinsic ellipticity after applying the corresponding +/-h "
                "reduced shear; no measured shape or conditional-flow shape mean"
            ),
            "actual_detection": (
                "unweighted mean over every cross-matched detection on the finite-prior "
                "support; measured-shape validity is not required"
            ),
            "predicted_detection": (
                "not reported in direct_selected mode because p_S is not p_det"
                if args.probability_factorization == "direct_selected"
                else "full-parent truth-shape mean weighted by classifier p_det"
            ),
            "actual_detection_and_selection": (
                "unweighted truth-shape mean for actual detections passing all measured cuts"
            ),
            "predicted_detection_and_selection": (
                "full-parent truth-shape mean weighted directly by p_S"
                if args.probability_factorization == "direct_selected"
                else "full-parent truth-shape mean weighted by p_det times the flow-estimated "
                "measured-cut pass probability"
            ),
        },
        "experiments": experiments,
        "matched_detection_separate_selection": matched_detection_selection,
        "pooled": {
            "measured": {
                "matched_selected": measured_matched_selected,
                "matched_detected": measured_matched_detected,
                "matched_detected_selected": measured_matched_detected_selected,
                "complete": measured_complete,
                "detection_only": measured_detected,
                "measured_selection_after_detection": (
                    np.asarray(measured_complete["response_first_column"])
                    - np.asarray(measured_detected["response_first_column"])
                ).tolist(),
            },
            "predicted": pooled_prediction,
            "truth_shape_population": {
                "actual_detection_and_selection": truth_actual_selected,
                "predicted_detection_and_selection": pooled_prediction[
                    "truth_complete"
                ],
                **(
                    {
                        "actual_detection": truth_actual_detected,
                        "predicted_detection": pooled_prediction["truth_detection"],
                    }
                    if args.probability_factorization == "usable_then_cut"
                    else {}
                ),
            },
        },
        "per_case": per_case_raw,
        "case_bootstrap": bootstrap,
        "truth_shape_case_bootstrap": truth_shape_bootstrap,
        "models": _prediction_run_metadata(prediction_runs),
        "detection_model": (
            {
                "source": detector_source,
                "aggregation": (
                    "single_probability"
                    if len(classifier_paths) == 1
                    else "arithmetic_mean_probability"
                ),
                "classifiers": [str(path) for path in classifier_paths],
                "classifier_sha256": [
                    _sha256(path) for path in classifier_paths
                ],
                "metadata": detector.metadata,
                "features": list(detection_features),
                "radius_arcsec": 3.0,
                "neighbour_selection": args.detection_neighbour_selection,
                "impact_exponent": float(args.detection_impact_exponent),
            }
            if classifier_paths
            else {
                "source": detector_source,
                "emulator_model": str(args.emulator_model),
                "emulator_model_sha256": _sha256(args.emulator_model),
                "emulator_metadata": str(args.emulator_metadata),
                "emulator_metadata_sha256": _sha256(args.emulator_metadata),
                "classifier_path_from_release": (
                    "models/blendemu/classification_model_lsst_r_extnbr_ho.json"
                ),
                "features": list(detection_features),
                "radius_arcsec": 3.0,
                "neighbour_selection": args.detection_neighbour_selection,
                "impact_exponent": float(args.detection_impact_exponent),
            }
        ),
        "sampling": {
            "draws_per_atom_per_flow_seed": int(args.draws),
            "qmc": True,
            "common_random_numbers_between_legs": True,
            "seed": int(sampling_seeds[0]) if len(sampling_seeds) == 1 else None,
            "seeds": list(sampling_seeds),
        },
        "inputs": {
            "constgold_root": str(args.constgold_root),
            "crowd_lookup": (
                None if args.crowd_lookup is None else str(args.crowd_lookup)
            ),
            "crowding_source": (
                "truth_scene" if args.crowd_lookup is None else "precomputed_lookup"
            ),
            "rblend_lookup": [str(path) for path in args.rblend_lookup],
        },
        "plot_sample": plot_sample_manifest,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(_serialize_sufficient(result), indent=2) + "\n")
    print(json.dumps(result["pooled"], indent=2), flush=True)


if __name__ == "__main__":
    main()
