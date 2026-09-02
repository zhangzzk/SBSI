#!/usr/bin/env python
"""Forward ConstGold response with predicted detection and measured selection.

For every requested ConstGold case, the model starts from the complete
truth-level population on the registered finite-prior support.  At each of the
two applied shears it computes

    mean(e | detected and C)
      = sum_z p_det(z) E[e 1_C | z,g,det]
        / sum_z p_det(z) P(C | z,g,det),

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

from sbsi.catalogue_likelihood import OutputCut, predict_detection_probability
from sbsi.coordinates import ellipticity_from_axis_ratio_angle
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


def _flow_seed_sem(responses: np.ndarray) -> list[float] | None:
    """Return ensemble-seed SEM, or null when only one checkpoint is present."""
    if len(responses) < 2:
        return None
    return (responses.std(axis=0, ddof=1) / np.sqrt(len(responses))).tolist()


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


def _measured_leg(
    root: Path,
    *,
    case: int,
    sign: float,
    active_ids: np.ndarray,
    abs_shape_max: float | None,
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
    cut = OutputCut(
        TARGET_NAMES,
        abs_shape=abs_shape_max,
        bounds=(
            ("measured_mag_auto", None, MAG_MAX),
            (
                "measured_log_flux_radius",
                np.log(RADIUS_MIN_ARCSEC / CONDITIONS["pixel_size"]),
                None,
            ),
        ),
    )
    finite = np.isfinite(values).all(axis=1)
    valid_shape = finite & ~(
        (values[:, 0] == -1.0)
        | ((values[:, 0] == 0.0) & (values[:, 1] == 0.0))
    )
    selected = valid_shape & np.asarray(cut(values), dtype=bool)
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
            (valid_shape & ~np.asarray(cut(values), dtype=bool)).sum()
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
    truth_shape: np.ndarray | None = None,
    fixed_population_masks: dict[str, np.ndarray] | None = None,
    plot_seed: int | None = None,
) -> tuple[dict, np.ndarray | None]:
    if tuple(bundle.target_transform.target_names) != TARGET_NAMES:
        raise ValueError("measurement checkpoint target order is not v3.2-like")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    samples = bundle.sample(frame, n_samples=draws, batch_size=batch_size, qmc=True)
    samples = np.asarray(samples, dtype=np.float64)
    samples[..., :2] += blend_shift[:, None, :]
    cut = OutputCut(
        TARGET_NAMES,
        abs_shape=abs_shape_max,
        bounds=(
            ("measured_mag_auto", None, MAG_MAX),
            (
                "measured_log_flux_radius",
                np.log(RADIUS_MIN_ARCSEC / CONDITIONS["pixel_size"]),
                None,
            ),
        ),
    )
    passed = np.asarray(cut(samples), dtype=bool)
    p_det = np.asarray(detection_probability, dtype=np.float64)
    pass_count = passed.sum(axis=1, dtype=np.float64)
    pass_shape_sum = (samples[..., :2] * passed[..., None]).sum(
        axis=1, dtype=np.float64
    )
    shape_sum = samples[..., :2].sum(axis=1, dtype=np.float64)
    scale = 1.0 / float(draws)

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
            np.mean(p_det * pass_count * scale)
        ),
    }
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
        selection_weight = p_det * pass_count * scale
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
            sufficient[name] = {
                "denominator": float(mask.sum()),
                "numerator": shape_sum[mask].sum(axis=0, dtype=np.float64) * scale,
            }
    plot_values = None
    if plot_seed is not None:
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
    return item


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
) -> dict:
    """Case-cluster bootstrap for truth-shape population-response closure."""
    if replicates <= 0:
        raise ValueError("bootstrap replicates must be positive")
    rng = np.random.default_rng(seed)
    case_array = np.asarray(cases, dtype=np.int64)
    branch_map = {
        "detection": (actual_detected, "truth_detection"),
        "detection_and_selection": (actual_selected, "truth_complete"),
    }
    residual = {
        name: np.empty((replicates, 2), dtype=np.float64)
        for name in branch_map
    }
    for index in range(replicates):
        sampled = rng.choice(case_array, size=len(case_array), replace=True)
        for name, (actual, predicted_branch) in branch_map.items():
            actual_plus = _combine_measured(
                [actual[+1][int(case)] for case in sampled]
            )
            actual_minus = _combine_measured(
                [actual[-1][int(case)] for case in sampled]
            )
            actual_response = np.asarray(
                _measured_summary(actual_plus, actual_minus, h)[
                    "response_first_column"
                ]
            )
            model_responses = []
            for model in predicted:
                model_plus = _combine(
                    [model[+1][int(case)] for case in sampled], predicted_branch
                )
                model_minus = _combine(
                    [model[-1][int(case)] for case in sampled], predicted_branch
                )
                model_responses.append(
                    _response_summary(model_plus, model_minus, h)[
                        "response_first_column"
                    ]
                )
            residual[name][index] = (
                np.mean(model_responses, axis=0) - actual_response
            )

    def summarize(values):
        return {
            "standard_error": values.std(axis=0, ddof=1).tolist(),
            "ci95": np.quantile(values, [0.025, 0.975], axis=0).tolist(),
        }

    return {
        "unit": "ConstGold case",
        "replicates": int(replicates),
        "seed": int(seed),
        "detection_predicted_minus_actual": summarize(residual["detection"]),
        "detection_and_selection_predicted_minus_actual": summarize(
            residual["detection_and_selection"]
        ),
    }


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
) -> dict:
    if replicates <= 0:
        raise ValueError("bootstrap replicates must be positive")
    rng = np.random.default_rng(seed)
    branches = (
        "matched_selected",
        "matched_detected",
        "detection_only",
        "complete",
    )
    residual = {
        branch: np.empty((replicates, 2), dtype=np.float64) for branch in branches
    }
    selection_residual = np.empty((replicates, 2), dtype=np.float64)
    case_array = np.asarray(cases, dtype=np.int64)
    for index in range(replicates):
        sampled = rng.choice(case_array, size=len(case_array), replace=True)
        measured_response = {}
        for branch, source in (
            ("matched_selected", measured_matched_selected),
            ("matched_detected", measured_matched_detected),
            ("detection_only", measured_detected),
            ("complete", measured_selected),
        ):
            plus = _combine_measured([source[+1][int(case)] for case in sampled])
            minus = _combine_measured([source[-1][int(case)] for case in sampled])
            measured_response[branch] = np.asarray(
                _measured_summary(plus, minus, h)["response_first_column"]
            )
        predicted_response = {}
        for branch in branches:
            by_model = []
            for model in predicted:
                plus = _combine([model[+1][int(case)] for case in sampled], branch)
                minus = _combine([model[-1][int(case)] for case in sampled], branch)
                by_model.append(
                    _response_summary(plus, minus, h)["response_first_column"]
                )
            predicted_response[branch] = np.mean(by_model, axis=0)
            residual[branch][index] = (
                predicted_response[branch] - measured_response[branch]
            )
        selection_residual[index] = (
            predicted_response["complete"]
            - predicted_response["detection_only"]
            - measured_response["complete"]
            + measured_response["detection_only"]
        )

    def summarize(values):
        return {
            "standard_error": values.std(axis=0, ddof=1).tolist(),
            "ci95": np.quantile(values, [0.025, 0.975], axis=0).tolist(),
        }

    return {
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
        "detection_only_predicted_minus_measured": summarize(
            residual["detection_only"]
        ),
        "selection_shift_predicted_minus_measured": summarize(selection_residual),
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--constgold-root", type=Path, required=True)
    parser.add_argument("--crowd-lookup", type=Path, required=True)
    parser.add_argument(
        "--rblend-lookup", type=Path, action="append", required=True
    )
    parser.add_argument("--measurement-model", type=Path, action="append", required=True)
    parser.add_argument("--emulator-model", type=Path)
    parser.add_argument("--emulator-metadata", type=Path)
    parser.add_argument(
        "--detection-classifier",
        type=Path,
        help=(
            "explicit SBSI detection-classifier checkpoint; when supplied, "
            "it replaces the classifier bundled with the BlendEMU emulator"
        ),
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
    parser.add_argument("--sampling-seed", type=int, default=7301)
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
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    cases = tuple(dict.fromkeys(int(case) for case in args.case))
    if not cases or args.h <= 0 or args.draws <= 0 or args.batch_size <= 0:
        raise ValueError("invalid cases, h, draws, or batch size")
    if args.detection_classifier is None and (
        args.emulator_model is None or args.emulator_metadata is None
    ):
        raise ValueError(
            "the bundled detector requires --emulator-model and --emulator-metadata"
        )
    abs_shape_max = None if args.disable_shape_cut else ABS_SHAPE_MAX

    support_all = pd.concat(
        [
            pd.read_feather(path, columns=["case", "input_index", "R_blend"])
            for path in args.rblend_lookup
        ],
        ignore_index=True,
    )
    crowd_all = pd.read_feather(
        args.crowd_lookup,
        columns=["case", "input_index", *CROWD_COLUMNS],
    )
    support_all = support_all.loc[support_all["case"].isin(cases)].copy()
    crowd_all = crowd_all.loc[crowd_all["case"].isin(cases)].copy()
    support_all = support_all.merge(
        crowd_all,
        on=["case", "input_index"],
        how="left",
        validate="one_to_one",
        sort=False,
    )
    if support_all[["R_blend", *CROWD_COLUMNS]].isna().any().any():
        raise RuntimeError("finite-prior support has missing R_blend/crowding values")

    if args.detection_classifier is not None:
        detector = load_selection_model(args.detection_classifier, device=args.device)
        detector_source = "explicit_sbsi_classifier"
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

    per_case_raw = {}
    measured_sufficient = {+1: {}, -1: {}}
    measured_detected_sufficient = {+1: {}, -1: {}}
    measured_matched_selected_sufficient = {+1: {}, -1: {}}
    measured_matched_detected_sufficient = {+1: {}, -1: {}}
    truth_actual_detected_sufficient = {+1: {}, -1: {}}
    truth_actual_selected_sufficient = {+1: {}, -1: {}}
    predicted_sufficient = [
        {+1: {}, -1: {}} for _ in models
    ]
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
        active_ids = support["input_index"].to_numpy(np.int64)
        if not np.array_equal(truth.iloc[active]["index"].to_numpy(), active_ids):
            raise RuntimeError("active truth/support alignment failed")
        prior = ScenePrior.from_catalogue(
            truth,
            guard_radius_arcsec=7.0,
            duplicate_policy="error",
        )

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
            )
            if direction == +1 and args.plot_sample_size:
                measured_plot_candidates.append(measured_plot_values)
            sheared = prior.shear(sign, 0.0)
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
            detection_probabilities[direction] = predict_detection_probability(
                detector, detection_frame
            )
            if _detector_feature_names(detector) != detection_features:
                raise RuntimeError("detection model feature identity changed during run")
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
            frames[direction] = frame
            sheared_e1, sheared_e2 = apply_shear_to_ellipticity(
                intrinsic_e1, intrinsic_e2, sign, 0.0
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
        }

        model_case = []
        for model_index, bundle in enumerate(models):
            print(
                f"case {case}: full-flow selection expectation "
                f"model {model_index + 1}/{len(models)}",
                flush=True,
            )
            for direction in (+1, -1):
                sufficient, plot_values = _prediction_sufficient(
                    bundle,
                    frames[direction],
                    detection_probability=detection_probabilities[direction],
                    blend_shift=shifts[direction],
                    draws=args.draws,
                    seed=args.sampling_seed,
                    batch_size=args.batch_size,
                    abs_shape_max=abs_shape_max,
                    truth_shape=truth_shapes[direction],
                    fixed_population_masks=fixed_population_masks,
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
                "response_by_flow_seed": responses.tolist(),
                "flow_seed_sem": _flow_seed_sem(responses),
            }
        complete_r = np.asarray(ensemble_case["complete"]["response_first_column"])
        detect_r = np.asarray(ensemble_case["detection_only"]["response_first_column"])
        unweighted_r = np.asarray(ensemble_case["unweighted"]["response_first_column"])
        ensemble_case["ordered_contributions"] = {
            "detection_weighting": (detect_r - unweighted_r).tolist(),
            "measured_selection_after_detection": (complete_r - detect_r).tolist(),
        }
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
        per_case_raw[str(case)] = {
            "measured": {
                "matched_selected": measured_matched_selected,
                "matched_detected": measured_matched_detected,
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
                "actual_detection": truth_actual_detected,
                "predicted_detection": ensemble_case["truth_detection"],
                "actual_detection_and_selection": truth_actual_selected,
                "predicted_detection_and_selection": ensemble_case[
                    "truth_complete"
                ],
            },
            "predicted_population": {
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
                "complete_selected_fraction_plus_by_flow_seed": [
                    predicted_sufficient[i][+1][case][
                        "mean_detected_and_selected_probability"
                    ]
                    for i in range(len(models))
                ],
                "complete_selected_fraction_minus_by_flow_seed": [
                    predicted_sufficient[i][-1][case][
                        "mean_detected_and_selected_probability"
                    ]
                    for i in range(len(models))
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
        "complete",
        "detection_only",
        "selection_only",
        "unweighted",
        "truth_detection",
        "truth_complete",
    )
    pooled_models = []
    for model_index in range(len(models)):
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
            "response_by_flow_seed": responses.tolist(),
            "flow_seed_sem": _flow_seed_sem(responses),
        }
    complete_r = np.asarray(pooled_prediction["complete"]["response_first_column"])
    detect_r = np.asarray(pooled_prediction["detection_only"]["response_first_column"])
    unweighted_r = np.asarray(pooled_prediction["unweighted"]["response_first_column"])
    pooled_prediction["ordered_contributions"] = {
        "detection_weighting": (detect_r - unweighted_r).tolist(),
        "measured_selection_after_detection": (complete_r - detect_r).tolist(),
    }

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
    )
    truth_shape_bootstrap = _truth_shape_case_bootstrap(
        cases=cases,
        actual_detected=truth_actual_detected_sufficient,
        actual_selected=truth_actual_selected_sufficient,
        predicted=predicted_sufficient,
        h=args.h,
        replicates=args.n_boot,
        seed=args.bootstrap_seed,
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
            "injected_g1": float(args.h),
            "injected_g2": 0.0,
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
            "ConstGold excites g1 only; response_first_column is [R11,R21]"
        ),
        "cases": list(cases),
        "h": float(args.h),
        "measured_output_cut": {
            "ellipticity_norm_max": abs_shape_max,
            "measured_mag_auto_max": MAG_MAX,
            "measured_radius_arcsec_min": RADIUS_MIN_ARCSEC,
            "measured_log_flux_radius_min": float(
                np.log(RADIUS_MIN_ARCSEC / CONDITIONS["pixel_size"])
            ),
        },
        "prediction_definition": (
            "sum_z p_det(z) E[e 1_C|z,g,det] / "
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
            "detection_only": (
                "independent measured detected populations in each leg versus full-parent "
                "p_det-weighted prediction; conditional shape plus classifier, no measured cut"
            ),
            "complete": (
                "independent measured detected-and-selected populations in each leg versus "
                "full-parent p_det times flow pass-probability prediction"
            ),
        },
        "detection_note": (
            "the detector is evaluated independently on the +/-h sheared scene "
            "views, so shape-sensitive transition probabilities contribute to "
            "the finite-difference response"
            if args.detection_classifier is not None
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
                "full-parent truth-shape mean weighted by classifier p_det"
            ),
            "actual_detection_and_selection": (
                "unweighted truth-shape mean for actual detections passing all measured cuts"
            ),
            "predicted_detection_and_selection": (
                "full-parent truth-shape mean weighted by p_det times the flow-estimated "
                "measured-cut pass probability"
            ),
        },
        "pooled": {
            "measured": {
                "matched_selected": measured_matched_selected,
                "matched_detected": measured_matched_detected,
                "complete": measured_complete,
                "detection_only": measured_detected,
                "measured_selection_after_detection": (
                    np.asarray(measured_complete["response_first_column"])
                    - np.asarray(measured_detected["response_first_column"])
                ).tolist(),
            },
            "predicted": pooled_prediction,
            "truth_shape_population": {
                "actual_detection": truth_actual_detected,
                "predicted_detection": pooled_prediction["truth_detection"],
                "actual_detection_and_selection": truth_actual_selected,
                "predicted_detection_and_selection": pooled_prediction[
                    "truth_complete"
                ],
            },
        },
        "per_case": per_case_raw,
        "case_bootstrap": bootstrap,
        "truth_shape_case_bootstrap": truth_shape_bootstrap,
        "models": [
            {
                "path": str(path),
                "sha256": _sha256(path),
                "training_seed": bundle.metadata.get("seed"),
            }
            for path, bundle in zip(args.measurement_model, models)
        ],
        "detection_model": (
            {
                "source": detector_source,
                "classifier": str(args.detection_classifier),
                "classifier_sha256": _sha256(args.detection_classifier),
                "metadata": detector.metadata,
                "features": list(detection_features),
                "radius_arcsec": 3.0,
                "neighbour_selection": args.detection_neighbour_selection,
                "impact_exponent": float(args.detection_impact_exponent),
            }
            if args.detection_classifier is not None
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
            "seed": int(args.sampling_seed),
        },
        "inputs": {
            "constgold_root": str(args.constgold_root),
            "crowd_lookup": str(args.crowd_lookup),
            "rblend_lookup": [str(path) for path in args.rblend_lookup],
        },
        "plot_sample": plot_sample_manifest,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(_serialize_sufficient(result), indent=2) + "\n")
    print(json.dumps(result["pooled"], indent=2), flush=True)


if __name__ == "__main__":
    main()
