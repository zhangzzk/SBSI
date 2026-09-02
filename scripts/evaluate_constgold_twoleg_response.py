#!/usr/bin/env python
"""Compare matched and unmatched two-leg ConstGold responses.

The matched estimator uses the intersection of usable detections in the two
ConstGold signs.  The unmatched estimator differences the two independently
selected leg means.  Neither estimator applies a measured-output cut.  Both
use the registered true-property population/domain cuts.

The model prediction evaluates the measurement-flow conditional mean on the
same identities used by the corresponding simulation estimator and adds the
finite-shear external blend shift

    R_blend * (e(S_g z) - e(z)).

For the unmatched estimator, detection membership is inherited from the image
simulation; it is not predicted by the conditional measurement flow.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.ipc as ipc
import torch

from sbsi.coordinates import ellipticity_from_axis_ratio_angle
from sbsi.measurement_model import load_measurement_model
from sbsi.preprocessing import source_select_selection
from sbsi.shear_map import apply_shear_to_ellipticity
from sbsi.training import build_shifted_context


RESCALE_KWARGS = {
    "pixel_rms": 0.312,
    "pixel_size": 0.2,
    "zero_mag": 30.0,
    "psf_fwhm": 0.73,
    "moffat_beta": 2.224,
}
LEG_COLUMNS = (
    "case",
    "input_index",
    "neighbored",
    "distance",
    "Re_input_p",
    "Re_input_s",
    "r_input_p",
    "r_input_s",
    "sersic_n_input_p",
    "sersic_n_input_s",
    "axis_ratio_input_p",
    "axis_ratio_input_s",
    "position_angle_input_p",
    "position_angle_input_s",
    "polarization_angle",
    "applied_g1",
    "applied_g2",
    "measured_e1",
    "measured_e2",
)
CROWD_COLUMNS = ("nbr_flux_near", "nbr_flux_far", "nbr_flux_max")
POPULATION_CUTS = (
    (18.0, 28.0),
    (18.0, 25.8),
    (0.1, 1.5),
    (0.5, 1.5),
    (0.0, 5.0),
)


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _keys(case, input_index) -> np.ndarray:
    case = np.asarray(case, dtype=np.int64)
    index = np.asarray(input_index, dtype=np.int64)
    if (case < 0).any() or (index < 0).any() or (index >= 2**32).any():
        raise ValueError("case/input_index cannot be packed into nonnegative 32-bit keys")
    return (case << np.int64(32)) | index


def _read_case_window(path: Path, columns, min_case: int, max_case: int) -> pd.DataFrame:
    """Read a sorted Feather case window without materializing later cases."""

    required = tuple(dict.fromkeys(("case", *columns)))
    parts = []
    previous_max = None
    with pa.memory_map(str(path), "r") as stream:
        reader = ipc.open_file(stream)
        missing = sorted(set(required) - set(reader.schema.names))
        if missing:
            raise KeyError(f"{path} lacks required columns: {missing}")
        for batch_index in range(reader.num_record_batches):
            batch = reader.get_batch(batch_index).select(required)
            cases = batch["case"].to_numpy(zero_copy_only=False).astype(np.int64)
            if not len(cases):
                continue
            batch_min = int(cases.min())
            batch_max = int(cases.max())
            if previous_max is not None and batch_min < previous_max:
                raise RuntimeError(f"{path} is not ordered by case")
            previous_max = batch_max
            if batch_min >= max_case:
                break
            if batch_max < min_case:
                continue
            table = pa.Table.from_batches([batch])
            keep = pc.and_(
                pc.greater_equal(table["case"], pa.scalar(min_case)),
                pc.less(table["case"], pa.scalar(max_case)),
            )
            table = table.filter(keep)
            if table.num_rows:
                parts.append(table.to_pandas())
    if not parts:
        raise RuntimeError(f"{path} has no rows in cases [{min_case},{max_case})")
    return pd.concat(parts, ignore_index=True)


def _prepare_leg(
    path: Path,
    lookup: pd.DataFrame,
    *,
    min_case: int,
    max_case: int,
    expected_sign: float,
) -> tuple[pd.DataFrame, dict]:
    raw = _read_case_window(path, LEG_COLUMNS, min_case, max_case)
    if raw[["case", "input_index"]].duplicated().any():
        raise ValueError(f"{path} contains duplicate leg identities")
    before_population = len(raw)
    frame = source_select_selection(raw, cuts=POPULATION_CUTS)
    after_population = len(frame)
    finite_shape = np.isfinite(
        frame[["measured_e1", "measured_e2"]].to_numpy(dtype=float)
    ).all(axis=1)
    finite_shape &= ~(
        (frame["measured_e1"].to_numpy(float) == -1.0)
        | (
            (frame["measured_e1"].to_numpy(float) == 0.0)
            & (frame["measured_e2"].to_numpy(float) == 0.0)
        )
    )
    failed_shape = int((~finite_shape).sum())
    frame = frame.loc[finite_shape].copy()
    measured_sign = frame["applied_g1"].to_numpy(float)
    if not np.allclose(measured_sign, expected_sign, rtol=0.0, atol=1e-12):
        raise ValueError(f"{path} does not carry the expected g1={expected_sign:+g}")
    if not np.allclose(frame["applied_g2"], 0.0, rtol=0.0, atol=1e-12):
        raise ValueError(f"{path} is not a pure g1 ConstGold leg")

    frame = frame.merge(
        lookup,
        on=["case", "input_index"],
        how="left",
        validate="one_to_one",
        indicator=True,
        sort=False,
    )
    unmatched_lookup = int((frame["_merge"] != "both").sum())
    frame = frame.loc[frame["_merge"] == "both"].drop(columns="_merge")
    finite_lookup = np.isfinite(
        frame[["R_blend", *CROWD_COLUMNS]].to_numpy(dtype=float)
    ).all(axis=1)
    nonfinite_lookup = int((~finite_lookup).sum())
    frame = frame.loc[finite_lookup].copy()
    frame["_key"] = _keys(frame["case"], frame["input_index"])
    frame = frame.sort_values("_key", kind="stable").reset_index(drop=True)
    if frame["_key"].duplicated().any():
        raise ValueError("prepared leg contains duplicate identities")
    return frame, {
        "rows_before_population_cut": int(before_population),
        "rows_after_population_cut": int(after_population),
        "dropped_failed_or_nonfinite_shape": failed_shape,
        "dropped_unmatched_lookup": unmatched_lookup,
        "dropped_nonfinite_lookup": nonfinite_lookup,
        "usable_rows": int(len(frame)),
    }


def _paired_predictions(
    bundle,
    frame: pd.DataFrame,
    *,
    h: float,
    draws: int,
    sampling_seed: int,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    names = tuple(bundle.target_transform.target_names)
    expected = (
        "measured_ngmix_g1",
        "measured_ngmix_g2",
        "measured_mag_auto",
        "measured_log_flux_radius",
    )
    if names != expected:
        raise ValueError(f"unexpected measurement target order: {names}")
    work = frame.copy()
    intrinsic_e1, intrinsic_e2 = ellipticity_from_axis_ratio_angle(
        work["axis_ratio_input_p"].to_numpy(float),
        work["position_angle_input_p"].to_numpy(float),
    )
    work["e1_input_rot0_p"] = intrinsic_e1
    work["e2_input_rot0_p"] = intrinsic_e2
    work["gamma1_input_p"] = 0.0
    work["gamma2_input_p"] = 0.0
    plus_context = build_shifted_context(
        work,
        (1.0, 0.0),
        +h,
        bundle.condition_preprocessor,
        bundle.metadata["condition_features"],
        RESCALE_KWARGS,
    )
    minus_context = build_shifted_context(
        work,
        (1.0, 0.0),
        -h,
        bundle.condition_preprocessor,
        bundle.metadata["condition_features"],
        RESCALE_KWARGS,
    )
    plus_result = np.empty((len(frame), 2), dtype=np.float64)
    minus_result = np.empty((len(frame), 2), dtype=np.float64)
    means = torch.as_tensor(
        bundle.target_transform.means, dtype=torch.float32, device=bundle.device
    )
    scales = torch.as_tensor(
        bundle.target_transform.scales, dtype=torch.float32, device=bundle.device
    )
    bundle.model.eval()
    with torch.no_grad():
        for start in range(0, len(frame), batch_size):
            stop = min(start + batch_size, len(frame))
            paired_seed = int(sampling_seed) * 1_000_003 + start
            torch.manual_seed(paired_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(paired_seed)
            plus_tensor = torch.as_tensor(
                plus_context[start:stop], dtype=torch.float32, device=bundle.device
            )
            plus_standardized = bundle.model.sample(
                plus_tensor, n_samples=draws, qmc=True
            ).mean(dim=1)
            torch.manual_seed(paired_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(paired_seed)
            minus_tensor = torch.as_tensor(
                minus_context[start:stop], dtype=torch.float32, device=bundle.device
            )
            minus_standardized = bundle.model.sample(
                minus_tensor, n_samples=draws, qmc=True
            ).mean(dim=1)
            plus_raw = plus_standardized * scales + means
            minus_raw = minus_standardized * scales + means
            plus_result[start:stop] = plus_raw[:, :2].double().cpu().numpy()
            minus_result[start:stop] = minus_raw[:, :2].double().cpu().numpy()

    plus_e1, plus_e2 = apply_shear_to_ellipticity(
        intrinsic_e1, intrinsic_e2, +h, 0.0
    )
    minus_e1, minus_e2 = apply_shear_to_ellipticity(
        intrinsic_e1, intrinsic_e2, -h, 0.0
    )
    rblend = frame["R_blend"].to_numpy(float)[:, None]
    plus_result += rblend * np.column_stack(
        (plus_e1 - intrinsic_e1, plus_e2 - intrinsic_e2)
    )
    minus_result += rblend * np.column_stack(
        (minus_e1 - intrinsic_e1, minus_e2 - intrinsic_e2)
    )
    return plus_result, minus_result


def _sufficient(frame: pd.DataFrame, values: np.ndarray, cases: np.ndarray) -> dict:
    output = {}
    frame_case = frame["case"].to_numpy(np.int64)
    for case in cases:
        selected = frame_case == case
        output[int(case)] = {
            "count": int(selected.sum()),
            "sum": values[selected].sum(axis=0, dtype=np.float64),
        }
    return output


def _response(plus: dict, minus: dict, cases: np.ndarray, h: float) -> np.ndarray:
    plus_count = sum(plus[int(case)]["count"] for case in cases)
    minus_count = sum(minus[int(case)]["count"] for case in cases)
    if plus_count <= 0 or minus_count <= 0:
        raise RuntimeError("one response leg is empty")
    plus_mean = sum(plus[int(case)]["sum"] for case in cases) / plus_count
    minus_mean = sum(minus[int(case)]["sum"] for case in cases) / minus_count
    return (plus_mean - minus_mean) / (2.0 * h)


def _means(plus: dict, minus: dict, cases: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    plus_count = sum(plus[int(case)]["count"] for case in cases)
    minus_count = sum(minus[int(case)]["count"] for case in cases)
    return (
        sum(plus[int(case)]["sum"] for case in cases) / plus_count,
        sum(minus[int(case)]["sum"] for case in cases) / minus_count,
    )


def _bootstrap_summary(
    measured_plus: dict,
    measured_minus: dict,
    predicted_plus: list[dict],
    predicted_minus: list[dict],
    *,
    cases: np.ndarray,
    h: float,
    n_boot: int,
    seed: int,
) -> dict:
    measured = _response(measured_plus, measured_minus, cases, h)
    predicted_by_seed = np.asarray(
        [
            _response(plus, minus, cases, h)
            for plus, minus in zip(predicted_plus, predicted_minus)
        ]
    )
    predicted = predicted_by_seed.mean(axis=0)
    difference = predicted - measured
    m = measured[0] / predicted[0] - 1.0
    rng = np.random.default_rng(seed)
    boot_measured = np.empty((n_boot, 2), dtype=np.float64)
    boot_predicted = np.empty((n_boot, 2), dtype=np.float64)
    boot_difference = np.empty((n_boot, 2), dtype=np.float64)
    boot_m = np.empty(n_boot, dtype=np.float64)
    for index in range(n_boot):
        selected = rng.choice(cases, size=len(cases), replace=True)
        sim = _response(measured_plus, measured_minus, selected, h)
        model = np.asarray(
            [
                _response(plus, minus, selected, h)
                for plus, minus in zip(predicted_plus, predicted_minus)
            ]
        ).mean(axis=0)
        boot_measured[index] = sim
        boot_predicted[index] = model
        boot_difference[index] = model - sim
        boot_m[index] = sim[0] / model[0] - 1.0

    plus_mean, minus_mean = _means(measured_plus, measured_minus, cases)
    predicted_means = [
        _means(plus, minus, cases)
        for plus, minus in zip(predicted_plus, predicted_minus)
    ]
    model_plus_mean = np.mean([value[0] for value in predicted_means], axis=0)
    model_minus_mean = np.mean([value[1] for value in predicted_means], axis=0)
    seed_sem = (
        predicted_by_seed.std(axis=0, ddof=1) / np.sqrt(len(predicted_by_seed))
        if len(predicted_by_seed) > 1
        else np.full(2, np.nan)
    )
    case_se_difference = boot_difference.std(axis=0, ddof=1)
    return {
        "measured_plus_mean": plus_mean.tolist(),
        "measured_minus_mean": minus_mean.tolist(),
        "predicted_plus_mean": model_plus_mean.tolist(),
        "predicted_minus_mean": model_minus_mean.tolist(),
        "measured_response": measured.tolist(),
        "measured_case_standard_error": boot_measured.std(axis=0, ddof=1).tolist(),
        "predicted_response": predicted.tolist(),
        "predicted_response_by_model_seed": predicted_by_seed.tolist(),
        "predicted_case_standard_error": boot_predicted.std(axis=0, ddof=1).tolist(),
        "predicted_model_seed_sem": seed_sem.tolist(),
        "predicted_minus_measured": difference.tolist(),
        "difference_case_standard_error": case_se_difference.tolist(),
        "difference_total_standard_error": np.hypot(
            case_se_difference, seed_sem
        ).tolist(),
        "m_e1_measured_over_predicted_minus_one": float(m),
        "m_e1_case_standard_error": float(boot_m.std(ddof=1)),
        "bootstrap_ci95": {
            "measured_response": np.quantile(
                boot_measured, [0.025, 0.975], axis=0
            ).tolist(),
            "predicted_response": np.quantile(
                boot_predicted, [0.025, 0.975], axis=0
            ).tolist(),
            "predicted_minus_measured": np.quantile(
                boot_difference, [0.025, 0.975], axis=0
            ).tolist(),
            "m_e1": np.quantile(boot_m, [0.025, 0.975]).tolist(),
        },
    }


def _single_case_summary(
    measured_plus: dict,
    measured_minus: dict,
    predicted_plus: list[dict],
    predicted_minus: list[dict],
    *,
    case: int,
    h: float,
) -> dict:
    selected = np.asarray([case], dtype=np.int64)
    measured = _response(measured_plus, measured_minus, selected, h)
    predicted_by_seed = np.asarray(
        [
            _response(plus, minus, selected, h)
            for plus, minus in zip(predicted_plus, predicted_minus)
        ]
    )
    predicted = predicted_by_seed.mean(axis=0)
    measured_leg_means = _means(measured_plus, measured_minus, selected)
    predicted_leg_means = [
        _means(plus, minus, selected)
        for plus, minus in zip(predicted_plus, predicted_minus)
    ]
    seed_sem = (
        predicted_by_seed.std(axis=0, ddof=1) / np.sqrt(len(predicted_by_seed))
        if len(predicted_by_seed) > 1
        else np.full(2, np.nan)
    )
    return {
        "plus_count": int(measured_plus[case]["count"]),
        "minus_count": int(measured_minus[case]["count"]),
        "measured_plus_mean": measured_leg_means[0].tolist(),
        "measured_minus_mean": measured_leg_means[1].tolist(),
        "predicted_plus_mean": np.mean(
            [value[0] for value in predicted_leg_means], axis=0
        ).tolist(),
        "predicted_minus_mean": np.mean(
            [value[1] for value in predicted_leg_means], axis=0
        ).tolist(),
        "measured_response": measured.tolist(),
        "predicted_response": predicted.tolist(),
        "predicted_response_by_model_seed": predicted_by_seed.tolist(),
        "predicted_model_seed_sem": seed_sem.tolist(),
        "predicted_minus_measured": (predicted - measured).tolist(),
        "m_e1_measured_over_predicted_minus_one": float(
            measured[0] / predicted[0] - 1.0
        ),
        "uncertainty_note": (
            "one case has no case-to-case standard error; only four-flow-seed SEM is reported"
        ),
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plus-catalogue", type=Path, required=True)
    parser.add_argument("--minus-catalogue", type=Path, required=True)
    parser.add_argument("--crowd-lookup", type=Path, required=True)
    parser.add_argument("--rblend-lookup", type=Path, required=True)
    parser.add_argument("--measurement-model", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-case", type=int, default=40)
    parser.add_argument("--max-case", type=int, default=50)
    parser.add_argument("--h", type=float, default=0.02)
    parser.add_argument("--draws", type=int, default=64)
    parser.add_argument("--sampling-seed", type=int, default=7301)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--n-boot", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260901)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if (
        args.max_case <= args.min_case
        or args.h <= 0
        or args.draws <= 0
        or args.batch_size <= 0
    ):
        raise ValueError("invalid case range, h, draws, or batch size")
    cases = np.arange(args.min_case, args.max_case, dtype=np.int64)

    print("loading keyed R_blend and crowding conditions", flush=True)
    rblend = _read_case_window(
        args.rblend_lookup, ("input_index", "R_blend"), args.min_case, args.max_case
    )
    crowd = _read_case_window(
        args.crowd_lookup, ("input_index", *CROWD_COLUMNS), args.min_case, args.max_case
    )
    lookup = rblend.merge(
        crowd,
        on=["case", "input_index"],
        how="inner",
        validate="one_to_one",
        sort=False,
    )
    if lookup[["case", "input_index"]].duplicated().any():
        raise ValueError("combined lookup contains duplicate identities")

    print("loading and preparing plus leg", flush=True)
    plus, plus_report = _prepare_leg(
        args.plus_catalogue,
        lookup,
        min_case=args.min_case,
        max_case=args.max_case,
        expected_sign=args.h,
    )
    print("loading and preparing minus leg", flush=True)
    minus, minus_report = _prepare_leg(
        args.minus_catalogue,
        lookup,
        min_case=args.min_case,
        max_case=args.max_case,
        expected_sign=-args.h,
    )
    shared_keys = np.intersect1d(
        plus["_key"].to_numpy(np.int64), minus["_key"].to_numpy(np.int64)
    )
    plus_position = np.searchsorted(plus["_key"].to_numpy(np.int64), shared_keys)
    minus_position = np.searchsorted(minus["_key"].to_numpy(np.int64), shared_keys)
    matched_plus = plus.iloc[plus_position].reset_index(drop=True)
    matched_minus = minus.iloc[minus_position].reset_index(drop=True)
    if not np.array_equal(matched_plus["_key"], matched_minus["_key"]):
        raise RuntimeError("matched legs are not identity aligned")
    invariant_columns = (
        "Re_input_p",
        "r_input_p",
        "sersic_n_input_p",
        "axis_ratio_input_p",
        "position_angle_input_p",
        "R_blend",
        *CROWD_COLUMNS,
    )
    matched_truth_max_abs_difference = {}
    for name in invariant_columns:
        difference = np.abs(
            matched_plus[name].to_numpy(float) - matched_minus[name].to_numpy(float)
        )
        matched_truth_max_abs_difference[name] = float(difference.max(initial=0.0))
        if not np.allclose(
            matched_plus[name], matched_minus[name], rtol=0.0, atol=2e-12
        ):
            raise RuntimeError(f"matched leg truth differs in {name}")

    union = pd.concat([plus, minus], ignore_index=True)
    union = union.drop_duplicates("_key", keep="first")
    union = union.sort_values("_key", kind="stable").reset_index(drop=True)
    union_keys = union["_key"].to_numpy(np.int64)
    plus_union_position = np.searchsorted(
        union_keys, plus["_key"].to_numpy(np.int64)
    )
    minus_union_position = np.searchsorted(
        union_keys, minus["_key"].to_numpy(np.int64)
    )

    measured_plus_values = plus[["measured_e1", "measured_e2"]].to_numpy(float)
    measured_minus_values = minus[["measured_e1", "measured_e2"]].to_numpy(float)
    measured_unmatched_plus = _sufficient(plus, measured_plus_values, cases)
    measured_unmatched_minus = _sufficient(minus, measured_minus_values, cases)
    measured_matched_plus = _sufficient(
        matched_plus,
        matched_plus[["measured_e1", "measured_e2"]].to_numpy(float),
        cases,
    )
    measured_matched_minus = _sufficient(
        matched_minus,
        matched_minus[["measured_e1", "measured_e2"]].to_numpy(float),
        cases,
    )

    predicted_unmatched_plus = []
    predicted_unmatched_minus = []
    predicted_matched_plus = []
    predicted_matched_minus = []
    model_records = []
    for model_path in args.measurement_model:
        print(f"evaluating paired flow expectation: {model_path.name}", flush=True)
        bundle = load_measurement_model(str(model_path), device=args.device)
        checkpoint_cuts = bundle.metadata.get("selection_cuts")
        if checkpoint_cuts is None or not np.allclose(
            np.asarray(checkpoint_cuts, dtype=float),
            np.asarray(POPULATION_CUTS, dtype=float),
            rtol=0.0,
            atol=0.0,
        ):
            raise ValueError(
                f"{model_path} selection cuts do not match the registered population"
            )
        union_plus_prediction, union_minus_prediction = _paired_predictions(
            bundle,
            union,
            h=args.h,
            draws=args.draws,
            sampling_seed=args.sampling_seed,
            batch_size=args.batch_size,
        )
        plus_prediction = union_plus_prediction[plus_union_position]
        minus_prediction = union_minus_prediction[minus_union_position]
        predicted_unmatched_plus.append(_sufficient(plus, plus_prediction, cases))
        predicted_unmatched_minus.append(_sufficient(minus, minus_prediction, cases))
        predicted_matched_plus.append(
            _sufficient(matched_plus, plus_prediction[plus_position], cases)
        )
        predicted_matched_minus.append(
            _sufficient(matched_minus, minus_prediction[minus_position], cases)
        )
        model_records.append({
            "path": str(model_path),
            "sha256": _sha256(model_path),
            "training_seed": bundle.metadata.get("seed"),
        })
        del (
            bundle,
            union_plus_prediction,
            union_minus_prediction,
            plus_prediction,
            minus_prediction,
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    matched = _bootstrap_summary(
        measured_matched_plus,
        measured_matched_minus,
        predicted_matched_plus,
        predicted_matched_minus,
        cases=cases,
        h=args.h,
        n_boot=args.n_boot,
        seed=args.bootstrap_seed,
    )
    unmatched = _bootstrap_summary(
        measured_unmatched_plus,
        measured_unmatched_minus,
        predicted_unmatched_plus,
        predicted_unmatched_minus,
        cases=cases,
        h=args.h,
        n_boot=args.n_boot,
        seed=args.bootstrap_seed + 1,
    )
    matched["plus_count"] = int(len(matched_plus))
    matched["minus_count"] = int(len(matched_minus))
    unmatched["plus_count"] = int(len(plus))
    unmatched["minus_count"] = int(len(minus))
    per_case = {
        str(int(case)): {
            "matched": _single_case_summary(
                measured_matched_plus,
                measured_matched_minus,
                predicted_matched_plus,
                predicted_matched_minus,
                case=int(case),
                h=args.h,
            ),
            "unmatched_per_leg_means": _single_case_summary(
                measured_unmatched_plus,
                measured_unmatched_minus,
                predicted_unmatched_plus,
                predicted_unmatched_minus,
                case=int(case),
                h=args.h,
            ),
        }
        for case in cases
    }

    result = {
        "analysis": "two-leg ConstGold measured-versus-flow-plus-R_blend response",
        "matrix_convention": (
            "reported vector is the first response-matrix column [R11,R21]; "
            "ConstGold excites g1 only"
        ),
        "case_range": [int(args.min_case), int(args.max_case)],
        "n_cases": int(len(cases)),
        "h": float(args.h),
        "population": {
            "true_cuts": {
                "r_input_p": [18.0, 25.8],
                "Re_input_p_arcsec": [0.5, 1.5],
                "neighbour": "0 < distance < 5 arcsec OR not neighbored",
            },
            "measured_output_selection": "none",
            "plus": plus_report,
            "minus": minus_report,
            "matched_identity_count": int(len(shared_keys)),
            "plus_only_count": int(len(plus) - len(shared_keys)),
            "minus_only_count": int(len(minus) - len(shared_keys)),
            "union_identity_count": int(len(union)),
            "matched_truth_max_abs_difference": matched_truth_max_abs_difference,
        },
        "estimators": {
            "matched": {
                "definition": (
                    "same usable both-detected identities in both signs; no measured-output selection"
                ),
                **matched,
            },
            "unmatched_per_leg_means": {
                "definition": (
                    "difference of independently detected usable leg means; detection membership "
                    "is inherited from ConstGold rather than predicted"
                ),
                **unmatched,
            },
        },
        "per_case": per_case,
        "prediction": {
            "method": (
                "full conditional-flow expectation from randomly shifted Sobol QMC; "
                "common random numbers across signs on the union of leg identities"
            ),
            "blend_shift": "R_blend * [e(S_g z) - e(z)] evaluated separately at +/-h",
            "draws_per_identity_per_model": int(args.draws),
            "sampling_seed": int(args.sampling_seed),
            "models": model_records,
        },
        "bootstrap": {
            "unit": "ConstGold case",
            "replicates": int(args.n_boot),
            "seed": int(args.bootstrap_seed),
        },
        "inputs": {
            "plus_catalogue": str(args.plus_catalogue),
            "minus_catalogue": str(args.minus_catalogue),
            "crowd_lookup": str(args.crowd_lookup),
            "rblend_lookup": str(args.rblend_lookup),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"population": result["population"], "estimators": result["estimators"]}, indent=2))


if __name__ == "__main__":
    main()
