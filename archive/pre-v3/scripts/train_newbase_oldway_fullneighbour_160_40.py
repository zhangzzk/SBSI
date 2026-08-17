#!/usr/bin/env python3
"""Train the frozen two-stage R_blend stack with full-neighbour context.

The case split is deliberately simple and case-disjoint at final validation:

* cases 0--159 receive the historical seeded 80/20 random-row split;
* the 80% rows fit one base;
* the correction fits either the disjoint 20% rows (default) or the same 80%;
* every labelled pair in cases 160--199 is final validation only.

The base is frozen before correction targets or context are constructed.  The
correction keeps the exact successful early 180-tree recipe and ordinary pair
residual MSE.  Its scene-response and multiplicity features are computed over
all deployed neighbours, including the unsheared half.  Supervised targets
remain restricted to the labelled sheared-neighbour catalogue.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import time
from typing import Any

import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split

from scripts.tune_sequential_rblend_optuna import (
    BASE_FEATURES,
    CONDITIONAL_FEATURES,
    atomic_json,
    atomic_model,
    json_clean,
    r2_score,
    xgb_common,
)
from scripts.v22_grouped_rscene_common import (
    conditional_matrix,
    load_source_metadata,
    make_scene_edges,
    assign_scene_bins,
    mmap_array,
    sha256,
)


FIT_CASE_MIN = 0
FIT_CASE_MAX = 159
FINAL_CASE_MIN = 160
FINAL_CASE_MAX = 199
N_CASES = 200
N_SCENE_BINS = 20
CORRECTION_TREES = 180
CORRECTION_PARAMS = {
    "objective": "reg:squarederror",
    "tree_method": "hist",
    "device": "cpu",
    "booster": "gbtree",
    "eval_metric": "rmse",
    "disable_default_eval_metric": 0,
    "max_bin": 256,
    "base_score": 0.0,
    "seed": 20260815,
    "subsample": 0.8,
    "colsample_bytree": 0.9,
    "learning_rate": 0.05,
    "max_depth": 5,
    "min_child_weight": 2000,
    "reg_lambda": 10.0,
    "reg_alpha": 0.0,
    "gamma": 0.0,
}


def load_base_recipe(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    recipe = payload["fixed_params"]
    if len(recipe["base"]) != 8 or int(recipe["base_rounds"]) <= 0:
        raise RuntimeError("new-base parameter summary is incomplete")
    return {"params": recipe["base"], "rounds": int(recipe["base_rounds"])}


def scene_index(
    case: np.ndarray, primary: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    change = np.empty(len(case), dtype=bool)
    change[0] = True
    change[1:] = (case[1:] != case[:-1]) | (primary[1:] != primary[:-1])
    starts = np.flatnonzero(change).astype(np.int64)
    counts = np.diff(np.append(starts, len(case))).astype(np.int32)
    scene_case = np.asarray(case[starts], dtype=np.int16)
    case_count = np.bincount(scene_case.astype(np.int64), minlength=N_CASES)
    offsets = np.concatenate(([0], np.cumsum(case_count))).astype(np.int64)
    if int(counts.sum()) != len(case) or int(offsets[-1]) != len(starts):
        raise RuntimeError("source scene indexing does not close")
    return starts, counts, scene_case, offsets


def metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    residual = np.asarray(target, dtype=np.float64) - np.asarray(
        prediction, dtype=np.float64
    )
    return {
        "r2": r2_score(target, prediction),
        "mse": float(np.mean(np.square(residual))),
        "residual_mean": float(residual.mean()),
    }


def case_balanced_pair_metrics(
    target: np.ndarray, prediction: np.ndarray, case: np.ndarray
) -> dict[str, Any]:
    output: dict[str, Any] = metrics(target, prediction)
    residual = np.asarray(target, dtype=np.float64) - np.asarray(
        prediction, dtype=np.float64
    )
    selected_case = np.asarray(case, dtype=np.int64)
    unique_case, inverse = np.unique(selected_case, return_inverse=True)
    count = np.bincount(inverse)
    total = np.bincount(inverse, weights=residual)
    mean = total / count
    output.update({
        "n_rows": int(len(residual)),
        "n_cases": int(len(unique_case)),
        "case_balanced_residual_mean": float(mean.mean()),
        "case_residual_sem": float(mean.std(ddof=1) / np.sqrt(len(mean))),
    })
    return output


def scene_metrics(
    truth: np.ndarray,
    active_prediction: np.ndarray,
    coordinate: np.ndarray,
    case: np.ndarray,
) -> dict[str, Any]:
    truth64 = np.asarray(truth, dtype=np.float64)
    prediction64 = np.asarray(active_prediction, dtype=np.float64)
    coordinate64 = np.asarray(coordinate, dtype=np.float64)
    case64 = np.asarray(case, dtype=np.int64)
    residual = truth64 - prediction64
    unique_case, inverse = np.unique(case64, return_inverse=True)
    count_case = np.bincount(inverse)
    total_case = np.bincount(inverse, weights=residual)
    mean_case = total_case / count_case
    edges = make_scene_edges(coordinate64, n_quantile_bins=N_SCENE_BINS)
    scene_bin = assign_scene_bins(coordinate64, edges).astype(np.int64)
    flat = inverse * N_SCENE_BINS + scene_bin
    count = np.bincount(
        flat, minlength=len(unique_case) * N_SCENE_BINS
    ).reshape(len(unique_case), N_SCENE_BINS)
    total = np.bincount(
        flat, weights=residual, minlength=len(unique_case) * N_SCENE_BINS
    ).reshape(len(unique_case), N_SCENE_BINS)
    curve = np.divide(
        total,
        count,
        out=np.full_like(total, np.nan, dtype=np.float64),
        where=count > 0,
    )
    case_balanced_curve = np.nanmean(curve, axis=0)
    output: dict[str, Any] = {
        "n_scenes": int(len(residual)),
        "n_cases": int(len(unique_case)),
        "row_weighted_residual_mean": float(residual.mean()),
        "case_balanced_residual_mean": float(mean_case.mean()),
        "case_residual_sem": float(
            mean_case.std(ddof=1) / np.sqrt(len(mean_case))
        ),
        "coordinate_mean": float(coordinate64.mean()),
        "curve_edges": edges,
        "case_balanced_curve": case_balanced_curve,
        "curve_rms": float(np.sqrt(np.nanmean(np.square(case_balanced_curve)))),
    }
    for name, threshold in (("tail_gt_0p1", 0.1), ("tail_gt_0p2", 0.2)):
        mask = coordinate64 > threshold
        local_case, local_inverse = np.unique(case64[mask], return_inverse=True)
        local_count = np.bincount(local_inverse)
        local_total = np.bincount(local_inverse, weights=residual[mask])
        local_mean = local_total / local_count
        output[name] = {
            "n_scenes": int(mask.sum()),
            "n_cases": int(len(local_case)),
            "case_balanced_residual_mean": float(local_mean.mean()),
            "case_residual_sem": float(
                local_mean.std(ddof=1) / np.sqrt(len(local_mean))
            ),
        }
    return output


def load_full_metadata(full_cache: Path, source_cache: Path) -> dict[str, Any]:
    with (full_cache / "metadata.json").open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    if metadata["case_window"] != [0, N_CASES - 1]:
        raise RuntimeError("full-neighbour cache does not cover cases 0--199")
    if metadata["base_features"] != BASE_FEATURES:
        raise RuntimeError("full-neighbour base features drifted")
    if metadata["source_metadata_sha256"] != sha256(
        source_cache / "metadata.json"
    ):
        raise RuntimeError("full-neighbour/source cache mismatch")
    neighbour = metadata["neighbour_definition"]
    if neighbour["population"] != "all rendered objects (both half-shear populations)":
        raise RuntimeError("full-neighbour cache is not the all-object population")
    return metadata


def full_path(cache: Path, metadata: dict[str, Any], name: str, case: int) -> Path:
    return cache / metadata["arrays"][name].format(case=case)


def physical_base_prediction(
    booster: xgb.Booster,
    x_scaled: np.ndarray,
    target_mean: float,
    target_scale: float,
) -> np.ndarray:
    prediction = booster.inplace_predict(
        np.asarray(x_scaled, dtype=np.float32)
    ).astype(np.float64)
    prediction = prediction * target_scale + target_mean
    if not np.isfinite(prediction).all():
        raise RuntimeError("base returned non-finite prediction")
    return prediction.astype(np.float32)


def build_full_base_context(
    *,
    base: xgb.Booster,
    full_cache: Path,
    full_meta: dict[str, Any],
    source_primary: np.ndarray,
    source_scene_starts: np.ndarray,
    source_scene_offsets: np.ndarray,
    target_mean: float,
    target_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    n_source_scene = len(source_scene_starts)
    scene_prediction = np.full(n_source_scene, np.nan, dtype=np.float32)
    scene_count = np.zeros(n_source_scene, dtype=np.int16)
    for case in range(N_CASES):
        x_scaled = np.load(
            full_path(full_cache, full_meta, "x_scaled", case), mmap_mode="r"
        )
        pair_scene = np.load(
            full_path(full_cache, full_meta, "pair_scene", case), mmap_mode="r"
        )
        full_primary = np.load(
            full_path(full_cache, full_meta, "scene_primary", case), mmap_mode="r"
        )
        full_count = np.load(
            full_path(full_cache, full_meta, "scene_count", case), mmap_mode="r"
        )
        pair_prediction = physical_base_prediction(
            base, x_scaled, target_mean, target_scale
        )
        full_scene_prediction = np.bincount(
            np.asarray(pair_scene, dtype=np.int64),
            weights=pair_prediction.astype(np.float64),
            minlength=len(full_primary),
        ).astype(np.float32)
        s0, s1 = source_scene_offsets[case : case + 2]
        active_primary = np.asarray(
            source_primary[source_scene_starts[s0:s1]], dtype=np.int64
        )
        mapping = np.searchsorted(full_primary, active_primary)
        if (
            np.any(mapping >= len(full_primary))
            or not np.array_equal(np.asarray(full_primary)[mapping], active_primary)
        ):
            raise RuntimeError(f"case {case}: source primary lacks full context")
        scene_prediction[s0:s1] = full_scene_prediction[mapping]
        scene_count[s0:s1] = np.asarray(full_count)[mapping]
        if case % 20 == 19:
            print(f"FULL_BASE_CONTEXT_DONE through case {case}", flush=True)
        del x_scaled, pair_scene, full_primary, full_count
        del pair_prediction, full_scene_prediction, active_primary, mapping
    if not np.isfinite(scene_prediction).all() or np.any(scene_count <= 0):
        raise RuntimeError("full-neighbour scene context is incomplete")
    return scene_prediction, scene_count


def corrected_full_coordinate(
    *,
    cases: range,
    base: xgb.Booster,
    correction: xgb.Booster,
    full_cache: Path,
    full_meta: dict[str, Any],
    source_primary: np.ndarray,
    source_scene_starts: np.ndarray,
    source_scene_offsets: np.ndarray,
    target_mean: float,
    target_scale: float,
) -> np.ndarray:
    output = np.full(len(source_scene_starts), np.nan, dtype=np.float32)
    for case in cases:
        x_scaled = np.load(
            full_path(full_cache, full_meta, "x_scaled", case), mmap_mode="r"
        )
        x_aux = np.load(
            full_path(full_cache, full_meta, "x_aux", case), mmap_mode="r"
        )
        pair_scene = np.load(
            full_path(full_cache, full_meta, "pair_scene", case), mmap_mode="r"
        )
        full_primary = np.load(
            full_path(full_cache, full_meta, "scene_primary", case), mmap_mode="r"
        )
        full_count = np.load(
            full_path(full_cache, full_meta, "scene_count", case), mmap_mode="r"
        )
        base_pair = physical_base_prediction(base, x_scaled, target_mean, target_scale)
        full_scene_base = np.bincount(
            np.asarray(pair_scene, dtype=np.int64),
            weights=base_pair.astype(np.float64),
            minlength=len(full_primary),
        ).astype(np.float32)
        features = np.empty((len(base_pair), len(CONDITIONAL_FEATURES)), dtype=np.float32)
        features[:, :7] = np.asarray(x_scaled, dtype=np.float32)
        features[:, 7] = base_pair
        features[:, 8:10] = np.asarray(x_aux, dtype=np.float32)
        features[:, 10] = full_scene_base[np.asarray(pair_scene, dtype=np.int64)]
        features[:, 11] = np.log1p(
            np.asarray(full_count, dtype=np.float32)[
                np.asarray(pair_scene, dtype=np.int64)
            ]
        )
        correction_pair = (
            correction.inplace_predict(features).astype(np.float64) * target_scale
        )
        full_scene_combined = np.bincount(
            np.asarray(pair_scene, dtype=np.int64),
            weights=base_pair.astype(np.float64) + correction_pair,
            minlength=len(full_primary),
        ).astype(np.float32)
        s0, s1 = source_scene_offsets[case : case + 2]
        active_primary = np.asarray(
            source_primary[source_scene_starts[s0:s1]], dtype=np.int64
        )
        mapping = np.searchsorted(full_primary, active_primary)
        if (
            np.any(mapping >= len(full_primary))
            or not np.array_equal(np.asarray(full_primary)[mapping], active_primary)
        ):
            raise RuntimeError(f"case {case}: final source primary lacks context")
        output[s0:s1] = full_scene_combined[mapping]
        del x_scaled, x_aux, pair_scene, full_primary, full_count
        del base_pair, full_scene_base, features, correction_pair
        del full_scene_combined, active_primary, mapping
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-cache", required=True)
    parser.add_argument("--full-neighbour-cache", required=True)
    parser.add_argument("--base-parameter-summary", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fit-case-min", type=int, default=FIT_CASE_MIN)
    parser.add_argument("--fit-case-max", type=int, default=FIT_CASE_MAX)
    parser.add_argument("--final-case-min", type=int, default=FINAL_CASE_MIN)
    parser.add_argument("--final-case-max", type=int, default=FINAL_CASE_MAX)
    parser.add_argument(
        "--scene-context", choices=("half", "full"), default="full"
    )
    parser.add_argument(
        "--correction-split",
        choices=("validation", "base"),
        default="validation",
        help=(
            "rows used to fit the residual correction: the disjoint 20%% "
            "validation rows or the same 80%% rows used by the base"
        ),
    )
    args = parser.parse_args()

    source_cache = Path(args.source_cache).resolve()
    full_cache = Path(args.full_neighbour_cache).resolve()
    parameter_summary = Path(args.base_parameter_summary).resolve()
    fit_case_min = int(args.fit_case_min)
    fit_case_max = int(args.fit_case_max)
    final_case_min = int(args.final_case_min)
    final_case_max = int(args.final_case_max)
    scene_context = str(args.scene_context)
    correction_split = str(args.correction_split)
    if not (
        0 <= fit_case_min <= fit_case_max < N_CASES
        and 0 <= final_case_min <= final_case_max < N_CASES
    ):
        raise ValueError("case windows must lie within 0--199")
    if not (fit_case_max < final_case_min or final_case_max < fit_case_min):
        raise ValueError("fit and final-validation case windows overlap")
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / "summary.json"
    if summary_path.exists():
        print(f"New-base full-neighbour 160/40 stack complete: {summary_path}")
        return

    source_meta = load_source_metadata(source_cache)
    full_meta = (
        load_full_metadata(full_cache, source_cache)
        if scene_context == "full"
        else None
    )
    case = mmap_array(source_cache, source_meta, "case")
    primary = mmap_array(source_cache, source_meta, "input_index")
    label = mmap_array(source_cache, source_meta, "label")
    x_scaled = mmap_array(source_cache, source_meta, "x_scaled")
    x_raw = mmap_array(source_cache, source_meta, "x_raw")
    starts, counts, scene_case, scene_offsets = scene_index(case, primary)

    fit_case = (np.asarray(case) >= fit_case_min) & (
        np.asarray(case) <= fit_case_max
    )
    fit_index = np.flatnonzero(fit_case).astype(np.int32)
    split = source_meta["official_random_row_split"]
    local = np.arange(len(fit_index), dtype=np.int32)
    base_local, correction_local = train_test_split(
        local,
        test_size=float(split["test_size"]),
        random_state=int(split["random_state"]),
    )
    base_mask = np.zeros(len(fit_index), dtype=bool)
    base_mask[base_local] = True
    base_index = fit_index[base_mask]
    validation_index = fit_index[~base_mask]
    correction_index = (
        base_index if correction_split == "base" else validation_index
    )
    split_matches_stored_official = None
    if fit_case_min == 40 and fit_case_max == 199:
        official_train = mmap_array(
            source_cache, source_meta, "official_train"
        )
        split_matches_stored_official = bool(np.array_equal(
            base_mask, np.asarray(official_train[fit_index], dtype=bool)
        ))
        if not split_matches_stored_official:
            raise RuntimeError("reconstructed 40--199 split differs from official mask")
    final_index = np.flatnonzero(
        (np.asarray(case) >= final_case_min)
        & (np.asarray(case) <= final_case_max)
    ).astype(np.int32)
    if (
        not len(base_index)
        or not len(validation_index)
        or not len(correction_index)
        or not len(final_index)
    ):
        raise RuntimeError("one or more requested case partitions is empty")
    del fit_case, fit_index, local, base_local, correction_local, base_mask

    standardization = source_meta["source_standardization"]
    target_mean = float(standardization["mean"])
    target_scale = float(standardization["std"])
    recipe = load_base_recipe(parameter_summary)
    base_params = xgb_common() | recipe["params"]
    base_target = (
        np.asarray(label[base_index], dtype=np.float64) - target_mean
    ) / target_scale
    dbase = xgb.DMatrix(
        np.asarray(x_scaled[base_index], dtype=np.float32),
        label=base_target.astype(np.float32),
        feature_names=BASE_FEATURES,
    )
    started = time.time()
    base = xgb.train(
        base_params, dbase, num_boost_round=recipe["rounds"], verbose_eval=False
    )
    base_fit_seconds = time.time() - started
    base_train_prediction = (
        base.predict(dbase).astype(np.float64) * target_scale + target_mean
    )
    base_train_metrics = metrics(label[base_index], base_train_prediction)
    del dbase, base_target, base_train_prediction
    gc.collect()

    all_base_prediction = physical_base_prediction(
        base, x_scaled, target_mean, target_scale
    )
    if scene_context == "full":
        if full_meta is None:
            raise RuntimeError("full-neighbour metadata was not loaded")
        scene_base, scene_count = build_full_base_context(
            base=base,
            full_cache=full_cache,
            full_meta=full_meta,
            source_primary=primary,
            source_scene_starts=starts,
            source_scene_offsets=scene_offsets,
            target_mean=target_mean,
            target_scale=target_scale,
        )
    else:
        scene_base = np.add.reduceat(
            all_base_prediction.astype(np.float64), starts
        ).astype(np.float32)
        scene_count = counts.astype(np.int16)
    row_scene_base = np.repeat(scene_base, counts)
    row_scene_count = np.repeat(scene_count, counts).astype(np.int32)
    if len(row_scene_base) != len(case) or len(row_scene_count) != len(case):
        raise RuntimeError("full-neighbour context expansion failed")

    correction_features = conditional_matrix(
        x_scaled,
        x_raw,
        all_base_prediction,
        row_scene_base,
        row_scene_count,
        correction_index,
    )
    correction_physical_target = (
        np.asarray(label[correction_index], dtype=np.float64)
        - np.asarray(all_base_prediction[correction_index], dtype=np.float64)
    )
    correction_target = (correction_physical_target / target_scale).astype(np.float32)
    correction_params = CORRECTION_PARAMS | {
        "n_jobs": int(os.environ.get("SLURM_CPUS_PER_TASK", "16"))
    }
    dcorrection = xgb.DMatrix(
        correction_features,
        label=correction_target,
        feature_names=CONDITIONAL_FEATURES,
    )
    started = time.time()
    correction = xgb.train(
        correction_params,
        dcorrection,
        num_boost_round=CORRECTION_TREES,
        verbose_eval=False,
    )
    correction_fit_seconds = time.time() - started
    correction_prediction = (
        correction.predict(dcorrection).astype(np.float64) * target_scale
    )
    correction_base = np.asarray(
        all_base_prediction[correction_index], dtype=np.float64
    )
    correction_before_metrics = metrics(label[correction_index], correction_base)
    correction_after_metrics = metrics(
        label[correction_index], correction_base + correction_prediction
    )
    del correction_features, correction_physical_target, correction_target
    del dcorrection, correction_prediction, correction_base
    gc.collect()

    final_features = conditional_matrix(
        x_scaled,
        x_raw,
        all_base_prediction,
        row_scene_base,
        row_scene_count,
        final_index,
    )
    final_correction = (
        correction.inplace_predict(final_features).astype(np.float64) * target_scale
    )
    final_base = np.asarray(all_base_prediction[final_index], dtype=np.float64)
    final_combined = final_base + final_correction
    final_pair = {
        "base": case_balanced_pair_metrics(
            label[final_index], final_base, case[final_index]
        ),
        "combined": case_balanced_pair_metrics(
            label[final_index], final_combined, case[final_index]
        ),
    }

    scene_truth = np.add.reduceat(np.asarray(label, dtype=np.float64), starts)
    scene_base_active = np.add.reduceat(
        all_base_prediction.astype(np.float64), starts
    )
    all_combined = all_base_prediction.astype(np.float64)
    all_combined[final_index] = final_combined
    scene_combined_active = np.add.reduceat(all_combined, starts)
    del all_combined
    final_scene_mask = (scene_case >= final_case_min) & (
        scene_case <= final_case_max
    )
    if scene_context == "full":
        if full_meta is None:
            raise RuntimeError("full-neighbour metadata was not loaded")
        final_corrected_coordinate = corrected_full_coordinate(
            cases=range(final_case_min, final_case_max + 1),
            base=base,
            correction=correction,
            full_cache=full_cache,
            full_meta=full_meta,
            source_primary=primary,
            source_scene_starts=starts,
            source_scene_offsets=scene_offsets,
            target_mean=target_mean,
            target_scale=target_scale,
        )
    else:
        final_corrected_coordinate = np.full(
            len(starts), np.nan, dtype=np.float32
        )
        final_corrected_coordinate[final_scene_mask] = (
            scene_combined_active[final_scene_mask]
        )
    if not np.isfinite(final_corrected_coordinate[final_scene_mask]).all():
        raise RuntimeError("final corrected full-neighbour coordinate is incomplete")
    final_scene = {
        f"base_conditioned_on_{scene_context}_base_sum": scene_metrics(
            scene_truth[final_scene_mask],
            scene_base_active[final_scene_mask],
            scene_base[final_scene_mask],
            scene_case[final_scene_mask],
        ),
        f"combined_conditioned_on_{scene_context}_base_sum": scene_metrics(
            scene_truth[final_scene_mask],
            scene_combined_active[final_scene_mask],
            scene_base[final_scene_mask],
            scene_case[final_scene_mask],
        ),
        f"combined_conditioned_on_{scene_context}_corrected_sum": scene_metrics(
            scene_truth[final_scene_mask],
            scene_combined_active[final_scene_mask],
            final_corrected_coordinate[final_scene_mask],
            scene_case[final_scene_mask],
        ),
    }

    base_path = output / "base.json"
    correction_path = output / "correction.json"
    atomic_model(base, base_path)
    atomic_model(correction, correction_path)
    context_population = (
        "all deployed neighbours, including the unsheared half"
        if scene_context == "full"
        else "labelled half-shear neighbours only"
    )
    correction_split_description = (
        f"same seeded 80% random-row training split as the base within "
        f"cases {fit_case_min}--{fit_case_max}"
        if correction_split == "base"
        else f"disjoint seeded 20% random-row validation split within "
        f"cases {fit_case_min}--{fit_case_max}"
    )
    payload = {
        "schema_version": 1,
        "tuning_performed": False,
        "base_parameter_source": str(parameter_summary),
        "base_parameter_source_sha256": sha256(parameter_summary),
        "features": {"base": BASE_FEATURES, "correction": CONDITIONAL_FEATURES},
        "source": {
            "cache": str(source_cache),
            "metadata_sha256": sha256(source_cache / "metadata.json"),
            "full_neighbour_cache": str(full_cache),
            "full_neighbour_metadata_sha256": sha256(full_cache / "metadata.json"),
            "scene_context_population": scene_context,
            "full_neighbour_cache_used": scene_context == "full",
            "target_mean": target_mean,
            "target_scale": target_scale,
        },
        "training": {
            "case_window": [fit_case_min, fit_case_max],
            "base": {
                "split": f"seeded 80% random-row training split within cases {fit_case_min}--{fit_case_max}",
                "n_rows": int(len(base_index)),
                "rounds": int(recipe["rounds"]),
                "params": base_params,
                "fit_seconds": base_fit_seconds,
                "in_sample_metrics": base_train_metrics,
            },
            "correction": {
                "split": correction_split_description,
                "split_name": correction_split,
                "n_rows": int(len(correction_index)),
                "target": "labelled sheared-pair response minus frozen base prediction",
                "scene_context": f"sum frozen-base predictions over {context_population}",
                "multiplicity_context": context_population,
                "rounds": CORRECTION_TREES,
                "params": correction_params,
                "loss": "ordinary pair residual MSE; no scene-level term",
                "fit_seconds": correction_fit_seconds,
                "before_metrics": correction_before_metrics,
                "after_metrics": correction_after_metrics,
            },
        },
        "final_case_disjoint_validation": {
            "case_window": [final_case_min, final_case_max],
            "n_rows": int(len(final_index)),
            "pair": final_pair,
            "scene": final_scene,
        },
        "deployment_stack": {
            "base_model": str(base_path),
            "base_sha256": sha256(base_path),
            "correction_model": str(correction_path),
            "correction_sha256": sha256(correction_path),
            "base_is_the_exact_frozen_model_used_for_correction_targets": True,
        },
        "protocol": {
            "base_and_correction_rows_are_disjoint": correction_split
            == "validation",
            "correction_uses_same_rows_as_base": correction_split == "base",
            "random_row_split": {
                "random_state": int(split["random_state"]),
                "test_size": float(split["test_size"]),
            },
            "final_validation_cases_are_disjoint": True,
            "split_matches_stored_official_when_applicable": split_matches_stored_official,
            "base_retrained_after_correction_targets": False,
            "coherent_anchor_truth_opened": False,
            "constgold_opened": False,
        },
    }
    atomic_json(summary_path, payload)
    print(json.dumps(json_clean(payload), indent=2, sort_keys=True), flush=True)
    print("NEWBASE_OLDWAY_FULLNEIGHBOUR_160_40_DONE", flush=True)


if __name__ == "__main__":
    main()
