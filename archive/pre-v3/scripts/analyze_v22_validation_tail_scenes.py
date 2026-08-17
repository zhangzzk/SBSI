#!/usr/bin/env python3
"""Measure V2.2 validation-pair residuals in anchor-tail-like scenes.

The frozen V2.2 emulator used a row-random 80/20 split after its rectangular
pair cuts.  This script reconstructs the exact held-out rows, but assigns them
to complete primary scenes using the full supported pair list.  Scene selection
uses only frozen V2.2 predictions and latent primary properties; response labels
never enter a cut.

The main anchor analogue is the already-frozen coherent-anchor rule
``sum_pair_prediction > 0.1``.  A primary-magnitude ladder localizes the
"relatively faint" part without choosing a threshold from the validation
residual.  Held-out sums are inverse-probability scaled by the exact validation
fraction, and uncertainties use the rendered half-shear case as the unit.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc

from scripts.plot_v22_training_cumulative_response import (
    COND,
    CUT_NAMES,
    MODEL_DIR,
    PAIR_COLUMNS,
    PAIR_FEATURES,
    configure_style,
    json_clean,
    score_pairs,
)
from scripts.plot_v22_validation_cumulative_response import (
    exact_validation_mask,
    load_split_spec,
    row_validation_summary,
)


KEY = ["case", "input_index"]
SCENE_SUM_COLUMNS = [
    "n_pairs",
    "scene_prediction",
    "scene_abs_prediction",
    "total_neighbour_flux_ratio",
    "validation_pair_count",
    "validation_label_sum",
    "validation_prediction_sum",
    "validation_residual_sum",
    "validation_null_sum",
    "validation_residual_square_sum",
]
MAIN_SELECTIONS = [
    ("all_supported", None, None),
    ("anchor_tail", 0.1, None),
    ("anchor_tail_faint24", 0.1, 24.0),
    ("anchor_tail_faint24p5", 0.1, 24.5),
]
GRID_RESPONSE_THRESHOLDS = [0.05, 0.1, 0.2]
GRID_PRIMARY_MAG_MINIMA = [None, 23.5, 24.0, 24.5, 25.0]


def finite_stat(values: np.ndarray) -> dict[str, float | int]:
    """Return a case mean, standard deviation, and SEM."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or values.size < 2 or not np.isfinite(values).all():
        raise ValueError("finite_stat needs at least two finite case values")
    sd = float(values.std(ddof=1))
    return {
        "mean": float(values.mean()),
        "case_sd": sd,
        "case_sem": float(sd / np.sqrt(values.size)),
        "n_cases": int(values.size),
    }


def aggregate_pair_batch(
    pairs: pd.DataFrame,
    prediction: np.ndarray,
    validation: np.ndarray,
    shear: float,
) -> pd.DataFrame:
    """Collapse one ordered batch of supported pair rows to scene partials."""
    prediction = np.asarray(prediction, dtype=np.float64)
    validation = np.asarray(validation, dtype=bool)
    if len(pairs) == 0:
        raise ValueError("cannot aggregate an empty pair batch")
    if prediction.shape != (len(pairs),) or validation.shape != (len(pairs),):
        raise ValueError("prediction/validation shape does not match pair rows")
    if shear <= 0.0 or not np.isfinite(prediction).all():
        raise ValueError("invalid shear or prediction")

    label = pairs.delta_et1.to_numpy(np.float64) / shear
    null = pairs.delta_et2.to_numpy(np.float64) / shear
    if not np.isfinite(label).all() or not np.isfinite(null).all():
        raise RuntimeError("supported pair has a non-finite response label")
    residual = label - prediction
    flux_ratio = np.power(
        10.0,
        -0.4
        * (
            pairs.r_input_s.to_numpy(np.float64)
            - pairs.r_input_p.to_numpy(np.float64)
        ),
    )
    work = pairs[[*KEY, "r_input_p", "Re_input_p", "distance"]].copy()
    work["n_pairs"] = np.ones(len(work), dtype=np.int16)
    work["scene_prediction"] = prediction
    work["scene_abs_prediction"] = np.abs(prediction)
    work["total_neighbour_flux_ratio"] = flux_ratio
    work["max_neighbour_flux_ratio"] = flux_ratio
    work["validation_pair_count"] = validation.astype(np.int8)
    work["validation_label_sum"] = np.where(validation, label, 0.0)
    work["validation_prediction_sum"] = np.where(
        validation, prediction, 0.0
    )
    work["validation_residual_sum"] = np.where(
        validation, residual, 0.0
    )
    work["validation_null_sum"] = np.where(validation, null, 0.0)
    work["validation_residual_square_sum"] = np.where(
        validation, np.square(residual), 0.0
    )
    grouped = work.groupby(KEY, sort=False, observed=True)
    output = grouped.agg(
        primary_mag=("r_input_p", "first"),
        primary_mag_min=("r_input_p", "min"),
        primary_mag_max=("r_input_p", "max"),
        primary_size=("Re_input_p", "first"),
        primary_size_min=("Re_input_p", "min"),
        primary_size_max=("Re_input_p", "max"),
        n_pairs=("n_pairs", "sum"),
        scene_prediction=("scene_prediction", "sum"),
        scene_abs_prediction=("scene_abs_prediction", "sum"),
        total_neighbour_flux_ratio=("total_neighbour_flux_ratio", "sum"),
        max_neighbour_flux_ratio=("max_neighbour_flux_ratio", "max"),
        min_pair_distance=("distance", "min"),
        validation_pair_count=("validation_pair_count", "sum"),
        validation_label_sum=("validation_label_sum", "sum"),
        validation_prediction_sum=("validation_prediction_sum", "sum"),
        validation_residual_sum=("validation_residual_sum", "sum"),
        validation_null_sum=("validation_null_sum", "sum"),
        validation_residual_square_sum=(
            "validation_residual_square_sum", "sum"
        ),
    ).reset_index()
    for prefix in ("primary_mag", "primary_size"):
        spread = output[f"{prefix}_max"] - output[f"{prefix}_min"]
        if float(spread.max()) > 1.0e-12:
            raise RuntimeError(f"{prefix} is not constant within a scene")
    return output


def combine_scene_partials(partials: Iterable[pd.DataFrame]) -> pd.DataFrame:
    """Combine batch-local scene partials, retaining exact response sums."""
    pieces = list(partials)
    if not pieces:
        raise ValueError("no scene partials")
    frame = pd.concat(pieces, ignore_index=True)
    grouped = frame.groupby(KEY, sort=True, observed=True)
    scenes = grouped.agg(
        primary_mag=("primary_mag", "first"),
        primary_mag_min=("primary_mag_min", "min"),
        primary_mag_max=("primary_mag_max", "max"),
        primary_size=("primary_size", "first"),
        primary_size_min=("primary_size_min", "min"),
        primary_size_max=("primary_size_max", "max"),
        n_pairs=("n_pairs", "sum"),
        scene_prediction=("scene_prediction", "sum"),
        scene_abs_prediction=("scene_abs_prediction", "sum"),
        total_neighbour_flux_ratio=("total_neighbour_flux_ratio", "sum"),
        max_neighbour_flux_ratio=("max_neighbour_flux_ratio", "max"),
        min_pair_distance=("min_pair_distance", "min"),
        validation_pair_count=("validation_pair_count", "sum"),
        validation_label_sum=("validation_label_sum", "sum"),
        validation_prediction_sum=("validation_prediction_sum", "sum"),
        validation_residual_sum=("validation_residual_sum", "sum"),
        validation_null_sum=("validation_null_sum", "sum"),
        validation_residual_square_sum=(
            "validation_residual_square_sum", "sum"
        ),
    ).reset_index()
    for prefix in ("primary_mag", "primary_size"):
        spread = scenes[f"{prefix}_max"] - scenes[f"{prefix}_min"]
        if float(spread.max()) > 1.0e-12:
            raise RuntimeError(f"{prefix} is not constant across pair batches")
        scenes = scenes.drop(columns=[f"{prefix}_min", f"{prefix}_max"])
    numeric = scenes.drop(columns=KEY).to_numpy(float)
    if not np.isfinite(numeric).all():
        raise RuntimeError("scene table contains non-finite values")
    if np.any(scenes.n_pairs.to_numpy(np.int64) <= 0):
        raise RuntimeError("supported scene has no pair")
    if np.any(
        scenes.validation_pair_count.to_numpy(np.int64)
        > scenes.n_pairs.to_numpy(np.int64)
    ):
        raise RuntimeError("scene validation count exceeds full pair count")
    scenes["n_pairs"] = scenes.n_pairs.astype(np.int16)
    scenes["validation_pair_count"] = scenes.validation_pair_count.astype(
        np.int16
    )
    return scenes


def selection_mask(
    scenes: pd.DataFrame,
    response_threshold: float | None,
    primary_mag_min: float | None,
) -> np.ndarray:
    """Select a prediction-defined scene tail and optional faint-primary cut."""
    keep = np.ones(len(scenes), dtype=bool)
    if response_threshold is not None:
        keep &= scenes.scene_prediction.to_numpy(float) > response_threshold
    if primary_mag_min is not None:
        keep &= scenes.primary_mag.to_numpy(float) >= primary_mag_min
    return keep


def _case_sum(
    case_index: np.ndarray,
    values: np.ndarray,
    n_cases: int,
) -> np.ndarray:
    return np.bincount(
        case_index,
        weights=np.asarray(values, dtype=float),
        minlength=n_cases,
    ).astype(float)


def summarize_selection(
    scenes: pd.DataFrame,
    keep: np.ndarray,
    *,
    name: str,
    response_threshold: float | None,
    primary_mag_min: float | None,
    case_min: int,
    n_cases: int,
    validation_scale: float,
    eligible_primary_count: np.ndarray,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Summarize one scene subset with case-level uncertainty."""
    keep = np.asarray(keep, dtype=bool)
    if keep.shape != (len(scenes),) or not keep.any():
        raise ValueError(f"empty or malformed selection {name}")
    local = scenes.loc[keep]
    case_index = local.case.to_numpy(np.int64) - case_min
    if np.any(case_index < 0) or np.any(case_index >= n_cases):
        raise RuntimeError("selected scene case outside requested range")
    n_scene = np.bincount(case_index, minlength=n_cases).astype(np.int64)
    if np.any(n_scene == 0):
        missing = (np.flatnonzero(n_scene == 0) + case_min).tolist()
        raise RuntimeError(f"selection {name} has no scenes in cases {missing}")
    n_validation = _case_sum(
        case_index, local.validation_pair_count, n_cases
    )
    if np.any(n_validation == 0):
        missing = (np.flatnonzero(n_validation == 0) + case_min).tolist()
        raise RuntimeError(
            f"selection {name} has no validation pairs in cases {missing}"
        )

    sums = {
        column: _case_sum(case_index, local[column], n_cases)
        for column in (
            "validation_label_sum",
            "validation_prediction_sum",
            "validation_residual_sum",
            "validation_null_sum",
            "validation_residual_square_sum",
            "scene_prediction",
            "scene_abs_prediction",
            "n_pairs",
            "primary_mag",
            "primary_size",
            "total_neighbour_flux_ratio",
            "max_neighbour_flux_ratio",
            "min_pair_distance",
        )
    }
    label = validation_scale * sums["validation_label_sum"] / n_scene
    prediction = (
        validation_scale * sums["validation_prediction_sum"] / n_scene
    )
    residual = validation_scale * sums["validation_residual_sum"] / n_scene
    null = validation_scale * sums["validation_null_sum"] / n_scene
    if np.max(np.abs(label - prediction - residual)) > 2.0e-13:
        raise RuntimeError("scene label-prediction residual identity failed")

    pair_residual_case = sums["validation_residual_sum"] / n_validation
    pair_label_case = sums["validation_label_sum"] / n_validation
    pair_prediction_case = sums["validation_prediction_sum"] / n_validation
    contribution = (
        validation_scale
        * sums["validation_residual_sum"]
        / eligible_primary_count
    )
    case_table = pd.DataFrame({
        "selection": name,
        "case": np.arange(case_min, case_min + n_cases, dtype=np.int64),
        "n_scenes": n_scene,
        "n_validation_pairs": n_validation.astype(np.int64),
        "fraction_all_eligible": n_scene / eligible_primary_count,
        "label_per_selected_primary": label,
        "prediction_per_selected_primary": prediction,
        "residual_per_selected_primary": residual,
        "null_per_selected_primary": null,
        "residual_contribution_per_all_eligible_primary": contribution,
        "pair_residual_mean": pair_residual_case,
    })
    n_validation_total = int(local.validation_pair_count.sum())
    residual_square_total = float(local.validation_residual_square_sum.sum())
    payload = {
        "name": name,
        "rule": {
            "scene_prediction_gt": (
                None
                if response_threshold is None
                else float(response_threshold)
            ),
            "primary_mag_ge": (
                None if primary_mag_min is None else float(primary_mag_min)
            ),
            "labels_used_in_selection": False,
        },
        "n_scenes": int(len(local)),
        "n_validation_pairs": n_validation_total,
        "n_scenes_without_validation_pair": int(
            (local.validation_pair_count.to_numpy(np.int64) == 0).sum()
        ),
        "fraction_all_eligible": finite_stat(
            n_scene / eligible_primary_count
        ),
        "scene_scale": {
            "validation_label_per_selected_primary": finite_stat(label),
            "validation_prediction_per_selected_primary": finite_stat(
                prediction
            ),
            "label_minus_prediction_per_selected_primary": finite_stat(
                residual
            ),
            "null_per_selected_primary": finite_stat(null),
            "residual_contribution_per_all_eligible_primary": finite_stat(
                contribution
            ),
            "exact_full_prediction_per_selected_primary": finite_stat(
                sums["scene_prediction"] / n_scene
            ),
        },
        "pair_scale": {
            "pooled_validation_label_mean": float(
                local.validation_label_sum.sum() / n_validation_total
            ),
            "pooled_validation_prediction_mean": float(
                local.validation_prediction_sum.sum() / n_validation_total
            ),
            "pooled_validation_label_minus_prediction_mean": float(
                local.validation_residual_sum.sum() / n_validation_total
            ),
            "pooled_validation_residual_rms": float(
                np.sqrt(residual_square_total / n_validation_total)
            ),
            "case_balanced_validation_label_mean": finite_stat(
                pair_label_case
            ),
            "case_balanced_validation_prediction_mean": finite_stat(
                pair_prediction_case
            ),
            "case_balanced_validation_label_minus_prediction_mean": (
                finite_stat(pair_residual_case)
            ),
        },
        "scene_properties": {
            "primary_mag": finite_stat(sums["primary_mag"] / n_scene),
            "primary_size": finite_stat(sums["primary_size"] / n_scene),
            "n_supported_pairs": finite_stat(sums["n_pairs"] / n_scene),
            "scene_prediction": finite_stat(
                sums["scene_prediction"] / n_scene
            ),
            "scene_abs_prediction_sum": finite_stat(
                sums["scene_abs_prediction"] / n_scene
            ),
            "total_neighbour_flux_ratio": finite_stat(
                sums["total_neighbour_flux_ratio"] / n_scene
            ),
            "max_neighbour_flux_ratio": finite_stat(
                sums["max_neighbour_flux_ratio"] / n_scene
            ),
            "min_pair_distance": finite_stat(
                sums["min_pair_distance"] / n_scene
            ),
        },
    }
    return payload, case_table


def paired_contrast(
    case_table: pd.DataFrame,
    left: str,
    right: str,
    column: str,
) -> dict[str, float | int | str]:
    """Case-paired left-minus-right contrast for two stored selections."""
    ltab = case_table.loc[case_table.selection.eq(left), ["case", column]]
    rtab = case_table.loc[case_table.selection.eq(right), ["case", column]]
    joined = ltab.merge(
        rtab, on="case", how="outer", validate="one_to_one",
        suffixes=("_left", "_right"), indicator=True,
    )
    if not joined._merge.eq("both").all():
        raise RuntimeError("paired contrast case join failed")
    stat = finite_stat(
        joined[f"{column}_left"].to_numpy(float)
        - joined[f"{column}_right"].to_numpy(float)
    )
    return {"left": left, "right": right, "column": column, **stat}


def stream_scene_population(
    catalogue: Path,
    tag: str,
    case_min: int,
    case_max: int,
    shear: float,
    score_chunk: int,
    split: dict[str, Any],
) -> tuple[pd.DataFrame, np.ndarray, dict[str, Any]]:
    """Score all supported rows and aggregate exact held-out rows by scene."""
    blendemu_root = "/home/z/Zekang.Zhang/blendemu"
    if blendemu_root not in sys.path:
        sys.path.insert(0, blendemu_root)
    from blendemu.inference import BlendingPredictor

    predictor = BlendingPredictor.load(
        MODEL_DIR, tag=tag, conditions=COND, device="cpu"
    )
    raw_cuts, _, _ = predictor._select("regression")
    if len(raw_cuts) != len(CUT_NAMES):
        raise RuntimeError(f"unexpected regression cuts: {raw_cuts}")
    cuts = dict(zip(CUT_NAMES, raw_cuts))
    n_cases = case_max - case_min + 1
    primary_sets = [set() for _ in range(n_cases)]
    validation_mask = exact_validation_mask(
        split["n_selected_rows"], split["test_size"], split["random_state"]
    )
    partials: list[pd.DataFrame] = []

    selected_cursor = 0
    validation_rows = 0
    full_label_sum = 0.0
    full_label_square_sum = 0.0
    validation_label_sum = 0.0
    validation_label_square_sum = 0.0
    validation_prediction_sum = 0.0
    validation_squared_residual_sum = 0.0
    processed_batches = 0
    scanned_rows = 0
    previous_min = -1

    with ipc.open_file(str(catalogue)) as reader:
        missing = set(PAIR_COLUMNS).difference(reader.schema.names)
        if missing:
            raise KeyError(f"response catalogue lacks columns: {sorted(missing)}")
        for batch_index in range(reader.num_record_batches):
            frame = pa.Table.from_batches([reader.get_batch(batch_index)]).select(
                PAIR_COLUMNS
            ).to_pandas()
            if frame.empty:
                continue
            batch_min = int(frame.case.min())
            if batch_min < previous_min:
                raise RuntimeError("response catalogue cases are not ordered")
            previous_min = batch_min
            scanned_rows += len(frame)
            frame = frame.loc[frame.case.between(case_min, case_max)]
            if frame.empty:
                if batch_min > case_max:
                    break
                continue
            processed_batches += 1
            finite_label = np.isfinite(frame.delta_et1.to_numpy(float))
            primary_ok = (
                frame.r_input_p.between(
                    *cuts["r_input_p"], inclusive="neither"
                )
                & frame.Re_input_p.between(
                    *cuts["Re_input_p"], inclusive="neither"
                )
                & finite_label
            )
            eligible = frame.loc[primary_ok, ["case", "input_index"]]
            for case, local in eligible.groupby("case", sort=False):
                primary_sets[int(case) - case_min].update(
                    local.input_index.to_numpy(np.int64).tolist()
                )

            pair_ok = primary_ok.copy()
            for column in ("r_input_s", "Re_input_s", "distance"):
                pair_ok &= frame[column].between(
                    *cuts[column], inclusive="neither"
                )
            pairs = frame.loc[pair_ok].copy()
            if pairs.empty:
                continue
            n_pairs = len(pairs)
            next_cursor = selected_cursor + n_pairs
            if next_cursor > len(validation_mask):
                raise RuntimeError("selected pair stream exceeds stored row count")
            use = validation_mask[selected_cursor:next_cursor]
            selected_cursor = next_cursor
            if not np.isfinite(pairs[PAIR_FEATURES].to_numpy(float)).all():
                raise RuntimeError("supported pair has a non-finite model feature")
            prediction = score_pairs(predictor, pairs, score_chunk)
            label = pairs.delta_et1.to_numpy(np.float64) / shear
            full_label_sum += float(label.sum())
            full_label_square_sum += float(np.square(label).sum())
            if use.any():
                residual = label[use] - prediction[use]
                validation_rows += int(use.sum())
                validation_label_sum += float(label[use].sum())
                validation_label_square_sum += float(np.square(label[use]).sum())
                validation_prediction_sum += float(prediction[use].sum())
                validation_squared_residual_sum += float(
                    np.square(residual).sum()
                )
            partials.append(
                aggregate_pair_batch(pairs, prediction, use, shear)
            )
            if processed_batches % 100 == 0:
                print(
                    f"processed {processed_batches} selected batches; "
                    f"model rows={selected_cursor:,}; "
                    f"validation rows={validation_rows:,}",
                    flush=True,
                )

    if selected_cursor != split["n_selected_rows"]:
        raise RuntimeError(
            f"selected rows {selected_cursor:,} != metadata "
            f"{split['n_selected_rows']:,}"
        )
    if validation_rows != split["n_validation_rows"]:
        raise RuntimeError(
            f"validation rows {validation_rows:,} != metadata "
            f"{split['n_validation_rows']:,}"
        )
    eligible_count = np.asarray(
        [len(items) for items in primary_sets], dtype=np.int64
    )
    if np.any(eligible_count == 0):
        raise RuntimeError("one or more cases has no eligible primary")
    scenes = combine_scene_partials(partials)
    del partials
    scene_count = scenes.groupby("case", sort=True).size().reindex(
        np.arange(case_min, case_max + 1), fill_value=0
    ).to_numpy(np.int64)
    if not np.array_equal(scene_count, eligible_count):
        difference = eligible_count - scene_count
        raise RuntimeError(
            "not every eligible primary has a supported pair; "
            f"missing total={int(difference.sum())}, max/case={int(difference.max())}"
        )

    full_label_mean = full_label_sum / selected_cursor
    full_label_variance = (
        full_label_square_sum - full_label_sum**2 / selected_cursor
    ) / (selected_cursor - 1)
    if not np.isclose(
        full_label_mean, split["target_mean"], rtol=0.0, atol=2.0e-12
    ):
        raise RuntimeError("full label mean does not replay model metadata")
    if not np.isclose(
        np.sqrt(full_label_variance),
        split["target_std"],
        rtol=0.0,
        atol=2.0e-12,
    ):
        raise RuntimeError("full label standard deviation does not replay metadata")
    row_validation = row_validation_summary(
        validation_rows,
        validation_label_sum,
        validation_label_square_sum,
        validation_prediction_sum,
        validation_squared_residual_sum,
    )
    if round(row_validation["r2"], 6) != round(split["validation_r2"], 6):
        raise RuntimeError("validation R2 does not replay stored model value")

    metadata = {
        "processed_batches": int(processed_batches),
        "scanned_rows_through_last_batch": int(scanned_rows),
        "n_selected_model_rows": int(selected_cursor),
        "n_validation_rows": int(validation_rows),
        "n_supported_scenes": int(len(scenes)),
        "n_eligible_primaries": int(eligible_count.sum()),
        "all_eligible_primaries_have_supported_pairs": True,
        "row_validation": row_validation,
        "reproduction_checks": {
            "selected_row_count": True,
            "validation_row_count": True,
            "eligible_scene_key_coverage": True,
            "target_mean": True,
            "target_standard_deviation": True,
            "validation_r2_to_six_decimals": True,
        },
    }
    return scenes, eligible_count, metadata


def validate_reference(
    payload: dict[str, Any], reference_path: Path
) -> dict[str, float | bool]:
    """Replay the preceding exact-validation cumulative-response artifact."""
    with reference_path.open(encoding="utf-8") as handle:
        reference = json.load(handle)
    total = reference["profiles"]["primary_mag"]["total"]
    current = payload["selections_by_name"]["all_supported"]
    checks = {
        "n_selected_model_rows": (
            payload["metadata"]["n_selected_model_rows"]
            == reference["metadata"]["n_selected_model_rows"]
        ),
        "n_validation_rows": (
            payload["metadata"]["n_validation_rows"]
            == reference["metadata"]["n_validation_rows"]
        ),
        "n_eligible_primaries": (
            payload["metadata"]["n_eligible_primaries"]
            == reference["metadata"]["n_primaries"]
        ),
    }
    comparisons = {
        "label": (
            current["scene_scale"][
                "validation_label_per_selected_primary"
            ]["mean"],
            total["additive_label_per_primary"]["mean"],
        ),
        "prediction": (
            current["scene_scale"][
                "validation_prediction_per_selected_primary"
            ]["mean"],
            total["additive_prediction_per_primary"]["mean"],
        ),
        "residual": (
            current["scene_scale"][
                "label_minus_prediction_per_selected_primary"
            ]["mean"],
            total["additive_label_minus_prediction_per_primary"]["mean"],
        ),
        "null": (
            current["scene_scale"]["null_per_selected_primary"]["mean"],
            total["additive_null_per_primary"]["mean"],
        ),
    }
    maximum = 0.0
    for name, (observed, expected) in comparisons.items():
        delta = abs(float(observed) - float(expected))
        checks[f"{name}_mean"] = delta < 3.0e-13
        maximum = max(maximum, delta)
    if not all(checks.values()):
        raise RuntimeError(f"reference replay failed: {checks}")
    return {**checks, "maximum_absolute_mean_difference": maximum}


def build_payload(
    scenes: pd.DataFrame,
    eligible_count: np.ndarray,
    stream_metadata: dict[str, Any],
    *,
    catalogue: Path,
    model_metadata: Path,
    tag: str,
    case_min: int,
    case_max: int,
    shear: float,
    split: dict[str, Any],
    scene_catalogue: Path,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """Build named and sensitivity-grid summaries."""
    n_cases = case_max - case_min + 1
    scale = split["n_selected_rows"] / split["n_validation_rows"]
    specs: list[tuple[str, float | None, float | None]] = list(
        MAIN_SELECTIONS
    )
    specs.extend([
        ("anchor_tail_outside", None, None),
        ("anchor_tail_faint24_outside", None, 24.0),
    ])
    for threshold in GRID_RESPONSE_THRESHOLDS:
        for mag_min in GRID_PRIMARY_MAG_MINIMA:
            mag_tag = "allmag" if mag_min is None else f"m{mag_min:g}"
            specs.append((f"grid_r{threshold:g}_{mag_tag}", threshold, mag_min))

    selections: list[dict[str, Any]] = []
    case_tables: list[pd.DataFrame] = []
    seen: set[str] = set()
    for name, threshold, mag_min in specs:
        if name in seen:
            continue
        seen.add(name)
        keep = selection_mask(scenes, threshold, mag_min)
        if name == "anchor_tail_outside":
            keep = scenes.scene_prediction.to_numpy(float) <= 0.1
        elif name == "anchor_tail_faint24_outside":
            keep = (
                (scenes.scene_prediction.to_numpy(float) <= 0.1)
                & (scenes.primary_mag.to_numpy(float) >= 24.0)
            )
        summary, cases = summarize_selection(
            scenes,
            keep,
            name=name,
            response_threshold=threshold,
            primary_mag_min=mag_min,
            case_min=case_min,
            n_cases=n_cases,
            validation_scale=scale,
            eligible_primary_count=eligible_count,
        )
        if name.endswith("_outside"):
            summary["rule"] = {
                "scene_prediction_le": 0.1,
                "primary_mag_ge": 24.0 if "faint24" in name else None,
                "labels_used_in_selection": False,
            }
        selections.append(summary)
        case_tables.append(cases)

    selection_map = {item["name"]: item for item in selections}
    cases = pd.concat(case_tables, ignore_index=True)
    contrasts = [
        paired_contrast(
            cases,
            "anchor_tail",
            "anchor_tail_outside",
            "residual_per_selected_primary",
        ),
        paired_contrast(
            cases,
            "anchor_tail_faint24",
            "anchor_tail_faint24_outside",
            "residual_per_selected_primary",
        ),
    ]
    payload = {
        "title": (
            "V2.2 exact-validation pair residuals in anchor-tail-like scenes"
        ),
        "metadata": {
            "tag": tag,
            "catalogue": str(catalogue.resolve()),
            "model_metadata": str(model_metadata.resolve()),
            "scene_catalogue": str(scene_catalogue.resolve()),
            "case_window": [int(case_min), int(case_max)],
            "n_cases": int(n_cases),
            "shear": float(shear),
            "validation_fraction": float(
                split["n_validation_rows"] / split["n_selected_rows"]
            ),
            "inverse_probability_scale": float(scale),
            "split": {
                "method": "sklearn train_test_split on selected pair rows",
                "test_size": float(split["test_size"]),
                "random_state": int(split["random_state"]),
                "status": (
                    "exact internal row-random validation; rows share cases "
                    "and often primaries with training rows"
                ),
            },
            "uncertainty_unit": "rendered half-shear simulation case",
            "residual_sign": "label minus V2.2; positive means underprediction",
            "scene_selection": (
                "full-scene sum of frozen V2.2 supported-pair predictions; "
                "validation labels are not used"
            ),
            **stream_metadata,
        },
        "selections": selections,
        "selections_by_name": selection_map,
        "paired_contrasts": contrasts,
    }
    table_rows = []
    for item in selections:
        scene = item["scene_scale"]
        pair = item["pair_scale"]
        props = item["scene_properties"]
        table_rows.append({
            "selection": item["name"],
            "scene_prediction_gt": item["rule"].get("scene_prediction_gt"),
            "scene_prediction_le": item["rule"].get("scene_prediction_le"),
            "primary_mag_ge": item["rule"].get("primary_mag_ge"),
            "n_scenes": item["n_scenes"],
            "n_validation_pairs": item["n_validation_pairs"],
            "fraction_all_eligible": item["fraction_all_eligible"]["mean"],
            "label_per_primary": scene[
                "validation_label_per_selected_primary"
            ]["mean"],
            "label_per_primary_sem": scene[
                "validation_label_per_selected_primary"
            ]["case_sem"],
            "prediction_per_primary": scene[
                "validation_prediction_per_selected_primary"
            ]["mean"],
            "prediction_per_primary_sem": scene[
                "validation_prediction_per_selected_primary"
            ]["case_sem"],
            "residual_per_primary": scene[
                "label_minus_prediction_per_selected_primary"
            ]["mean"],
            "residual_per_primary_sem": scene[
                "label_minus_prediction_per_selected_primary"
            ]["case_sem"],
            "null_per_primary": scene["null_per_selected_primary"]["mean"],
            "null_per_primary_sem": scene[
                "null_per_selected_primary"
            ]["case_sem"],
            "residual_contribution_all": scene[
                "residual_contribution_per_all_eligible_primary"
            ]["mean"],
            "residual_contribution_all_sem": scene[
                "residual_contribution_per_all_eligible_primary"
            ]["case_sem"],
            "pair_residual_pooled": pair[
                "pooled_validation_label_minus_prediction_mean"
            ],
            "pair_residual_case_balanced": pair[
                "case_balanced_validation_label_minus_prediction_mean"
            ]["mean"],
            "pair_residual_case_sem": pair[
                "case_balanced_validation_label_minus_prediction_mean"
            ]["case_sem"],
            "primary_mag_mean": props["primary_mag"]["mean"],
            "n_pairs_mean": props["n_supported_pairs"]["mean"],
            "scene_prediction_mean": props["scene_prediction"]["mean"],
            "total_neighbour_flux_ratio_mean": props[
                "total_neighbour_flux_ratio"
            ]["mean"],
            "max_neighbour_flux_ratio_mean": props[
                "max_neighbour_flux_ratio"
            ]["mean"],
            "min_pair_distance_mean": props["min_pair_distance"]["mean"],
        })
    return payload, pd.DataFrame(table_rows), cases


def plot_results(payload: dict[str, Any], output_prefix: Path) -> None:
    """Plot the main selections and threshold sensitivity."""
    configure_style()
    names = [item[0] for item in MAIN_SELECTIONS]
    labels = [
        "All supported",
        "$R_{scene}>0.1$",
        "$R_{scene}>0.1$\n$r_p\geq24$",
        "$R_{scene}>0.1$\n$r_p\geq24.5$",
    ]
    selected = [payload["selections_by_name"][name] for name in names]
    x = np.arange(len(selected), dtype=float)
    label = np.asarray([
        item["scene_scale"]["validation_label_per_selected_primary"]["mean"]
        for item in selected
    ])
    label_sem = np.asarray([
        item["scene_scale"]["validation_label_per_selected_primary"][
            "case_sem"
        ]
        for item in selected
    ])
    prediction = np.asarray([
        item["scene_scale"][
            "validation_prediction_per_selected_primary"
        ]["mean"]
        for item in selected
    ])
    prediction_sem = np.asarray([
        item["scene_scale"][
            "validation_prediction_per_selected_primary"
        ]["case_sem"]
        for item in selected
    ])
    residual = label - prediction
    residual_sem = np.asarray([
        item["scene_scale"][
            "label_minus_prediction_per_selected_primary"
        ]["case_sem"]
        for item in selected
    ])
    null = np.asarray([
        item["scene_scale"]["null_per_selected_primary"]["mean"]
        for item in selected
    ])
    null_sem = np.asarray([
        item["scene_scale"]["null_per_selected_primary"]["case_sem"]
        for item in selected
    ])

    figure, axes = plt.subplots(1, 2, figsize=(9.2, 3.75))
    axes[0].errorbar(
        x - 0.08, label, yerr=label_sem, color="black", marker="o",
        linewidth=1.2, capsize=2.5, label="Validation label",
    )
    axes[0].errorbar(
        x + 0.08, prediction, yerr=prediction_sem, color="#0072B2",
        marker="s", markerfacecolor="white", linestyle="--",
        linewidth=1.2, capsize=2.5, label="V2.2",
    )
    axes[0].set_ylabel("Summed response per selected primary")
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].text(
        -0.10, 1.04, "A", transform=axes[0].transAxes,
        fontweight="bold", va="top",
    )

    axes[1].errorbar(
        x - 0.05, residual, yerr=residual_sem, color="#D55E00",
        marker="o", linewidth=1.2, capsize=2.5, label="Label $-$ V2.2",
    )
    axes[1].errorbar(
        x + 0.05, null, yerr=null_sem, color="#009E73", marker="^",
        markerfacecolor="white", linestyle="--", linewidth=1.1,
        capsize=2.5, label="Cross-component null",
    )
    axes[1].axhline(0.0, color="0.4", linestyle="--", linewidth=0.8)
    axes[1].set_ylabel("Residual response per selected primary")
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].text(
        -0.10, 1.04, "B", transform=axes[1].transAxes,
        fontweight="bold", va="top",
    )
    for axis in axes:
        axis.set_xticks(x, labels)
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(axis="x", labelsize=7.5)
    figure.suptitle(
        "V2.2 held-out pair residuals in anchor-tail-like half-shear scenes",
        fontsize=10.5,
    )
    figure.text(
        0.5, 0.925,
        "Exact row-random validation split; error bars are one SEM over 160 rendered cases",
        ha="center", fontsize=7.5,
    )
    figure.subplots_adjust(left=0.085, right=0.99, bottom=0.22, top=0.84, wspace=0.28)
    figure.savefig(f"{output_prefix}.pdf", bbox_inches="tight")
    figure.savefig(f"{output_prefix}.png", dpi=300, bbox_inches="tight")
    plt.close(figure)


def write_markdown(payload: dict[str, Any], output: Path) -> None:
    """Write a compact human-readable result report."""
    meta = payload["metadata"]
    rows = [
        "# V2.2 validation-pair residuals in anchor-tail-like scenes",
        "",
        "The direct anchor analogue groups all supported V2.2 pairs by primary "
        "and applies the same frozen rule `scene prediction > 0.1`. Validation "
        "labels do not enter the scene cut. Positive residual means the emulator "
        "underpredicts.",
        "",
        f"This uses `{meta['n_validation_rows']:,}` exact held-out pair rows in "
        f"`{meta['n_supported_scenes']:,}` primary scenes across "
        f"cases `{meta['case_window'][0]}--{meta['case_window'][1]}`. The split "
        "is internal and row-random: held-out rows share simulations and often "
        "primaries with training rows.",
        "",
        "| subset | scenes | validation pairs | scene fraction | label | V2.2 | label - V2.2 | raw pair residual |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    display = [
        ("all_supported", "All supported scenes"),
        ("anchor_tail", "R_scene > 0.1"),
        ("anchor_tail_faint24", "R_scene > 0.1, r_p >= 24"),
        ("anchor_tail_faint24p5", "R_scene > 0.1, r_p >= 24.5"),
        ("anchor_tail_outside", "R_scene <= 0.1"),
    ]
    for name, label in display:
        item = payload["selections_by_name"][name]
        scene = item["scene_scale"]
        pair = item["pair_scale"]
        rows.append(
            f"| {label} | {item['n_scenes']:,} | "
            f"{item['n_validation_pairs']:,} | "
            f"{100.0 * item['fraction_all_eligible']['mean']:.3f}% | "
            f"{scene['validation_label_per_selected_primary']['mean']:+.6f} +- "
            f"{scene['validation_label_per_selected_primary']['case_sem']:.6f} | "
            f"{scene['validation_prediction_per_selected_primary']['mean']:+.6f} +- "
            f"{scene['validation_prediction_per_selected_primary']['case_sem']:.6f} | "
            f"{scene['label_minus_prediction_per_selected_primary']['mean']:+.6f} +- "
            f"{scene['label_minus_prediction_per_selected_primary']['case_sem']:.6f} | "
            f"{pair['pooled_validation_label_minus_prediction_mean']:+.6f} |"
        )
    tail = payload["selections_by_name"]["anchor_tail"]
    faint = payload["selections_by_name"]["anchor_tail_faint24"]
    rows.extend([
        "",
        "## Tail scene properties",
        "",
        "| subset | mean r_p | mean pairs | mean R_scene | mean total F_s/F_p | mean max F_s/F_p | mean nearest distance |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for item, label in ((tail, "R_scene > 0.1"), (faint, "also r_p >= 24")):
        prop = item["scene_properties"]
        rows.append(
            f"| {label} | {prop['primary_mag']['mean']:.3f} | "
            f"{prop['n_supported_pairs']['mean']:.3f} | "
            f"{prop['scene_prediction']['mean']:.4f} | "
            f"{prop['total_neighbour_flux_ratio']['mean']:.3f} | "
            f"{prop['max_neighbour_flux_ratio']['mean']:.3f} | "
            f"{prop['min_pair_distance']['mean']:.3f} arcsec |"
        )
    contribution = tail["scene_scale"][
        "residual_contribution_per_all_eligible_primary"
    ]
    rows.extend([
        "",
        "## Interpretation guardrails",
        "",
        f"The `R_scene > 0.1` pairs contribute "
        f"`{contribution['mean']:+.7f} +- {contribution['case_sem']:.7f}` "
        "to the response residual averaged over all eligible primaries. This "
        "is the directly comparable additive budget; the raw pair-row mean is "
        "smaller because each primary contributes several pairs.",
        "",
        "These are per-neighbour half-shear labels grouped after measurement. "
        "They test whether the pair emulator has the same conditional residual "
        "in tail-like contexts; they are not coherent multi-neighbour scene "
        "measurements.",
        "",
    ])
    output.write_text("\n".join(rows), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalogue", required=True)
    parser.add_argument("--model-metadata", required=True)
    parser.add_argument("--reference-validation-json", required=True)
    parser.add_argument("--tag", default="lsst_r_extnbr_v22")
    parser.add_argument("--case-min", type=int, default=40)
    parser.add_argument("--case-max", type=int, default=199)
    parser.add_argument("--shear", type=float, default=0.2)
    parser.add_argument("--score-chunk", type=int, default=1_000_000)
    parser.add_argument("--scene-catalogue", required=True)
    parser.add_argument("--output-prefix", required=True)
    args = parser.parse_args()
    if args.case_min > args.case_max or args.shear <= 0 or args.score_chunk <= 0:
        raise ValueError("invalid case range, shear, or score chunk")
    catalogue = Path(args.catalogue)
    model_metadata = Path(args.model_metadata)
    reference = Path(args.reference_validation_json)
    for path in (catalogue, model_metadata, reference):
        if not path.is_file():
            raise FileNotFoundError(path)
    output_prefix = Path(args.output_prefix)
    scene_catalogue = Path(args.scene_catalogue)
    outputs = [
        Path(f"{output_prefix}.{suffix}")
        for suffix in ("csv", "cases.csv", "json", "md", "pdf", "png")
    ]
    outputs.append(scene_catalogue)
    existing = [str(path) for path in outputs if path.exists()]
    if existing:
        raise FileExistsError(f"refusing existing outputs: {existing}")
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    scene_catalogue.parent.mkdir(parents=True, exist_ok=True)

    split = load_split_spec(model_metadata, args.tag)
    scenes, eligible_count, stream_metadata = stream_scene_population(
        catalogue,
        args.tag,
        args.case_min,
        args.case_max,
        args.shear,
        args.score_chunk,
        split,
    )
    scenes.to_feather(scene_catalogue, compression="zstd")
    payload, table, cases = build_payload(
        scenes,
        eligible_count,
        stream_metadata,
        catalogue=catalogue,
        model_metadata=model_metadata,
        tag=args.tag,
        case_min=args.case_min,
        case_max=args.case_max,
        shear=args.shear,
        split=split,
        scene_catalogue=scene_catalogue,
    )
    payload["reference_replay"] = validate_reference(payload, reference)
    table.to_csv(f"{output_prefix}.csv", index=False)
    cases.to_csv(f"{output_prefix}.cases.csv", index=False)
    with open(f"{output_prefix}.json", "x", encoding="utf-8") as handle:
        json.dump(json_clean(payload), handle, indent=2, allow_nan=False)
        handle.write("\n")
    write_markdown(payload, Path(f"{output_prefix}.md"))
    plot_results(payload, output_prefix)
    print(
        f"saved V2.2 validation tail-scene residuals to {output_prefix}.*",
        flush=True,
    )


if __name__ == "__main__":
    main()
