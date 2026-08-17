#!/usr/bin/env python3
"""Plot bin-summed V2.2 labels and predictions on its exact validation rows.

V2.2 used sklearn's fixed 80/20 random row split after applying its pair cuts.
This script reconstructs that split exactly, scores only its validation rows,
and inverse-probability scales their additive sums to the full-pair response
scale.  The split is by pair row, not by primary or rendered case.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc
from sklearn.model_selection import train_test_split

from scripts.plot_v22_training_cumulative_response import (
    AXES,
    COND,
    CUT_NAMES,
    MODEL_DIR,
    PAIR_COLUMNS,
    PAIR_FEATURES,
    accumulate_axis,
    axis_values,
    configure_style,
    flatten_profiles,
    initialize_accumulators,
    json_clean,
    score_pairs,
    summarize_axis,
)


def exact_validation_mask(
    n_rows: int,
    test_size: float,
    random_state: int,
) -> np.ndarray:
    """Reproduce sklearn's validation membership for the frozen row order."""
    if n_rows < 2 or not 0.0 < test_size < 1.0:
        raise ValueError("invalid row count or validation fraction")
    indices = np.arange(n_rows, dtype=np.int64)
    train_index, validation_index = train_test_split(
        indices,
        test_size=test_size,
        random_state=random_state,
    )
    mask = np.zeros(n_rows, dtype=bool)
    mask[validation_index] = True
    if np.any(mask[train_index]) or int(mask.sum()) != len(validation_index):
        raise RuntimeError("training and validation indices are not disjoint")
    del indices, train_index, validation_index
    return mask


def row_validation_summary(
    n_rows: int,
    label_sum: float,
    label_square_sum: float,
    prediction_sum: float,
    squared_residual_sum: float,
) -> dict[str, float | int]:
    """Return pooled row means and R2 from streaming sufficient statistics."""
    if n_rows < 2:
        raise ValueError("row validation summary needs at least two rows")
    values = np.asarray(
        [label_sum, label_square_sum, prediction_sum, squared_residual_sum],
        dtype=float,
    )
    if not np.isfinite(values).all() or squared_residual_sum < 0.0:
        raise ValueError("invalid row-validation sufficient statistics")
    centered_square_sum = label_square_sum - label_sum**2 / n_rows
    if centered_square_sum <= 0.0:
        raise ValueError("validation labels have no positive variance")
    return {
        "n_rows": int(n_rows),
        "label_mean": float(label_sum / n_rows),
        "prediction_mean": float(prediction_sum / n_rows),
        "label_minus_prediction_mean": float(
            (label_sum - prediction_sum) / n_rows
        ),
        "r2": float(1.0 - squared_residual_sum / centered_square_sum),
    }


def load_split_spec(metadata_path: Path, tag: str) -> dict[str, Any]:
    with metadata_path.open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    if metadata.get("tag") != tag:
        raise RuntimeError(
            f"model metadata tag {metadata.get('tag')!r} does not match {tag!r}"
        )
    task = metadata["tasks"]["regression"]
    training = metadata["training"]
    metrics = task["metrics"]
    spec = {
        "test_size": float(training["test_size"]),
        "random_state": int(training["random_state"]),
        "n_training_rows": int(metrics["train_rows"]),
        "n_validation_rows": int(metrics["validation_rows"]),
        "validation_r2": float(metrics["best_score"]),
        "target_mean": float(task["standardization"]["mean"]),
        "target_std": float(task["standardization"]["std"]),
        "model_metadata": str(metadata_path.resolve()),
    }
    spec["n_selected_rows"] = (
        spec["n_training_rows"] + spec["n_validation_rows"]
    )
    expected_validation = int(
        np.ceil(spec["test_size"] * spec["n_selected_rows"])
    )
    if expected_validation != spec["n_validation_rows"]:
        raise RuntimeError(
            "stored validation count does not match sklearn test-size rounding"
        )
    return spec


def stream_validation_population(
    catalogue: Path,
    tag: str,
    case_min: int,
    case_max: int,
    shear: float,
    score_chunk: int,
    split: dict[str, Any],
) -> dict[str, Any]:
    """Accumulate exact validation rows in model-training row order."""
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
    accumulators = initialize_accumulators(n_cases)
    primaries = [set() for _ in range(n_cases)]
    validation_mask = exact_validation_mask(
        split["n_selected_rows"],
        split["test_size"],
        split["random_state"],
    )
    scale = split["n_selected_rows"] / split["n_validation_rows"]

    previous_min = -1
    processed_batches = 0
    scanned_rows = 0
    selected_cursor = 0
    validation_rows = 0
    full_label_sum = 0.0
    full_label_square_sum = 0.0
    validation_label_sum = 0.0
    validation_label_square_sum = 0.0
    validation_prediction_sum = 0.0
    validation_squared_residual_sum = 0.0

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
                primaries[int(case) - case_min].update(
                    local.input_index.to_numpy(np.int64).tolist()
                )

            pair_ok = primary_ok.copy()
            for column in ("r_input_s", "Re_input_s", "distance"):
                pair_ok &= frame[column].between(
                    *cuts[column], inclusive="neither"
                )
            model_pairs = frame.loc[pair_ok].copy()
            n_model_pairs = len(model_pairs)
            if n_model_pairs == 0:
                continue
            next_cursor = selected_cursor + n_model_pairs
            if next_cursor > len(validation_mask):
                raise RuntimeError("selected pair stream exceeds stored model row count")
            label_all = model_pairs.delta_et1.to_numpy(np.float64) / shear
            full_label_sum += float(label_all.sum())
            full_label_square_sum += float(np.square(label_all).sum())
            use = validation_mask[selected_cursor:next_cursor]
            selected_cursor = next_cursor
            pairs = model_pairs.loc[use].copy()
            if pairs.empty:
                continue
            if not np.isfinite(pairs[PAIR_FEATURES].to_numpy(float)).all():
                raise RuntimeError("validation pair has a non-finite model feature")
            null = pairs.delta_et2.to_numpy(np.float64) / shear
            if not np.isfinite(null).all():
                raise RuntimeError("validation pair has a non-finite null label")
            prediction = score_pairs(predictor, pairs, score_chunk)
            label = pairs.delta_et1.to_numpy(np.float64) / shear
            validation_rows += len(pairs)
            validation_label_sum += float(label.sum())
            validation_label_square_sum += float(np.square(label).sum())
            validation_prediction_sum += float(prediction.sum())
            validation_squared_residual_sum += float(
                np.square(label - prediction).sum()
            )
            case_index = pairs.case.to_numpy(np.int64) - case_min
            for spec in AXES:
                values = axis_values(pairs, spec["name"])
                accumulate_axis(
                    accumulators[spec["name"]],
                    values,
                    spec["edges"],
                    case_index,
                    scale * label,
                    scale * prediction,
                    scale * null,
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
    n_primary = np.asarray([len(item) for item in primaries], dtype=np.int64)
    if np.any(n_primary == 0):
        missing_cases = (np.flatnonzero(n_primary == 0) + case_min).tolist()
        raise RuntimeError(f"cases without eligible primaries: {missing_cases[:10]}")
    profiles = {
        spec["name"]: summarize_axis(
            accumulators[spec["name"]], n_primary, spec["edges"]
        )
        for spec in AXES
    }
    reference_name = AXES[0]["name"]
    reference = accumulators[reference_name]
    for spec in AXES[1:]:
        candidate = accumulators[spec["name"]]
        if not np.array_equal(
            candidate["counts"].sum(axis=1),
            reference["counts"].sum(axis=1),
        ):
            raise RuntimeError(f"{spec['name']} does not replay validation counts")
        for field in ("label_sum", "prediction_sum", "null_sum"):
            difference = (
                candidate[field].sum(axis=1)
                - reference[field].sum(axis=1)
            )
            if np.max(np.abs(difference)) > 5.0e-11:
                raise RuntimeError(f"{spec['name']} does not replay {field}")

    full_label_mean = full_label_sum / selected_cursor
    full_label_variance = (
        full_label_square_sum - full_label_sum**2 / selected_cursor
    ) / (selected_cursor - 1)
    if not np.isclose(
        full_label_mean, split["target_mean"], rtol=0.0, atol=2.0e-12
    ):
        raise RuntimeError("full selected label mean does not replay metadata")
    if not np.isclose(
        np.sqrt(full_label_variance),
        split["target_std"],
        rtol=0.0,
        atol=2.0e-12,
    ):
        raise RuntimeError("full selected label standard deviation does not replay metadata")
    row_validation = row_validation_summary(
        validation_rows,
        validation_label_sum,
        validation_label_square_sum,
        validation_prediction_sum,
        validation_squared_residual_sum,
    )
    if round(row_validation["r2"], 6) != round(split["validation_r2"], 6):
        raise RuntimeError(
            f"validation R2 {row_validation['r2']:.9f} does not replay "
            f"stored {split['validation_r2']:.9f}"
        )
    checks = {
        "selected_row_count": True,
        "validation_row_count": True,
        "target_mean": True,
        "target_standard_deviation": True,
        "validation_r2_to_six_decimals": True,
        "three_axis_pair_counts": True,
        "three_axis_additive_sums": True,
    }
    metadata = {
        "tag": tag,
        "catalogue": str(catalogue.resolve()),
        "case_window": [int(case_min), int(case_max)],
        "shear": float(shear),
        "v21_primary_domain": False,
        "regression_cuts": [[float(value) for value in cut] for cut in raw_cuts],
        "n_cases": int(n_cases),
        "n_primaries": int(n_primary.sum()),
        "n_selected_model_rows": int(selected_cursor),
        "n_validation_rows": int(validation_rows),
        "validation_fraction": float(validation_rows / selected_cursor),
        "inverse_probability_scale": float(scale),
        "processed_batches": int(processed_batches),
        "scanned_rows_through_last_batch": int(scanned_rows),
        "evaluation_status": (
            "in-sample-case audit on V2.2's exact row-random validation split; "
            "validation rows were not used to fit tree values but share cases "
            "and often primaries with training rows"
        ),
        "uncertainty_unit": "rendered half-shear simulation case",
        "normalization": (
            "within each case, sum validation-pair responses in the bin, "
            "multiply by N_all_model_rows/N_validation_rows, and divide by all "
            "V2.2-eligible primaries; bins add exactly to the validation-estimated "
            "full-pair response"
        ),
    }
    return {
        "metadata": metadata,
        "split": split,
        "row_validation": row_validation,
        "reproduction_checks": checks,
        "profiles": profiles,
    }


def plot_profiles(payload: dict[str, Any], output_prefix: Path) -> None:
    configure_style()
    figure, axes = plt.subplots(
        2,
        3,
        figsize=(10.8, 5.7),
        sharex="col",
        gridspec_kw={"height_ratios": [2.0, 1.0]},
    )
    for column, spec in enumerate(AXES):
        bins = payload["profiles"][spec["name"]]["bins"]
        x = np.asarray([item["feature_mean"] for item in bins], dtype=float)
        label = np.asarray([
            item["additive_label_per_primary"]["mean"] for item in bins
        ])
        label_sem = np.asarray([
            item["additive_label_per_primary"]["case_sem"] for item in bins
        ])
        prediction = np.asarray([
            item["additive_prediction_per_primary"]["mean"] for item in bins
        ])
        prediction_sem = np.asarray([
            item["additive_prediction_per_primary"]["case_sem"] for item in bins
        ])
        residual = label - prediction
        residual_sem = np.asarray([
            item["additive_label_minus_prediction_per_primary"]["case_sem"]
            for item in bins
        ])
        top = axes[0, column]
        bottom = axes[1, column]
        top.plot(
            x, label, color="#000000", marker="o", markersize=3.1,
            linewidth=1.15, label="Validation label",
        )
        top.fill_between(
            x, label - label_sem, label + label_sem,
            color="#000000", alpha=0.11, linewidth=0.0,
        )
        top.plot(
            x, prediction, color="#0072B2", marker="s",
            markerfacecolor="white", markersize=3.0, linestyle="--",
            linewidth=1.15, label="V2.2 emulator",
        )
        top.fill_between(
            x, prediction - prediction_sem, prediction + prediction_sem,
            color="#0072B2", alpha=0.13, linewidth=0.0,
        )
        bottom.plot(
            x, residual, color="#D55E00", marker="o", markersize=3.0,
            linewidth=1.15,
        )
        bottom.fill_between(
            x, residual - residual_sem, residual + residual_sem,
            color="#D55E00", alpha=0.16, linewidth=0.0,
        )
        top.axhline(0.0, color="0.60", linestyle=":", linewidth=0.7)
        bottom.axhline(0.0, color="0.40", linestyle="--", linewidth=0.75)
        top.set_title(spec["label"])
        bottom.set_xlabel(spec["label"])
        top.spines[["top", "right"]].set_visible(False)
        bottom.spines[["top", "right"]].set_visible(False)
        top.text(
            -0.13, 1.04, chr(ord("A") + column), transform=top.transAxes,
            fontweight="bold", fontsize=10, va="top",
        )
    axes[0, 0].set_ylabel(
        "Validation-estimated response\nper eligible primary"
    )
    axes[1, 0].set_ylabel("Label $-$ V2.2\nper eligible primary")
    for column in (1, 2):
        axes[0, column].set_ylabel("")
        axes[1, column].set_ylabel("")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.916),
        ncol=2, frameon=False,
    )
    total = payload["profiles"]["primary_mag"]["total"]
    label_total = total["additive_label_per_primary"]
    prediction_total = total["additive_prediction_per_primary"]
    residual_total = total["additive_label_minus_prediction_per_primary"]
    metadata = payload["metadata"]
    figure.suptitle(
        "V2.2 cumulative pair response on its exact validation rows",
        fontsize=11.0,
        y=0.992,
    )
    figure.text(
        0.5,
        0.951,
        (
            f"c40–199; {metadata['n_validation_rows']:,} validation / "
            f"{metadata['n_selected_model_rows']:,} model rows; "
            f"validation sums scaled ×{metadata['inverse_probability_scale']:.3f}; "
            "bands are one case SEM"
        ),
        ha="center",
        va="top",
        fontsize=8.0,
    )
    figure.text(
        0.5,
        0.882,
        (
            f"Sum over bins: label {label_total['mean']:+.6f}; "
            f"V2.2 {prediction_total['mean']:+.6f}; "
            f"label − V2.2 {residual_total['mean']:+.6f} "
            f"± {residual_total['case_sem']:.6f}"
        ),
        ha="center",
        va="top",
        fontsize=7.5,
    )
    figure.subplots_adjust(
        left=0.075, right=0.99, bottom=0.09, top=0.83,
        hspace=0.12, wspace=0.28,
    )
    figure.savefig(f"{output_prefix}.pdf", bbox_inches="tight")
    figure.savefig(f"{output_prefix}.png", dpi=300, bbox_inches="tight")
    plt.close(figure)


def write_markdown(payload: dict[str, Any], output: Path) -> None:
    metadata = payload["metadata"]
    total = payload["profiles"]["primary_mag"]["total"]
    row = payload["row_validation"]
    lines = [
        "# V2.2 cumulative response on its exact validation rows",
        "",
        f"The fixed V2.2 split contributes `{metadata['n_validation_rows']:,}` "
        f"validation rows out of `{metadata['n_selected_model_rows']:,}` model "
        f"rows across cases `{metadata['case_window'][0]}--"
        f"{metadata['case_window'][1]}`. The split uses sklearn "
        f"`train_test_split(test_size={payload['split']['test_size']}, "
        f"random_state={payload['split']['random_state']})`.",
        "",
        "Each bin sums only validation-row responses, multiplies by the inverse "
        f"sampling fraction (`{metadata['inverse_probability_scale']:.8f}`), "
        "and divides by all V2.2-eligible primaries in the case. Thus its scale "
        "is directly comparable with the full-pair response and bins add exactly.",
        "",
        f"- Validation-estimated label sum: "
        f"`{total['additive_label_per_primary']['mean']:+.8f} +- "
        f"{total['additive_label_per_primary']['case_sem']:.8f}`.",
        f"- Validation-estimated V2.2 sum: "
        f"`{total['additive_prediction_per_primary']['mean']:+.8f} +- "
        f"{total['additive_prediction_per_primary']['case_sem']:.8f}`.",
        f"- Label minus V2.2: "
        f"`{total['additive_label_minus_prediction_per_primary']['mean']:+.8f} "
        f"+- {total['additive_label_minus_prediction_per_primary']['case_sem']:.8f}`.",
        f"- Pooled validation-row R2: `{row['r2']:.9f}` (stored model value "
        f"`{payload['split']['validation_r2']:.6f}`).",
        "",
        "## Largest absolute additive residual bin on each axis",
        "",
        "| feature | interval | label - V2.2 per primary | pair fraction |",
        "|---|---:|---:|---:|",
    ]
    for spec in AXES:
        item = max(
            payload["profiles"][spec["name"]]["bins"],
            key=lambda value: abs(
                value["additive_label_minus_prediction_per_primary"]["mean"]
            ),
        )
        residual = item["additive_label_minus_prediction_per_primary"]
        lines.append(
            f"| {spec['label']} | [{item['lower']:.3f}, {item['upper']:.3f}) | "
            f"{residual['mean']:+.7f} +- {residual['case_sem']:.7f} | "
            f"{100.0 * item['pair_fraction']:.3f}% |"
        )
    lines.extend([
        "",
        "This is the model's internal random pair-row validation split. It is "
        "not a case-held-out test: validation pairs share rendered cases and "
        "often primaries with training pairs. Coherent-anchor and ConstGold "
        "labels were not read.",
        "",
    ])
    output.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalogue", required=True)
    parser.add_argument("--model-metadata", required=True)
    parser.add_argument("--tag", default="lsst_r_extnbr_v22")
    parser.add_argument("--case-min", type=int, default=40)
    parser.add_argument("--case-max", type=int, default=199)
    parser.add_argument("--shear", type=float, default=0.2)
    parser.add_argument("--score-chunk", type=int, default=1_000_000)
    parser.add_argument("--output-prefix", required=True)
    args = parser.parse_args()
    if args.case_min > args.case_max or args.shear <= 0.0 or args.score_chunk <= 0:
        raise ValueError("invalid case window, shear, or score chunk")
    catalogue = Path(args.catalogue)
    model_metadata = Path(args.model_metadata)
    for path in (catalogue, model_metadata):
        if not path.is_file():
            raise FileNotFoundError(path)
    output_prefix = Path(args.output_prefix)
    outputs = [Path(f"{output_prefix}.{suffix}") for suffix in (
        "csv", "json", "md", "pdf", "png",
    )]
    existing = [str(path) for path in outputs if path.exists()]
    if existing:
        raise FileExistsError(f"refusing existing outputs: {existing}")
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    split = load_split_spec(model_metadata, args.tag)
    payload = stream_validation_population(
        catalogue,
        args.tag,
        args.case_min,
        args.case_max,
        args.shear,
        args.score_chunk,
        split,
    )
    table = flatten_profiles(payload)
    table.insert(0, "population", "exact_validation_rows")
    table.insert(
        1,
        "inverse_probability_scale",
        payload["metadata"]["inverse_probability_scale"],
    )
    table.to_csv(f"{output_prefix}.csv", index=False)
    with open(f"{output_prefix}.json", "x", encoding="utf-8") as handle:
        json.dump(json_clean(payload), handle, indent=2, allow_nan=False)
        handle.write("\n")
    write_markdown(payload, Path(f"{output_prefix}.md"))
    plot_profiles(payload, output_prefix)
    print(
        f"saved cumulative V2.2 validation response to {output_prefix}.*",
        flush=True,
    )


if __name__ == "__main__":
    main()
