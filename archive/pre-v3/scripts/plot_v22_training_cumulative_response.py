#!/usr/bin/env python3
"""Plot bin-summed V2.2 training labels and predictions versus pair properties."""

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

from sbs_shear.domain import in_domain


BLENDEMU_ROOT = "/home/z/Zekang.Zhang/blendemu"
MODEL_DIR = os.path.join(BLENDEMU_ROOT, "models")
if BLENDEMU_ROOT not in sys.path:
    sys.path.insert(0, BLENDEMU_ROOT)

COND = dict(
    pixel_size=0.2,
    zero_point=30.0,
    psf_fwhm=0.73,
    moffat_beta=2.224,
    pixel_rms=0.312,
)
PAIR_FEATURES = [
    "Re_input_p",
    "Re_input_s",
    "r_input_p",
    "r_input_s",
    "sersic_n_input_p",
    "sersic_n_input_s",
    "distance",
]
PAIR_COLUMNS = [
    "case",
    "input_index",
    "delta_et1",
    "delta_et2",
    *PAIR_FEATURES,
]
CUT_NAMES = [
    "r_input_s",
    "r_input_p",
    "Re_input_s",
    "Re_input_p",
    "distance",
]
AXES = [
    {
        "name": "primary_mag",
        "label": r"Primary magnitude $r_p$",
        "edges": np.linspace(18.0, 25.8, 27),
    },
    {
        "name": "secondary_mag",
        "label": r"Secondary magnitude $r_s$",
        "edges": np.linspace(13.0, 29.0, 33),
    },
    {
        "name": "log10_flux_ratio",
        "label": r"$\log_{10}(F_s/F_p)$",
        "edges": np.linspace(-4.4, 5.2, 33),
    },
]


def finite_stat(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or values.size < 2 or not np.isfinite(values).all():
        raise ValueError("finite_stat requires at least two finite case values")
    sd = float(values.std(ddof=1))
    return {
        "mean": float(values.mean()),
        "case_sd": sd,
        "case_sem": float(sd / np.sqrt(values.size)),
        "n_cases": int(values.size),
    }


def score_pairs(predictor: Any, frame: pd.DataFrame, chunk: int) -> np.ndarray:
    prediction = np.empty(len(frame), dtype=np.float64)
    for lo in range(0, len(frame), chunk):
        hi = min(lo + chunk, len(frame))
        prediction[lo:hi] = predictor.predict_on_pairs(
            frame.iloc[lo:hi][PAIR_FEATURES], task="response"
        )["response"].to_numpy(float)
    if not np.isfinite(prediction).all():
        raise RuntimeError("V2.2 returned a non-finite pair prediction")
    return prediction


def initialize_accumulators(
    n_cases: int,
) -> dict[str, dict[str, np.ndarray]]:
    output = {}
    for spec in AXES:
        n_bins = len(spec["edges"]) - 1
        output[spec["name"]] = {
            "counts": np.zeros((n_cases, n_bins), dtype=np.int64),
            "label_sum": np.zeros((n_cases, n_bins), dtype=np.float64),
            "prediction_sum": np.zeros((n_cases, n_bins), dtype=np.float64),
            "null_sum": np.zeros((n_cases, n_bins), dtype=np.float64),
            "value_sum": np.zeros((n_cases, n_bins), dtype=np.float64),
        }
    return output


def axis_values(pairs: pd.DataFrame, name: str) -> np.ndarray:
    if name == "primary_mag":
        return pairs.r_input_p.to_numpy(float)
    if name == "secondary_mag":
        return pairs.r_input_s.to_numpy(float)
    if name == "log10_flux_ratio":
        return -0.4 * (
            pairs.r_input_s.to_numpy(float)
            - pairs.r_input_p.to_numpy(float)
        )
    raise KeyError(name)


def accumulate_axis(
    accumulator: dict[str, np.ndarray],
    values: np.ndarray,
    edges: np.ndarray,
    case_index: np.ndarray,
    label: np.ndarray,
    prediction: np.ndarray,
    null: np.ndarray,
) -> None:
    n_cases, n_bins = accumulator["counts"].shape
    bins = np.searchsorted(edges, values, side="right") - 1
    bad = (bins < 0) | (bins >= n_bins)
    if bad.any():
        raise RuntimeError(
            f"{int(bad.sum())} supported pairs fall outside plot edges; "
            f"examples={values[bad][:5]}"
        )
    flat = case_index * n_bins + bins
    if np.any(case_index < 0) or np.any(case_index >= n_cases):
        raise RuntimeError("case index outside accumulator")
    minlength = n_cases * n_bins
    accumulator["counts"] += np.bincount(
        flat, minlength=minlength
    ).reshape(n_cases, n_bins)
    for field, weights in (
        ("label_sum", label),
        ("prediction_sum", prediction),
        ("null_sum", null),
        ("value_sum", values),
    ):
        accumulator[field] += np.bincount(
            flat, weights=weights, minlength=minlength
        ).reshape(n_cases, n_bins)


def summarize_axis(
    accumulator: dict[str, np.ndarray],
    n_primary: np.ndarray,
    edges: np.ndarray,
) -> dict[str, Any]:
    counts = np.asarray(accumulator["counts"], dtype=np.int64)
    label_sum = np.asarray(accumulator["label_sum"], dtype=float)
    prediction_sum = np.asarray(accumulator["prediction_sum"], dtype=float)
    null_sum = np.asarray(accumulator["null_sum"], dtype=float)
    value_sum = np.asarray(accumulator["value_sum"], dtype=float)
    n_primary = np.asarray(n_primary, dtype=np.int64)
    if counts.ndim != 2 or counts.shape[1] != len(edges) - 1:
        raise ValueError("accumulator shape does not match bin edges")
    if n_primary.shape != (counts.shape[0],) or np.any(n_primary <= 0):
        raise ValueError("invalid eligible-primary counts")
    for array in (label_sum, prediction_sum, null_sum, value_sum):
        if array.shape != counts.shape or not np.isfinite(array).all():
            raise ValueError("invalid floating accumulator")

    bins = []
    for index in range(counts.shape[1]):
        count = counts[:, index]
        if count.sum() == 0:
            continue
        additive_label = label_sum[:, index] / n_primary
        additive_prediction = prediction_sum[:, index] / n_primary
        additive_residual = additive_label - additive_prediction
        additive_null = null_sum[:, index] / n_primary
        bins.append({
            "bin": int(index),
            "lower": float(edges[index]),
            "upper": float(edges[index + 1]),
            "feature_mean": float(value_sum[:, index].sum() / count.sum()),
            "n_pairs": int(count.sum()),
            "pair_fraction": float(count.sum() / counts.sum()),
            "n_cases_with_pairs": int((count > 0).sum()),
            "additive_label_per_primary": finite_stat(additive_label),
            "additive_prediction_per_primary": finite_stat(additive_prediction),
            "additive_label_minus_prediction_per_primary": finite_stat(
                additive_residual
            ),
            "additive_null_per_primary": finite_stat(additive_null),
        })

    total_label = label_sum.sum(axis=1) / n_primary
    total_prediction = prediction_sum.sum(axis=1) / n_primary
    total_residual = total_label - total_prediction
    total_null = null_sum.sum(axis=1) / n_primary
    binned_label = sum(
        item["additive_label_per_primary"]["mean"] for item in bins
    )
    binned_prediction = sum(
        item["additive_prediction_per_primary"]["mean"] for item in bins
    )
    binned_residual = sum(
        item["additive_label_minus_prediction_per_primary"]["mean"]
        for item in bins
    )
    total = {
        "n_cases": int(counts.shape[0]),
        "n_pairs": int(counts.sum()),
        "n_primaries": int(n_primary.sum()),
        "mean_pairs_per_primary": float(counts.sum() / n_primary.sum()),
        "additive_label_per_primary": finite_stat(total_label),
        "additive_prediction_per_primary": finite_stat(total_prediction),
        "additive_label_minus_prediction_per_primary": finite_stat(
            total_residual
        ),
        "additive_null_per_primary": finite_stat(total_null),
        "maximum_case_partition_error": float(max(
            np.max(np.abs(label_sum.sum(axis=1) / n_primary - total_label)),
            np.max(np.abs(
                prediction_sum.sum(axis=1) / n_primary - total_prediction
            )),
        )),
        "mean_bin_sum_closure": {
            "label": float(binned_label - total_label.mean()),
            "prediction": float(binned_prediction - total_prediction.mean()),
            "label_minus_prediction": float(
                binned_residual - total_residual.mean()
            ),
        },
    }
    for value in total["mean_bin_sum_closure"].values():
        if abs(value) > 2.0e-14:
            raise RuntimeError("bin means do not close to the global response")
    return {"edges": edges.tolist(), "bins": bins, "total": total}


def stream_training_population(
    catalogue: Path,
    tag: str,
    case_min: int,
    case_max: int,
    shear: float,
    score_chunk: int,
) -> tuple[dict[str, Any], list[list[float]]]:
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
    previous_min = -1
    processed_batches = 0
    scanned_rows = 0

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
            finite_label = np.isfinite(
                frame[["delta_et1", "delta_et2"]].to_numpy(float)
            ).all(axis=1)
            frame = frame.loc[finite_label]
            primary_ok = (
                frame.r_input_p.between(
                    *cuts["r_input_p"], inclusive="neither"
                )
                & frame.Re_input_p.between(
                    *cuts["Re_input_p"], inclusive="neither"
                )
            )
            primary_ok &= in_domain(
                frame.r_input_p.to_numpy(float),
                frame.Re_input_p.to_numpy(float),
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
            pairs = frame.loc[pair_ok].copy()
            if pairs.empty:
                continue
            if not np.isfinite(pairs[PAIR_FEATURES].to_numpy(float)).all():
                raise RuntimeError("supported pair has a non-finite model feature")
            prediction = score_pairs(predictor, pairs, score_chunk)
            label = pairs.delta_et1.to_numpy(float) / shear
            null = pairs.delta_et2.to_numpy(float) / shear
            case_index = pairs.case.to_numpy(np.int64) - case_min
            for spec in AXES:
                values = axis_values(pairs, spec["name"])
                accumulate_axis(
                    accumulators[spec["name"]],
                    values,
                    spec["edges"],
                    case_index,
                    label,
                    prediction,
                    null,
                )
            if processed_batches % 100 == 0:
                first = accumulators[AXES[0]["name"]]["counts"].sum()
                print(
                    f"processed {processed_batches} selected batches; "
                    f"active pairs={first:,}",
                    flush=True,
                )

    n_primary = np.asarray([len(item) for item in primaries], dtype=np.int64)
    if np.any(n_primary == 0):
        missing_cases = (np.flatnonzero(n_primary == 0) + case_min).tolist()
        raise RuntimeError(f"cases without eligible primaries: {missing_cases[:10]}")
    profiles = {}
    for spec in AXES:
        profiles[spec["name"]] = summarize_axis(
            accumulators[spec["name"]], n_primary, spec["edges"]
        )
    reference_name = AXES[0]["name"]
    reference_counts = accumulators[reference_name]["counts"]
    for spec in AXES[1:]:
        name = spec["name"]
        if not np.array_equal(accumulators[name]["counts"].sum(axis=1),
                              reference_counts.sum(axis=1)):
            raise RuntimeError(f"{name} pair counts do not replay primary-mag counts")
        for field in ("label_sum", "prediction_sum", "null_sum"):
            difference = (
                accumulators[name][field].sum(axis=1)
                - accumulators[reference_name][field].sum(axis=1)
            )
            if np.max(np.abs(difference)) > 2.0e-11:
                raise RuntimeError(f"{name} does not replay {field}")
    metadata = {
        "tag": tag,
        "catalogue": str(catalogue.resolve()),
        "case_window": [int(case_min), int(case_max)],
        "shear": float(shear),
        "v21_primary_domain": True,
        "regression_cuts": [[float(value) for value in cut] for cut in raw_cuts],
        "n_cases": int(n_cases),
        "n_primaries": int(n_primary.sum()),
        "n_pairs": int(reference_counts.sum()),
        "processed_batches": int(processed_batches),
        "scanned_rows_through_last_batch": int(scanned_rows),
        "uncertainty_unit": "rendered half-shear simulation case",
        "evaluation_status": "in-sample audit on V2.2 half-shear training cases",
        "normalization": (
            "within each case, sum pair responses in the bin and divide by "
            "all eligible primaries; bin contributions therefore add exactly "
            "to the global summed response"
        ),
    }
    return {"metadata": metadata, "profiles": profiles}, raw_cuts


def validate_reference(payload: dict[str, Any], reference_path: Path) -> dict[str, Any]:
    with reference_path.open(encoding="utf-8") as handle:
        reference = json.load(handle)
    summary = reference["summary"]
    total = payload["profiles"]["primary_mag"]["total"]
    checks = {
        "case_window": payload["metadata"]["case_window"] == reference["case_window"],
        "v21_primary_domain": bool(reference["v21_domain"]),
        "n_primaries": total["n_primaries"] == summary["n_primaries"],
        "n_pairs": total["n_pairs"] == summary["n_pairs"],
        "label_sum_mean": bool(np.isclose(
            total["additive_label_per_primary"]["mean"],
            summary["label_sum_mean"], rtol=0.0, atol=2.0e-10,
        )),
        "prediction_sum_mean": bool(np.isclose(
            total["additive_prediction_per_primary"]["mean"],
            summary["prediction_sum_mean"], rtol=0.0, atol=2.0e-8,
        )),
        "label_minus_prediction": bool(np.isclose(
            total["additive_label_minus_prediction_per_primary"]["mean"],
            -summary["prediction_minus_label"], rtol=0.0, atol=2.0e-8,
        )),
    }
    if not all(checks.values()):
        raise RuntimeError(f"training profiles fail reference closure: {checks}")
    return {"path": str(reference_path.resolve()), "checks": checks}


def flatten_profiles(payload: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for spec in AXES:
        for item in payload["profiles"][spec["name"]]["bins"]:
            row = {
                "feature": spec["name"],
                "feature_label": spec["label"],
                "bin": item["bin"],
                "lower": item["lower"],
                "upper": item["upper"],
                "feature_mean": item["feature_mean"],
                "n_pairs": item["n_pairs"],
                "pair_fraction": item["pair_fraction"],
                "n_cases_with_pairs": item["n_cases_with_pairs"],
            }
            for field in (
                "additive_label_per_primary",
                "additive_prediction_per_primary",
                "additive_label_minus_prediction_per_primary",
                "additive_null_per_primary",
            ):
                row[f"{field}_mean"] = item[field]["mean"]
                row[f"{field}_case_sem"] = item[field]["case_sem"]
            rows.append(row)
    return pd.DataFrame(rows)


def configure_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 8.0,
        "axes.labelsize": 8.5,
        "axes.titlesize": 9.0,
        "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.0,
        "legend.fontsize": 8.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": "white",
    })


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
        profile = payload["profiles"][spec["name"]]
        bins = profile["bins"]
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
            x,
            label,
            color="#000000",
            marker="o",
            markersize=3.1,
            linewidth=1.15,
            label="Half-shear label",
        )
        top.fill_between(
            x, label - label_sem, label + label_sem,
            color="#000000", alpha=0.11, linewidth=0.0,
        )
        top.plot(
            x,
            prediction,
            color="#0072B2",
            marker="s",
            markerfacecolor="white",
            markersize=3.0,
            linestyle="--",
            linewidth=1.15,
            label="V2.2 emulator",
        )
        top.fill_between(
            x, prediction - prediction_sem, prediction + prediction_sem,
            color="#0072B2", alpha=0.13, linewidth=0.0,
        )
        bottom.plot(
            x,
            residual,
            color="#D55E00",
            marker="o",
            markersize=3.0,
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
            -0.13,
            1.04,
            chr(ord("A") + column),
            transform=top.transAxes,
            fontweight="bold",
            fontsize=10,
            va="top",
        )
    axes[0, 0].set_ylabel("Bin-summed response\nper eligible primary")
    axes[1, 0].set_ylabel("Label $-$ V2.2\nper eligible primary")
    for column in (1, 2):
        axes[0, column].set_ylabel("")
        axes[1, column].set_ylabel("")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.916),
        ncol=2,
        frameon=False,
    )
    total = payload["profiles"]["primary_mag"]["total"]
    label_total = total["additive_label_per_primary"]
    prediction_total = total["additive_prediction_per_primary"]
    residual_total = total["additive_label_minus_prediction_per_primary"]
    figure.suptitle(
        "V2.2 cumulative pair response on its half-shear training cases",
        fontsize=11.0,
        y=0.992,
    )
    figure.text(
        0.5,
        0.951,
        (
            f"c40–199; {payload['metadata']['n_pairs']:,} supported pairs / "
            f"{payload['metadata']['n_primaries']:,} primaries; shaded bands are "
            "one SEM across rendered cases"
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
    total = payload["profiles"]["primary_mag"]["total"]
    lines = [
        "# V2.2 cumulative response on half-shear training cases",
        "",
        f"Cases `{payload['metadata']['case_window'][0]}--"
        f"{payload['metadata']['case_window'][1]}` contain "
        f"`{payload['metadata']['n_pairs']:,}` supported pairs and "
        f"`{payload['metadata']['n_primaries']:,}` eligible primaries.",
        "",
        "Each plotted bin is the sum of pair responses in that bin divided by "
        "the number of all eligible primaries in the same rendered case. Thus "
        "the plotted bins add exactly to the global summed response.",
        "",
        f"- Label sum: "
        f"`{total['additive_label_per_primary']['mean']:+.8f} +- "
        f"{total['additive_label_per_primary']['case_sem']:.8f}`.",
        f"- V2.2 sum: "
        f"`{total['additive_prediction_per_primary']['mean']:+.8f} +- "
        f"{total['additive_prediction_per_primary']['case_sem']:.8f}`.",
        f"- Label minus V2.2: "
        f"`{total['additive_label_minus_prediction_per_primary']['mean']:+.8f} "
        f"+- {total['additive_label_minus_prediction_per_primary']['case_sem']:.8f}`.",
        "",
        "## Largest absolute additive residual bin on each axis",
        "",
        "| feature | interval | label - V2.2 per primary | pair fraction |",
        "|---|---:|---:|---:|",
    ]
    for spec in AXES:
        bins = payload["profiles"][spec["name"]]["bins"]
        item = max(
            bins,
            key=lambda row: abs(
                row["additive_label_minus_prediction_per_primary"]["mean"]
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
        "This is an in-sample audit of the V2.2 half-shear training-case "
        "population. It does not use coherent-anchor or ConstGold labels.",
        "",
    ])
    output.write_text("\n".join(lines), encoding="utf-8")


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
        return float(value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalogue", required=True)
    parser.add_argument("--reference-closure", required=True)
    parser.add_argument("--tag", default="lsst_r_extnbr_v22")
    parser.add_argument("--case-min", type=int, default=40)
    parser.add_argument("--case-max", type=int, default=199)
    parser.add_argument("--shear", type=float, default=0.2)
    parser.add_argument("--score-chunk", type=int, default=1_000_000)
    parser.add_argument("--output-prefix", required=True)
    args = parser.parse_args()
    if (
        args.case_min > args.case_max
        or args.shear <= 0.0
        or args.score_chunk <= 0
    ):
        raise ValueError("invalid case window, shear, or score chunk")
    catalogue = Path(args.catalogue)
    reference = Path(args.reference_closure)
    for path in (catalogue, reference):
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

    payload, _ = stream_training_population(
        catalogue,
        args.tag,
        args.case_min,
        args.case_max,
        args.shear,
        args.score_chunk,
    )
    payload["reference_closure"] = validate_reference(payload, reference)
    flatten_profiles(payload).to_csv(f"{output_prefix}.csv", index=False)
    with open(f"{output_prefix}.json", "x", encoding="utf-8") as handle:
        json.dump(json_clean(payload), handle, indent=2, allow_nan=False)
        handle.write("\n")
    write_markdown(payload, Path(f"{output_prefix}.md"))
    plot_profiles(payload, output_prefix)
    print(
        f"saved cumulative V2.2 training response to {output_prefix}.*",
        flush=True,
    )


if __name__ == "__main__":
    main()
