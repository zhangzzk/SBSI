#!/usr/bin/env python3
"""Compare V2.2 loss reweighting, OOF bias correction, and variance weighting.

Cases 0--39 are external to every model trained by the OOF pipeline.  The
reported validation half is cases 20--39; cases 0--19 are retained as a
development replication.  Response bins and scene-tail selections are fixed
from earlier V2.2 diagnostics and use only frozen baseline predictions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd
import xgboost as xgb


ROOT = Path(__file__).resolve().parents[1]
BLENDEMU = Path("/home/z/Zekang.Zhang/blendemu")
for path in (ROOT, BLENDEMU, BLENDEMU / "scripts"):
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)

from scripts.train_v22_oof_learning import (  # noqa: E402
    CONDITIONAL_FEATURES,
    conditional_matrix,
    physical_predict,
)


PAIR_BIN_EDGES = np.asarray(
    [
        -np.inf,
        0.4388450291513489,
        0.6715223563877121,
        0.831023961656823,
        0.9566271383115033,
        1.1223961638504578,
        np.inf,
    ],
    dtype=np.float64,
)
AMPLITUDE_TAGS = [
    (0.010, "lsst_r_extnbr_v22_rpowposw0010"),
    (0.020, "lsst_r_extnbr_v22_rpowposw0020"),
    (0.035, "lsst_r_extnbr_v22_rpowposw0035"),
    (0.050, "lsst_r_extnbr_v22_rpowposw0050"),
    (0.065, "lsst_r_extnbr_v22_rpowposa0065"),
    (0.100, "lsst_r_extnbr_v22_rpowposa010"),
    (0.150, "lsst_r_extnbr_v22_rpowposa015"),
    (0.200, "lsst_r_extnbr_v22_rpowposw0200"),
]


def strict_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def stat(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("stat requires at least two finite values")
    sd = float(values.std(ddof=1))
    return {
        "mean": float(values.mean()),
        "case_sd": sd,
        "case_sem": float(sd / np.sqrt(len(values))),
        "n_cases": int(len(values)),
    }


def load_cache(cache: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    with (cache / "metadata.json").open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    arrays = {
        name: np.load(cache / filename, mmap_mode="r")
        for name, filename in metadata["arrays"].items()
    }
    return metadata, arrays


def predict_tag(tag: str, x_scaled: np.ndarray) -> np.ndarray:
    metadata_path = BLENDEMU / "models" / f"emulator_metadata_{tag}.json"
    with metadata_path.open(encoding="utf-8") as handle:
        task = json.load(handle)["tasks"]["regression"]
    booster = xgb.Booster({"device": "cpu", "n_jobs": -1})
    booster.load_model(BLENDEMU / "models" / task["model_file"])
    standardization = task["standardization"]
    return physical_predict(
        booster,
        x_scaled,
        float(standardization["mean"]),
        float(standardization["std"]),
    )


def case_means(
    case: np.ndarray,
    values: np.ndarray,
    case_min: int,
    case_max: int,
) -> np.ndarray:
    use = (case >= case_min) & (case <= case_max)
    local = case[use].astype(np.int64) - case_min
    n_cases = case_max - case_min + 1
    count = np.bincount(local, minlength=n_cases).astype(np.float64)
    if np.any(count == 0):
        raise RuntimeError("case statistic has an empty case")
    return np.bincount(
        local, weights=np.asarray(values[use], dtype=np.float64), minlength=n_cases
    ) / count


def row_summary(
    case: np.ndarray,
    label: np.ndarray,
    prediction: np.ndarray,
    baseline: np.ndarray,
    case_min: int,
    case_max: int,
) -> dict[str, Any]:
    residual = np.asarray(label, dtype=np.float64) - np.asarray(
        prediction, dtype=np.float64
    )
    baseline_residual = np.asarray(label, dtype=np.float64) - np.asarray(
        baseline, dtype=np.float64
    )
    residual_case = case_means(case, residual, case_min, case_max)
    mse_case = case_means(case, np.square(residual), case_min, case_max)
    baseline_mse_case = case_means(
        case, np.square(baseline_residual), case_min, case_max
    )
    mse_difference = mse_case - baseline_mse_case
    mse_percent = 100.0 * mse_difference / baseline_mse_case
    use = (case >= case_min) & (case <= case_max)
    return {
        "case_window": [case_min, case_max],
        "n_rows": int(use.sum()),
        "pooled_label_mean": float(np.asarray(label[use], dtype=np.float64).mean()),
        "pooled_prediction_mean": float(
            np.asarray(prediction[use], dtype=np.float64).mean()
        ),
        "pooled_residual_mean": float(residual[use].mean()),
        "pooled_mse": float(np.mean(np.square(residual[use]))),
        "residual_mean": stat(residual_case),
        "mse": stat(mse_case),
        "mse_minus_baseline": stat(mse_difference),
        "mse_percent_change_from_baseline": stat(mse_percent),
        "case_values": {
            "residual_mean": residual_case.tolist(),
            "mse": mse_case.tolist(),
            "mse_minus_baseline": mse_difference.tolist(),
        },
    }


def pair_bin_summary(
    case: np.ndarray,
    label: np.ndarray,
    baseline: np.ndarray,
    predictions: dict[str, np.ndarray],
    case_min: int = 20,
    case_max: int = 39,
) -> tuple[dict[str, Any], pd.DataFrame]:
    use = (case >= case_min) & (case <= case_max)
    local_case = case[use].astype(np.int64) - case_min
    n_cases = case_max - case_min + 1
    bins = np.searchsorted(PAIR_BIN_EDGES, baseline[use], side="right") - 1
    if np.any((bins < 0) | (bins >= len(PAIR_BIN_EDGES) - 1)):
        raise RuntimeError("response-power bin assignment failed")
    n_bins = len(PAIR_BIN_EDGES) - 1
    flat = local_case * n_bins + bins
    count = np.bincount(flat, minlength=n_cases * n_bins).reshape(n_cases, n_bins)
    label_sum = np.bincount(
        flat,
        weights=np.asarray(label[use], dtype=np.float64),
        minlength=n_cases * n_bins,
    ).reshape(n_cases, n_bins)
    if np.any(count == 0):
        raise RuntimeError("at least one validation case lacks a frozen response bin")
    rows: list[dict[str, Any]] = []
    payload: dict[str, Any] = {}
    for name, prediction in predictions.items():
        pred_sum = np.bincount(
            flat,
            weights=np.asarray(prediction[use], dtype=np.float64),
            minlength=n_cases * n_bins,
        ).reshape(n_cases, n_bins)
        model_bins = []
        for index in range(n_bins):
            label_mean = label_sum[:, index] / count[:, index]
            pred_mean = pred_sum[:, index] / count[:, index]
            residual = label_mean - pred_mean
            ratio = label_mean / pred_mean
            item = {
                "bin": index,
                "lower": None if index == 0 else float(PAIR_BIN_EDGES[index]),
                "upper": None if index == n_bins - 1 else float(PAIR_BIN_EDGES[index + 1]),
                "n_pairs": int(count[:, index].sum()),
                "label_mean": stat(label_mean),
                "prediction_mean": stat(pred_mean),
                "label_minus_prediction": stat(residual),
                "label_over_prediction": stat(ratio),
            }
            model_bins.append(item)
            rows.append({
                "model": name,
                "bin": index,
                "n_pairs": item["n_pairs"],
                "label": item["label_mean"]["mean"],
                "label_sem": item["label_mean"]["case_sem"],
                "prediction": item["prediction_mean"]["mean"],
                "prediction_sem": item["prediction_mean"]["case_sem"],
                "residual": item["label_minus_prediction"]["mean"],
                "residual_sem": item["label_minus_prediction"]["case_sem"],
                "ratio": item["label_over_prediction"]["mean"],
                "ratio_sem": item["label_over_prediction"]["case_sem"],
            })
        high = np.arange(n_bins) >= 4
        high_label = label_sum[:, high].sum(axis=1) / count[:, high].sum(axis=1)
        high_pred = pred_sum[:, high].sum(axis=1) / count[:, high].sum(axis=1)
        payload[name] = {
            "bins": model_bins,
            "highest_two_bins": {
                "n_pairs": int(count[:, high].sum()),
                "label_mean": stat(high_label),
                "prediction_mean": stat(high_pred),
                "label_minus_prediction": stat(high_label - high_pred),
                "label_over_prediction": stat(high_label / high_pred),
            },
        }
    return payload, pd.DataFrame(rows)


def group_structure(
    case: np.ndarray,
    input_index: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    monotonic = np.all(
        (case[1:] > case[:-1])
        | ((case[1:] == case[:-1]) & (input_index[1:] >= input_index[:-1]))
    )
    if monotonic:
        order = np.arange(len(case), dtype=np.int32)
    else:
        order = np.lexsort((input_index, case)).astype(np.int32)
    ordered_case = case[order]
    ordered_input = input_index[order]
    starts = np.r_[
        0,
        1 + np.flatnonzero(
            (ordered_case[1:] != ordered_case[:-1])
            | (ordered_input[1:] != ordered_input[:-1])
        ),
    ].astype(np.int64)
    return order, starts, ordered_case[starts]


def reduce_group(values: np.ndarray, order: np.ndarray, starts: np.ndarray) -> np.ndarray:
    return np.add.reduceat(np.asarray(values, dtype=np.float64)[order], starts)


def selected_case_mean(
    group_case: np.ndarray,
    values: np.ndarray,
    selection: np.ndarray,
    case_min: int = 0,
    case_max: int = 39,
) -> np.ndarray:
    use = np.asarray(selection, dtype=bool)
    local = group_case[use].astype(np.int64) - case_min
    n_cases = case_max - case_min + 1
    count = np.bincount(local, minlength=n_cases).astype(np.float64)
    if np.any(count == 0):
        raise RuntimeError("scene selection is empty in at least one case")
    return np.bincount(
        local,
        weights=np.asarray(values, dtype=np.float64)[use],
        minlength=n_cases,
    ) / count


def scene_summaries(
    case: np.ndarray,
    input_index: np.ndarray,
    shear_angle: np.ndarray,
    label: np.ndarray,
    null: np.ndarray,
    predictions: dict[str, np.ndarray],
) -> tuple[dict[str, Any], pd.DataFrame]:
    order, starts, group_case = group_structure(case, input_index)
    phase = np.deg2rad(2.0 * np.asarray(shear_angle, dtype=np.float64))
    cosine, sine = np.cos(phase), np.sin(phase)
    group_label = reduce_group(label, order, starts)
    group_null = reduce_group(null, order, starts)
    group_cos = reduce_group(cosine, order, starts)
    group_sin = reduce_group(sine, order, starts)
    direction_power = np.square(group_cos) + np.square(group_sin)
    valid_vector = direction_power > 1.0e-8
    y1 = np.full(len(starts), np.nan, dtype=np.float64)
    y2 = np.full(len(starts), np.nan, dtype=np.float64)
    y1[valid_vector] = (
        group_label[valid_vector] * group_cos[valid_vector]
        - group_null[valid_vector] * group_sin[valid_vector]
    ) / direction_power[valid_vector]
    y2[valid_vector] = (
        group_label[valid_vector] * group_sin[valid_vector]
        + group_null[valid_vector] * group_cos[valid_vector]
    ) / direction_power[valid_vector]
    baseline_scene = reduce_group(predictions["baseline"], order, starts)
    selections = {
        "all": np.ones(len(starts), dtype=bool),
        "half_scene_gt_0p05": baseline_scene > 0.05,
        "half_scene_gt_0p10": baseline_scene > 0.10,
    }
    payload: dict[str, Any] = {
        "n_scenes": int(len(starts)),
        "n_vector_scenes": int(valid_vector.sum()),
        "tail_definition": (
            "frozen baseline half-scene pair sum; 0.05 corresponds to the "
            "earlier full-neighbour coherent threshold 0.1"
        ),
        "models": {},
    }
    rows: list[dict[str, Any]] = []
    for name, prediction in predictions.items():
        scene_prediction = reduce_group(prediction, order, starts)
        prediction_cos = reduce_group(prediction * cosine, order, starts)
        prediction_sin = reduce_group(prediction * sine, order, starts)
        power = np.square(prediction_cos) + np.square(prediction_sin)
        usable = valid_vector & (power > 0.0)
        dot = prediction_cos * y1 + prediction_sin * y2
        cross = -prediction_sin * y1 + prediction_cos * y2
        local = group_case[usable].astype(np.int64)
        case_power = np.bincount(local, weights=power[usable], minlength=40)
        case_dot = np.bincount(local, weights=dot[usable], minlength=40)
        case_cross = np.bincount(local, weights=cross[usable], minlength=40)
        if np.any(case_power <= 0.0):
            raise RuntimeError("vector model has zero case power")
        slope = case_dot / case_power
        orthogonal = case_cross / case_power
        model_payload: dict[str, Any] = {
            "vector_closure": {
                "all_c0_39": stat(slope),
                "development_c0_19": stat(slope[:20]),
                "validation_c20_39": stat(slope[20:]),
                "orthogonal_all_c0_39": stat(orthogonal),
                "orthogonal_validation_c20_39": stat(orthogonal[20:]),
            },
            "scene_scalar": {},
        }
        for selection_name, selection in selections.items():
            label_case = selected_case_mean(group_case, group_label, selection)
            pred_case = selected_case_mean(group_case, scene_prediction, selection)
            residual_case = label_case - pred_case
            n_by_case = np.bincount(
                group_case[selection].astype(np.int64), minlength=40
            )
            factor = 2.0
            summary = {
                "n_scenes": int(selection.sum()),
                "n_scenes_by_case": n_by_case.astype(int).tolist(),
                "half_scene_label": stat(label_case),
                "half_scene_prediction": stat(pred_case),
                "half_scene_label_minus_prediction": stat(residual_case),
                "full_equivalent_label": stat(factor * label_case),
                "full_equivalent_prediction": stat(factor * pred_case),
                "full_equivalent_label_minus_prediction": stat(
                    factor * residual_case
                ),
                "validation_full_equivalent_residual": stat(
                    factor * residual_case[20:]
                ),
            }
            model_payload["scene_scalar"][selection_name] = summary
            rows.append({
                "model": name,
                "selection": selection_name,
                "n_scenes": int(selection.sum()),
                "full_equivalent_residual": summary[
                    "validation_full_equivalent_residual"
                ]["mean"],
                "full_equivalent_residual_sem": summary[
                    "validation_full_equivalent_residual"
                ]["case_sem"],
                "vector_slope_validation": model_payload["vector_closure"][
                    "validation_c20_39"
                ]["mean"],
                "vector_slope_validation_sem": model_payload["vector_closure"][
                    "validation_c20_39"
                ]["case_sem"],
            })
        payload["models"][name] = model_payload
    return payload, pd.DataFrame(rows)


def variance_calibration(
    case: np.ndarray,
    variance_prediction: np.ndarray,
    centered_square: np.ndarray,
) -> tuple[dict[str, Any], pd.DataFrame]:
    development = case <= 19
    validation = case >= 20
    edges = np.quantile(
        np.asarray(variance_prediction[development], dtype=np.float64),
        np.linspace(0.0, 1.0, 11),
    )
    edges[0], edges[-1] = -np.inf, np.inf
    if np.any(np.diff(edges) <= 0.0):
        raise RuntimeError("conditional variance decile edges are not unique")
    bins = np.searchsorted(edges, variance_prediction[validation], side="right") - 1
    val_case = case[validation].astype(np.int64) - 20
    flat = val_case * 10 + bins
    count = np.bincount(flat, minlength=200).reshape(20, 10)
    pred_sum = np.bincount(
        flat,
        weights=np.asarray(variance_prediction[validation], dtype=np.float64),
        minlength=200,
    ).reshape(20, 10)
    obs_sum = np.bincount(
        flat,
        weights=np.asarray(centered_square[validation], dtype=np.float64),
        minlength=200,
    ).reshape(20, 10)
    if np.any(count == 0):
        raise RuntimeError("variance decile is empty in a validation case")
    rows = []
    bins_payload = []
    for index in range(10):
        predicted = pred_sum[:, index] / count[:, index]
        observed = obs_sum[:, index] / count[:, index]
        item = {
            "decile": index,
            "n_pairs": int(count[:, index].sum()),
            "predicted_variance": stat(predicted),
            "observed_centered_residual_square": stat(observed),
            "observed_over_predicted": stat(observed / predicted),
        }
        bins_payload.append(item)
        rows.append({
            "decile": index,
            "n_pairs": item["n_pairs"],
            "predicted": item["predicted_variance"]["mean"],
            "predicted_sem": item["predicted_variance"]["case_sem"],
            "observed": item["observed_centered_residual_square"]["mean"],
            "observed_sem": item["observed_centered_residual_square"]["case_sem"],
            "ratio": item["observed_over_predicted"]["mean"],
            "ratio_sem": item["observed_over_predicted"]["case_sem"],
        })
    payload = {
        "edges_from_development_c0_19": [
            None if not np.isfinite(value) else float(value) for value in edges
        ],
        "validation_c20_39": bins_payload,
        "pooled_correlation": float(np.corrcoef(
            variance_prediction[validation], centered_square[validation]
        )[0, 1]),
    }
    return payload, pd.DataFrame(rows)


def make_plot(
    output: Path,
    payload: dict[str, Any],
    pair_bins: pd.DataFrame,
    scene_table: pd.DataFrame,
    variance_table: pd.DataFrame,
) -> None:
    plt.rcParams.update({
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 150,
    })
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 9.5))
    ax = axes[0, 0]
    alphas = [0.0] + [value for value, _ in AMPLITUDE_TAGS]
    names = ["baseline"] + [f"amp_{value:.3f}" for value, _ in AMPLITUDE_TAGS]
    tail = np.asarray([
        payload["pair_bins"][name]["highest_two_bins"]["label_minus_prediction"]["mean"]
        for name in names
    ])
    tail_sem = np.asarray([
        payload["pair_bins"][name]["highest_two_bins"]["label_minus_prediction"]["case_sem"]
        for name in names
    ])
    mse = np.asarray([
        payload["models"][name]["validation_c20_39"][
            "mse_percent_change_from_baseline"
        ]["mean"]
        for name in names
    ])
    mse_sem = np.asarray([
        payload["models"][name]["validation_c20_39"][
            "mse_percent_change_from_baseline"
        ]["case_sem"]
        for name in names
    ])
    ax.errorbar(alphas, tail, yerr=tail_sem, marker="o", color="#0072B2", label="High-bin label - prediction")
    ax.axhline(0.0, color="0.4", ls="--", lw=1)
    ax.set_xlabel(r"Positive-response loss strength $\alpha$")
    ax.set_ylabel("High-response pair deficit", color="#0072B2")
    ax.tick_params(axis="y", labelcolor="#0072B2")
    twin = ax.twinx()
    twin.errorbar(alphas, mse, yerr=mse_sem, marker="s", color="#D55E00", label="Global row MSE")
    twin.axhline(0.0, color="#D55E00", ls=":", lw=1)
    twin.set_ylabel("Global row MSE change (%)", color="#D55E00")
    twin.tick_params(axis="y", labelcolor="#D55E00")
    ax.set_title("A  Tail/global trade-off on cases 20-39")

    ax = axes[0, 1]
    shown = ["baseline", "amp_0.020", "amp_0.050", "amp_0.065", "bias_oof", "inverse_variance"]
    colors = {
        "baseline": "#000000",
        "amp_0.020": "#56B4E9",
        "amp_0.050": "#0072B2",
        "amp_0.065": "#009E73",
        "bias_oof": "#CC79A7",
        "inverse_variance": "#D55E00",
    }
    x = np.arange(6)
    baseline_rows = pair_bins.loc[pair_bins.model.eq("baseline")].sort_values("bin")
    ax.errorbar(x, baseline_rows.label, yerr=baseline_rows.label_sem, marker="o", color="0.5", lw=2.5, label="label")
    for name in shown:
        local = pair_bins.loc[pair_bins.model.eq(name)].sort_values("bin")
        ax.plot(x, local.prediction, marker=".", color=colors[name], label=name.replace("_", " "))
    ax.set_xticks(x, ["near 0", "1", "2", "3", "4", "5"])
    ax.set_xlabel("Frozen V2.2 response-power bin")
    ax.set_ylabel("Mean pair response")
    ax.set_title("B  Conditional pair means")
    ax.legend(fontsize=8, ncol=2)

    ax = axes[1, 0]
    selection = "half_scene_gt_0p05"
    order_names = ["baseline", "amp_0.020", "amp_0.050", "amp_0.065", "bias_oof", "bias_oof_mean_aligned", "inverse_variance"]
    rows = scene_table.loc[
        scene_table.selection.eq(selection) & scene_table.model.isin(order_names)
    ].set_index("model").loc[order_names].reset_index()
    xpos = np.arange(len(rows))
    ax.errorbar(
        xpos,
        rows.full_equivalent_residual,
        yerr=rows.full_equivalent_residual_sem,
        fmt="o",
        capsize=3,
        color="#0072B2",
    )
    ax.axhline(0.0, color="0.4", ls="--", lw=1)
    ax.set_xticks(xpos, [name.replace("_", "\n") for name in rows.model], rotation=0)
    ax.set_ylabel("Full-equivalent label - prediction")
    ax.set_title("C  Faint/crowded half-scene tail (baseline sum > 0.05)")

    ax = axes[1, 1]
    ax.errorbar(
        variance_table.predicted,
        variance_table.observed,
        xerr=variance_table.predicted_sem,
        yerr=variance_table.observed_sem,
        marker="o",
        capsize=2,
        color="#D55E00",
    )
    maximum = float(max(variance_table.predicted.max(), variance_table.observed.max()))
    ax.plot([0, maximum], [0, maximum], color="0.4", ls="--", lw=1)
    for _, row in variance_table.iterrows():
        ax.annotate(str(int(row.decile)), (row.predicted, row.observed), fontsize=7)
    ax.set_xlabel("Predicted conditional variance")
    ax.set_ylabel("Observed centered residual squared")
    ax.set_title("D  Variance calibration on cases 20-39")
    fig.suptitle("V2.2 half-shear learning tests: external cases 0-39", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(output.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-prefix", required=True)
    args = parser.parse_args()

    cache, run_dir = Path(args.cache), Path(args.run_dir)
    output = Path(args.output_prefix)
    output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".json", ".csv", ".pair_bins.csv", ".scene_tail.csv", ".variance.csv", ".md", ".png", ".pdf"):
        path = output.with_suffix(suffix)
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}")

    metadata, arrays = load_cache(cache)
    evaluation = np.flatnonzero(np.asarray(arrays["case"]) < 40).astype(np.int32)
    case = np.asarray(arrays["case"][evaluation], dtype=np.int16)
    input_index = np.asarray(arrays["input_index"][evaluation], dtype=np.int64)
    shear_angle = np.asarray(arrays["shear_angle"][evaluation], dtype=np.float32)
    label = np.asarray(arrays["label"][evaluation], dtype=np.float64)
    null = np.asarray(arrays["null"][evaluation], dtype=np.float64)
    x_scaled = np.asarray(arrays["x_scaled"][evaluation], dtype=np.float32)
    x_raw = np.asarray(arrays["x_raw"][evaluation], dtype=np.float32)
    baseline = np.asarray(arrays["v22_prediction"][evaluation], dtype=np.float64)
    if len(evaluation) != metadata["n_evaluation_rows"] or set(np.unique(case)) != set(range(40)):
        raise RuntimeError("external evaluation population drifted")

    predictions: dict[str, np.ndarray] = {"baseline": baseline}
    model_specs: dict[str, Any] = {
        "baseline": {"kind": "frozen_v22", "alpha": 0.0}
    }
    for alpha, tag in AMPLITUDE_TAGS:
        name = f"amp_{alpha:.3f}"
        predictions[name] = predict_tag(tag, x_scaled)
        model_specs[name] = {
            "kind": "positive_response_amplitude_weighting",
            "alpha": alpha,
            "tag": tag,
        }

    conditional = conditional_matrix(
        x_scaled, x_raw, baseline, np.arange(len(evaluation), dtype=np.int32)
    )
    with (run_dir / "bias_final_summary.json").open(encoding="utf-8") as handle:
        bias_summary = json.load(handle)
    bias_booster = xgb.Booster({"device": "cpu", "n_jobs": -1})
    bias_booster.load_model(run_dir / "bias_final.json")
    bias_std = bias_summary["target_standardization"]
    correction = physical_predict(
        bias_booster,
        conditional,
        float(bias_std["mean"]),
        float(bias_std["std"]),
    )
    predictions["bias_oof"] = baseline + correction
    alignment = float(bias_summary["mean_alignment_offset"])
    predictions["bias_oof_mean_aligned"] = baseline + correction + alignment
    model_specs["bias_oof"] = {
        "kind": "conditional_bias_stack",
        "training": "case-OOF residuals c40-199",
        "features": CONDITIONAL_FEATURES,
        "mean_alignment_offset_applied": 0.0,
    }
    model_specs["bias_oof_mean_aligned"] = {
        "kind": "conditional_bias_stack_mean_aligned",
        "training": "case-OOF residuals c40-199",
        "features": CONDITIONAL_FEATURES,
        "mean_alignment_offset_applied": alignment,
    }

    with (run_dir / "variance_weighted_mean_summary.json").open(encoding="utf-8") as handle:
        weighted_summary = json.load(handle)
    weighted_booster = xgb.Booster({"device": "cpu", "n_jobs": -1})
    weighted_booster.load_model(run_dir / "variance_weighted_mean.json")
    source_std = weighted_summary["source_standardization"]
    predictions["inverse_variance"] = physical_predict(
        weighted_booster,
        x_scaled,
        float(source_std["mean"]),
        float(source_std["std"]),
    )
    model_specs["inverse_variance"] = {
        "kind": "stabilized_inverse_conditional_variance_weighting",
        "weight_summary": weighted_summary["weight_summary"],
    }

    with (run_dir / "variance_final_summary.json").open(encoding="utf-8") as handle:
        variance_summary = json.load(handle)
    variance_booster = xgb.Booster({"device": "cpu", "n_jobs": -1})
    variance_booster.load_model(run_dir / "variance_final.json")
    variance_std = variance_summary["target_standardization"]
    variance_prediction = physical_predict(
        variance_booster,
        conditional,
        float(variance_std["mean"]),
        float(variance_std["std"]),
    )
    variance_floor = float(weighted_summary["weight_summary"]["variance_floor"])
    variance_prediction = np.maximum(variance_prediction, variance_floor)
    centered_square = np.square(label - baseline - correction)

    models_payload: dict[str, Any] = {}
    summary_rows = []
    for name, prediction in predictions.items():
        development = row_summary(case, label, prediction, baseline, 0, 19)
        validation = row_summary(case, label, prediction, baseline, 20, 39)
        models_payload[name] = {
            "specification": model_specs[name],
            "development_c0_19": development,
            "validation_c20_39": validation,
        }
        summary_rows.append({
            "model": name,
            "kind": model_specs[name]["kind"],
            "alpha": model_specs[name].get("alpha"),
            "validation_residual": validation["residual_mean"]["mean"],
            "validation_residual_sem": validation["residual_mean"]["case_sem"],
            "validation_mse": validation["mse"]["mean"],
            "validation_mse_sem": validation["mse"]["case_sem"],
            "validation_mse_percent_change": validation[
                "mse_percent_change_from_baseline"
            ]["mean"],
            "validation_mse_percent_change_sem": validation[
                "mse_percent_change_from_baseline"
            ]["case_sem"],
        })

    pair_payload, pair_table = pair_bin_summary(
        case, label, baseline, predictions
    )
    scene_payload, scene_table = scene_summaries(
        case, input_index, shear_angle, label, null, predictions
    )
    variance_payload, variance_table = variance_calibration(
        case, variance_prediction, centered_square
    )
    payload = {
        "design": {
            "source": "half-shear response pairs only",
            "conditional_training_cases": [40, 199],
            "external_evaluation_cases": [0, 39],
            "reported_validation_cases": [20, 39],
            "base_oof_folds": 4,
            "response_bin_edges": [
                None if not np.isfinite(value) else float(value)
                for value in PAIR_BIN_EDGES
            ],
            "labels_used_in_tail_selection": False,
            "constgold_opened": False,
            "anchor_truth_opened": False,
        },
        "models": models_payload,
        "pair_bins": pair_payload,
        "scenes": scene_payload,
        "conditional_variance": variance_payload,
    }
    strict_json(output.with_suffix(".json"), payload)
    pd.DataFrame(summary_rows).to_csv(output.with_suffix(".csv"), index=False)
    pair_table.to_csv(output.with_suffix(".pair_bins.csv"), index=False)
    scene_table.to_csv(output.with_suffix(".scene_tail.csv"), index=False)
    variance_table.to_csv(output.with_suffix(".variance.csv"), index=False)
    make_plot(output, payload, pair_table, scene_table, variance_table)

    baseline_tail = pair_payload["baseline"]["highest_two_bins"][
        "label_minus_prediction"
    ]
    bias_tail = pair_payload["bias_oof"]["highest_two_bins"][
        "label_minus_prediction"
    ]
    variance_tail = pair_payload["inverse_variance"]["highest_two_bins"][
        "label_minus_prediction"
    ]
    lines = [
        "# V2.2 OOF conditional-learning comparison",
        "",
        "All quoted checks below use external half-shear cases 20--39. Cases 40--199 "
        "were used for case-cross-fitted residual learning; constgold and coherent anchors "
        "were not opened.",
        "",
        f"- Baseline high-response pair deficit: `{baseline_tail['mean']:+.6f} +- {baseline_tail['case_sem']:.6f}`.",
        f"- OOF bias-stack high-response deficit: `{bias_tail['mean']:+.6f} +- {bias_tail['case_sem']:.6f}`.",
        f"- Inverse-variance retrain high-response deficit: `{variance_tail['mean']:+.6f} +- {variance_tail['case_sem']:.6f}`.",
        f"- Variance-model pooled prediction/squared-residual correlation: `{variance_payload['pooled_correlation']:+.5f}`.",
        "",
        "See the JSON and CSV tables for paired case-level global-MSE changes, vector closure, "
        "and the full-equivalent scene-tail residual.",
    ]
    output.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(output),
        "baseline_tail": baseline_tail,
        "bias_tail": bias_tail,
        "inverse_variance_tail": variance_tail,
        "variance_correlation": variance_payload["pooled_correlation"],
    }, indent=2, sort_keys=True, allow_nan=False))
    print("V22_OOF_LEARNING_EVALUATION_DONE", flush=True)


if __name__ == "__main__":
    main()
