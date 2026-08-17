#!/usr/bin/env python3
"""Evaluate two half-shear-selected weak-loss models on coherent anchors.

The response-weight strengths are fixed before this script reads coherent-anchor
truth.  The tail is likewise fixed from the frozen V2.2 scene prediction:
``R_blend,V2.2 > 0.1``.  Every uncertainty is one SEM across rendered cases.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd


TAIL_THRESHOLD = 0.1
BLOCKS = {
    "all_c400_899": (400, 899),
    "c400_599": (400, 599),
    "c600_699": (600, 699),
    "c700_899": (700, 899),
}
SUBSETS = ("all", "outside_tail", "tail")
REQUIRED_COLUMNS = [
    "case",
    "input_index",
    "R_blend_truth",
    "prediction_baseline",
    "prediction_model",
    "model_tag",
]


def strict_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def case_stat(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("case_stat requires at least two finite values")
    sd = float(values.std(ddof=1))
    return {
        "mean": float(values.mean()),
        "case_sd": sd,
        "case_sem": float(sd / np.sqrt(len(values))),
        "n_cases": int(len(values)),
    }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_parts(
    parts_root: Path,
    tag: str,
    case_min: int,
    case_max: int,
) -> pd.DataFrame:
    paths = [parts_root / tag / f"case{case}.feather" for case in range(case_min, case_max + 1)]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise RuntimeError(f"{tag}: missing {len(missing)} case parts; first={missing[0]}")
    frame = pd.concat(
        [pd.read_feather(path, columns=REQUIRED_COLUMNS) for path in paths],
        ignore_index=True,
    )
    if frame.duplicated(["case", "input_index"]).any():
        raise RuntimeError(f"{tag}: duplicate coherent-anchor keys")
    cases = set(frame.case.astype(int).unique())
    expected = set(range(case_min, case_max + 1))
    if cases != expected:
        raise RuntimeError(f"{tag}: case coverage mismatch")
    if set(frame.model_tag.unique()) != {tag}:
        raise RuntimeError(f"{tag}: model-tag drift in scored parts")
    values = frame[["R_blend_truth", "prediction_baseline", "prediction_model"]].to_numpy(float)
    if not np.isfinite(values).all():
        raise RuntimeError(f"{tag}: non-finite response value")
    return frame.sort_values(["case", "input_index"], kind="stable").reset_index(drop=True)


def subset_mask(frame: pd.DataFrame, subset: str) -> np.ndarray:
    baseline = frame.prediction_baseline.to_numpy(float)
    if subset == "all":
        return np.ones(len(frame), dtype=bool)
    if subset == "outside_tail":
        return baseline <= TAIL_THRESHOLD
    if subset == "tail":
        return baseline > TAIL_THRESHOLD
    raise ValueError(f"unknown subset {subset}")


def summarize_slice(
    frame: pd.DataFrame,
    prediction_column: str,
    case_min: int,
    case_max: int,
    selection: np.ndarray,
) -> tuple[dict[str, Any], pd.DataFrame]:
    case_mask = frame.case.between(case_min, case_max).to_numpy(bool)
    use = case_mask & np.asarray(selection, dtype=bool)
    chosen = frame.loc[use, [
        "case", "R_blend_truth", "prediction_baseline",
    ]].copy()
    chosen["prediction"] = frame.loc[use, prediction_column].to_numpy(float)
    expected_cases = case_max - case_min + 1
    by_case = chosen.groupby("case", sort=True)[
        ["R_blend_truth", "prediction_baseline", "prediction"]
    ].mean()
    if len(by_case) != expected_cases:
        raise RuntimeError(
            f"selection covers {len(by_case)} cases, expected {expected_cases}"
        )
    by_case["truth_minus_baseline"] = (
        by_case.R_blend_truth - by_case.prediction_baseline
    )
    by_case["truth_minus_prediction"] = (
        by_case.R_blend_truth - by_case.prediction
    )
    by_case["prediction_minus_baseline"] = (
        by_case.prediction - by_case.prediction_baseline
    )
    base_gap = float(by_case.truth_minus_baseline.mean())
    model_gap = float(by_case.truth_minus_prediction.mean())
    payload = {
        "n_anchors": int(len(chosen)),
        "anchors_per_case": case_stat(
            chosen.groupby("case", sort=True).size().to_numpy(float)
        ),
        "truth": case_stat(by_case.R_blend_truth.to_numpy(float)),
        "baseline_prediction": case_stat(by_case.prediction_baseline.to_numpy(float)),
        "model_prediction": case_stat(by_case.prediction.to_numpy(float)),
        "baseline_truth_minus_prediction": case_stat(
            by_case.truth_minus_baseline.to_numpy(float)
        ),
        "model_truth_minus_prediction": case_stat(
            by_case.truth_minus_prediction.to_numpy(float)
        ),
        "model_minus_baseline_prediction": case_stat(
            by_case.prediction_minus_baseline.to_numpy(float)
        ),
        "fraction_of_baseline_gap_removed": (
            None if base_gap == 0.0 else float(1.0 - model_gap / base_gap)
        ),
    }
    model_sem = float(payload["model_truth_minus_prediction"]["case_sem"])
    payload["model_gap_consistent_with_zero_at_2sem"] = bool(
        abs(model_gap) <= 2.0 * model_sem
    )
    payload["absolute_gap_improved"] = bool(abs(model_gap) < abs(base_gap))
    return payload, by_case.reset_index()


def make_plot(output: Path, table: pd.DataFrame) -> None:
    models = list(dict.fromkeys(table.model.tolist()))
    groups = list(BLOCKS)
    labels = {"baseline_v22": "Baseline V2.2"}
    for model in models:
        if model.startswith("alpha_"):
            labels[model] = rf"$\alpha={float(model.removeprefix('alpha_').replace('p', '.')):g}$"
    palette = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9"]
    colors = {model: palette[index % len(palette)] for index, model in enumerate(models)}
    x = np.arange(len(groups), dtype=float)
    offsets = np.linspace(-0.22, 0.22, len(models))
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.9), sharey=True)
    titles = {
        "all": "All coherent anchors",
        "outside_tail": r"Outside tail: raw V2.2 $\leq 0.1$",
        "tail": r"Tail: raw V2.2 $> 0.1$",
    }
    group_labels = ["All\n400–899", "400–599", "600–699", "700–899"]
    for axis, subset in zip(axes, SUBSETS):
        for offset, model in zip(offsets, models):
            selected = table[(table.subset == subset) & (table.model == model)].set_index("group")
            means = np.asarray([selected.loc[group, "gap"] for group in groups], float)
            sems = np.asarray([selected.loc[group, "gap_sem"] for group in groups], float)
            axis.errorbar(
                x + offset,
                means,
                yerr=sems,
                marker="o",
                linestyle="none",
                capsize=3,
                color=colors.get(model),
                label=labels.get(model, model),
            )
        axis.axhline(0.0, color="0.35", linestyle="--", linewidth=1)
        axis.set_xticks(x, group_labels)
        axis.set_title(titles[subset])
        axis.grid(axis="y", alpha=0.18)
    axes[0].set_ylabel(r"Coherent truth $-$ model prediction")
    axes[1].legend(frameon=False, loc="upper left")
    fig.suptitle(
        "Half-shear-selected response weighting transferred to coherent anchors\n"
        "Errors are one SEM across rendered cases; tail selection uses frozen baseline only",
        y=1.03,
    )
    fig.tight_layout()
    fig.savefig(output.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parts-root", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--tags", nargs="+", required=True)
    parser.add_argument("--model-names", nargs="+", required=True)
    parser.add_argument("--case-min", type=int, default=400)
    parser.add_argument("--case-max", type=int, default=899)
    parser.add_argument("--previously-inspected", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (args.case_min, args.case_max) != (400, 899):
        raise ValueError("canonical transfer requires cases 400--899")
    if len(args.tags) != len(args.model_names):
        raise ValueError("--tags and --model-names must have equal length")
    if len(set(args.tags)) != len(args.tags) or len(set(args.model_names)) != len(args.model_names):
        raise ValueError("model tags and names must be unique")
    if "baseline_v22" in args.model_names:
        raise ValueError("baseline_v22 is reserved")
    for suffix in (".json", ".csv", ".cases.csv", ".md", ".png", ".pdf"):
        if args.output.with_suffix(suffix).exists():
            raise FileExistsError(f"refusing to overwrite {args.output.with_suffix(suffix)}")

    frames = {
        tag: load_parts(args.parts_root, tag, args.case_min, args.case_max)
        for tag in args.tags
    }
    first = frames[args.tags[0]]
    key = first[["case", "input_index"]].to_numpy()
    reference = first[["R_blend_truth", "prediction_baseline"]].to_numpy(float)
    for tag in args.tags[1:]:
        frame = frames[tag]
        if not np.array_equal(key, frame[["case", "input_index"]].to_numpy()):
            raise RuntimeError(f"{tag}: anchor keys differ from first model")
        if not np.array_equal(
            reference,
            frame[["R_blend_truth", "prediction_baseline"]].to_numpy(float),
        ):
            raise RuntimeError(f"{tag}: coherent truth or baseline differs")

    combined = first[[
        "case", "input_index", "R_blend_truth", "prediction_baseline"
    ]].copy()
    model_columns = {"baseline_v22": "prediction_baseline"}
    for tag, name in zip(args.tags, args.model_names):
        model_columns[name] = f"prediction_{name}"
        combined[model_columns[name]] = frames[tag].prediction_model.to_numpy(float)

    payload: dict[str, Any] = {
        "design": {
            "source": "g=0.02 coherent-neighbour anchor simulations",
            "cases": [args.case_min, args.case_max],
            "case_is_uncertainty_unit": True,
            "tail_definition": "frozen V2.2 scene prediction > 0.1",
            "tail_threshold": TAIL_THRESHOLD,
            "tail_selection_uses_new_model_or_truth": False,
            "model_strengths_selected_on_half_shear_only": True,
            "anchor_truth_used_for_training_or_strength_selection": False,
            "anchor_cases_previously_inspected": bool(args.previously_inspected),
            "constgold_opened": False,
        },
        "model_artifacts": {},
        "models": {},
    }
    for tag in args.tags:
        model_path = args.model_dir / f"regression_model_{tag}.json"
        metadata_path = args.model_dir / f"emulator_metadata_{tag}.json"
        if not model_path.is_file() or not metadata_path.is_file():
            raise FileNotFoundError(f"missing model artifact for {tag}")
        payload["model_artifacts"][tag] = {
            "model": str(model_path),
            "model_sha256": file_sha256(model_path),
            "metadata": str(metadata_path),
            "metadata_sha256": file_sha256(metadata_path),
        }

    rows: list[dict[str, Any]] = []
    case_rows: list[pd.DataFrame] = []
    for model_name, prediction_column in model_columns.items():
        payload["models"][model_name] = {}
        for group_name, (case_min, case_max) in BLOCKS.items():
            payload["models"][model_name][group_name] = {}
            for subset in SUBSETS:
                summary, by_case = summarize_slice(
                    combined,
                    prediction_column,
                    case_min,
                    case_max,
                    subset_mask(combined, subset),
                )
                payload["models"][model_name][group_name][subset] = summary
                gap = summary["model_truth_minus_prediction"]
                base_gap = summary["baseline_truth_minus_prediction"]
                rows.append({
                    "model": model_name,
                    "group": group_name,
                    "subset": subset,
                    "n_anchors": summary["n_anchors"],
                    "gap": gap["mean"],
                    "gap_sem": gap["case_sem"],
                    "baseline_gap": base_gap["mean"],
                    "baseline_gap_sem": base_gap["case_sem"],
                    "model_minus_baseline": summary["model_minus_baseline_prediction"]["mean"],
                    "model_minus_baseline_sem": summary["model_minus_baseline_prediction"]["case_sem"],
                    "fraction_of_baseline_gap_removed": summary["fraction_of_baseline_gap_removed"],
                    "gap_consistent_with_zero_at_2sem": summary["model_gap_consistent_with_zero_at_2sem"],
                })
                by_case.insert(0, "subset", subset)
                by_case.insert(0, "group", group_name)
                by_case.insert(0, "model", model_name)
                case_rows.append(by_case)

    table = pd.DataFrame(rows)
    cases_table = pd.concat(case_rows, ignore_index=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    strict_json(args.output.with_suffix(".json"), payload)
    table.to_csv(args.output.with_suffix(".csv"), index=False)
    cases_table.to_csv(args.output.with_suffix(".cases.csv"), index=False)
    make_plot(args.output, table)

    lines = [
        "# Weak response-weight transfer to coherent anchors",
        "",
        "Positive gaps mean underprediction.  The tail is selected only from the frozen V2.2 "
        "scene prediction (`>0.1`), and errors are one SEM across rendered cases.",
        "",
        "| model | all c400--899 | tail c400--899 | tail c700--899 |",
        "|---|---:|---:|---:|",
    ]
    for model_name in model_columns:
        values = []
        for group_name, subset in (
            ("all_c400_899", "all"),
            ("all_c400_899", "tail"),
            ("c700_899", "tail"),
        ):
            item = payload["models"][model_name][group_name][subset][
                "model_truth_minus_prediction"
            ]
            values.append(f"{item['mean']:+.6f} +- {item['case_sem']:.6f}")
        lines.append(f"| {model_name} | " + " | ".join(values) + " |")
    lines.extend([
        "",
        "No coherent-anchor label was used to train either model or choose its loss strength. "
        "These anchor cases had been inspected in earlier diagnostics, so this is a direct "
        "transfer check rather than a newly blind validation population.",
    ])
    args.output.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "rows": int(len(combined)),
        "tail_rows": int((combined.prediction_baseline > TAIL_THRESHOLD).sum()),
    }, indent=2, sort_keys=True))
    print("ANCHOR_RESPONSE_WEAK_WEIGHT_TRANSFER_ANALYSIS_DONE", flush=True)


if __name__ == "__main__":
    main()
