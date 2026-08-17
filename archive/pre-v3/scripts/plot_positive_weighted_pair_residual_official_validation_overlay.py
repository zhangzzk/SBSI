#!/usr/bin/env python3
"""Overlay four pair-residual curves on one official random-split arm.

The population is either the random 80% fitting subset or the random 20%
validation subset inside half-shear cases 40--199, identified exactly by the
stored ``official_train`` mask in the frozen OOF cache.  This is the split used
while training the pair emulators, not the independent external cases 20--39.

An optional V2.1 primary-domain intersection applies the same true-property
cut used by the older V2.2 calibration figure: primary ``Re > 0.5`` arcsec and
``sn_true(mag, Re) > 10``.  Secondaries are never cut.  Every curve uses 20
equal-population bins of its own frozen prediction.  The label is never used
for binning.  Constgold and coherent-anchor products are never read.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from scripts.plot_positive_weighted_pair_residual_validation import (
    flatten,
    score_model,
    sha256,
)
from scripts.plot_positive_weighted_pair_residual_validation_overlay import (
    MODEL_SPECS,
    plot_overlay,
    summarize_model,
)
from scripts.plot_v22_emulator_label_calibration import json_clean
from sbs_shear.domain import in_domain, metadata as v21_domain_metadata


def v21_primary_mask(
    raw: np.ndarray,
    raw_features: list[str],
) -> np.ndarray:
    """Return the V2.1 mask from cached primary truth columns only."""
    required = ("Re_input_p", "r_input_p")
    missing = [name for name in required if name not in raw_features]
    if missing:
        raise KeyError(f"cached raw features lack {missing}")
    if raw.ndim != 2 or raw.shape[1] != len(raw_features):
        raise ValueError("raw feature matrix does not match raw_features")
    return in_domain(
        raw[:, raw_features.index("r_input_p")],
        raw[:, raw_features.index("Re_input_p")],
    )


def official_split_mask(
    official_train: np.ndarray,
    split_name: str,
) -> np.ndarray:
    """Select exactly one arm of the stored official random row split."""
    official_train = np.asarray(official_train, dtype=bool)
    if official_train.ndim != 1:
        raise ValueError("official_train must be one-dimensional")
    if split_name == "training":
        return official_train
    if split_name == "validation":
        return ~official_train
    raise ValueError(f"unknown official split {split_name!r}")


def load_official_split(
    cache: Path,
    case_min: int,
    case_max: int,
    split_name: str = "validation",
    v21_primary_domain: bool = False,
    v21_mask_path: Path | None = None,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[str, Any],
    dict[str, Any],
]:
    """Load one exact internal random-split arm from cases 40--199."""
    metadata_path = cache / "metadata.json"
    with metadata_path.open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    if metadata["fit_case_window"] != [case_min, case_max]:
        raise RuntimeError("cache fitting window is not cases 40--199")

    arrays = metadata["arrays"]
    case_all = np.load(cache / arrays["case"], mmap_mode="r")
    start = int(np.searchsorted(case_all, case_min, side="left"))
    stop = int(np.searchsorted(case_all, case_max, side="right"))
    official_train = np.load(
        cache / arrays["official_train"], mmap_mode="r"
    )
    window_train = np.asarray(official_train[start:stop], dtype=bool)
    local_index = np.flatnonzero(
        official_split_mask(window_train, split_name)
    ).astype(np.int64)
    index = local_index + start

    split = metadata["official_random_row_split"]
    expected_validation = int(split["validation_rows"])
    expected_train = int(split["train_rows"])
    expected_selected = (
        expected_train if split_name == "training" else expected_validation
    )
    if len(index) != expected_selected:
        raise RuntimeError(
            f"selected {len(index):,} {split_name} rows; expected "
            f"{expected_selected:,}"
        )
    if int(window_train.sum()) != expected_train:
        raise RuntimeError("official training-row count disagrees with metadata")
    if len(window_train) != expected_train + expected_validation:
        raise RuntimeError("official split does not cover the fitting window")

    domain_details = None
    rows_before_domain = len(index)
    if v21_primary_domain:
        if v21_mask_path is not None:
            mask_path = Path(v21_mask_path).resolve()
            exact_all = np.load(mask_path, mmap_mode="r")
            if exact_all.dtype != np.bool_ or exact_all.shape != window_train.shape:
                raise RuntimeError("exact V2.1 mask shape or dtype is wrong")
            keep = np.asarray(exact_all[local_index], dtype=bool)
            mask_metadata_path = mask_path.parent / "metadata.json"
            with mask_metadata_path.open(encoding="utf-8") as handle:
                mask_metadata = json.load(handle)
            if mask_metadata["cache"] != str(cache.resolve()):
                raise RuntimeError("exact V2.1 mask belongs to another cache")
            if mask_metadata["fit_case_window"] != [case_min, case_max]:
                raise RuntimeError("exact V2.1 mask uses another case window")
            domain_details = dict(mask_metadata["domain"])
            domain_details.update({
                "truth_precision": mask_metadata["truth_precision"],
                "mask": str(mask_path),
                "mask_metadata": str(mask_metadata_path),
                "mask_metadata_sha256": sha256(mask_metadata_path),
            })
        else:
            raw_path = cache / arrays["x_raw"]
            raw_all = np.load(raw_path, mmap_mode="r")
            raw_features = list(metadata["raw_features"])
            primary_columns = np.column_stack([
                np.asarray(
                    raw_all[index, raw_features.index("Re_input_p")],
                    dtype=np.float64,
                ),
                np.asarray(
                    raw_all[index, raw_features.index("r_input_p")],
                    dtype=np.float64,
                ),
            ])
            keep = v21_primary_mask(
                primary_columns,
                ["Re_input_p", "r_input_p"],
            )
            domain_details = v21_domain_metadata()
            domain_details.update({
                "truth_precision": "cached float32",
                "raw_feature_file": str(raw_path),
            })
        index = index[keep]
        domain_details.update({
            "applied_to": "primary only",
            "secondary_domain_cut": False,
            "n_rows_before_domain": int(rows_before_domain),
            "n_rows_after_domain": int(len(index)),
            "n_rows_removed": int(rows_before_domain - len(index)),
            "keep_fraction": float(len(index) / rows_before_domain),
        })

    case = np.asarray(case_all[index], dtype=np.int16)
    label = np.asarray(
        np.load(cache / arrays["label"], mmap_mode="r")[index],
        dtype=np.float64,
    )
    features = np.asarray(
        np.load(cache / arrays["x_scaled"], mmap_mode="r")[index],
        dtype=np.float32,
    )
    baseline_path = cache / arrays["v22_prediction"]
    baseline = np.asarray(
        np.load(baseline_path, mmap_mode="r")[index], dtype=np.float64
    )

    if set(np.unique(case).tolist()) != set(range(case_min, case_max + 1)):
        raise RuntimeError(f"official {split_name} subset lacks at least one case")
    n_selected = len(index)
    if features.shape != (n_selected, len(metadata["model_features"])):
        raise RuntimeError("official split feature matrix has the wrong shape")
    for name, values in (
        ("label", label),
        ("features", features),
        ("baseline", baseline),
    ):
        if not np.isfinite(values).all():
            raise RuntimeError(f"official {split_name} {name} is non-finite")

    provenance = {
        "kind": f"frozen_prediction_on_official_random_{split_name}_rows",
        "tag": metadata["source_tag"],
        "prediction_file": str(baseline_path),
        "official_train_file": str(cache / arrays["official_train"]),
        "cache_metadata": str(metadata_path),
        "source_model": metadata["source_model"],
        "source_model_sha256": metadata["source_model_sha256"],
        "source_metadata": metadata["source_metadata"],
        "source_metadata_sha256": metadata["source_metadata_sha256"],
    }
    selection = {
        "case_window": [case_min, case_max],
        "split_name": split_name,
        "official_train_value": bool(split_name == "training"),
        "n_rows_before_domain": expected_selected,
        "n_rows": n_selected,
        "n_official_validation_rows_before_domain": expected_validation,
        "n_official_training_rows_before_domain": expected_train,
        "random_state": int(split["random_state"]),
        "test_size": float(split["test_size"]),
        "n_cases": int(case_max - case_min + 1),
        "minimum_rows_per_case": int(
            np.bincount(case.astype(np.int64) - case_min).min()
        ),
        "maximum_rows_per_case": int(
            np.bincount(case.astype(np.int64) - case_min).max()
        ),
        "v21_primary_domain": bool(v21_primary_domain),
        "v21_domain": domain_details,
    }
    return case, label, features, baseline, metadata, {
        "baseline_provenance": provenance,
        "selection": selection,
    }


def markdown(payload: dict[str, Any]) -> str:
    v21_applied = payload["design"]["v21_primary_domain"]
    split_name = payload["design"]["official_split"]
    split_fraction = "80%" if split_name == "training" else "20%"
    population_note = (
        f"The {split_name} rows are additionally intersected with the V2.1 "
        "primary domain: true Re > 0.5 arcsec and expected true S/N > 10. "
        "No secondary cut is applied."
        if v21_applied else
        "No V2.1 primary-domain intersection is applied."
    )
    lines = [
        f"# Four-model pair calibration on official {split_name} rows",
        "",
        "This is the one-panel V2.2 / alpha=0.035 / 0.05 / 0.10 overlay on "
        f"the exact random {split_fraction} {split_name} subset within half-shear "
        "cases 40--199. This is the internal model split, not the independent "
        "external cases 20--39. Each curve uses 20 equal-population bins of its own "
        "prediction; labels do not enter the binning.",
        "",
        population_note,
        "",
        f"- {split_name.capitalize()} pairs: `{payload['n_pairs']:,}`.",
        f"- Rendered cases: `{payload['n_cases']}`.",
        f"- Split random state: `{payload['design']['random_state']}`.",
        "- Constgold opened: `false`.",
        "- Coherent-anchor truth opened: `false`.",
        "",
        "| model | global label - prediction | lowest-prediction 5% | highest-prediction 5% |",
        "|---|---:|---:|---:|",
    ]
    for model in payload["models"]:
        global_residual = model["global_pair_label_minus_prediction"]
        low = model["bins"][0]["label_minus_prediction"]
        high = model["bins"][-1]["label_minus_prediction"]
        lines.append(
            f"| {model['display']} "
            f"| {global_residual['mean']:+.6f} +- {global_residual['case_sem']:.6f} "
            f"| {low['mean']:+.6f} +- {low['case_sem']:.6f} "
            f"| {high['mean']:+.6f} +- {high['case_sem']:.6f} |"
        )
    lines.extend([
        "",
        "Errors are paired SEMs across the 160 rendered cases. The gray "
        "backdrop is the V2.2 prediction density only.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--case-min", type=int, default=40)
    parser.add_argument("--case-max", type=int, default=199)
    parser.add_argument("--n-bins", type=int, default=20)
    parser.add_argument(
        "--threads",
        type=int,
        default=int(os.environ.get("SLURM_CPUS_PER_TASK", "4")),
    )
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument(
        "--official-split",
        choices=("training", "validation"),
        default="validation",
        help="arm of the stored random 80/20 row split to plot",
    )
    parser.add_argument(
        "--v21-primary-domain",
        action="store_true",
        help=(
            "intersect the selected official rows with the V2.1 primary cut "
            "(true Re > 0.5 arcsec and expected true S/N > 10)"
        ),
    )
    parser.add_argument(
        "--v21-mask",
        help=(
            "optional exact float64 V2.1 mask aligned to cached fitting rows; "
            "requires --v21-primary-domain"
        ),
    )
    args = parser.parse_args()
    if (args.case_min, args.case_max) != (40, 199):
        raise ValueError("official fitting window is fixed to cases 40--199")
    if args.n_bins < 2 or args.threads < 1:
        raise ValueError("n-bins and threads must be positive")
    if args.v21_mask and not args.v21_primary_domain:
        raise ValueError("--v21-mask requires --v21-primary-domain")

    stem = Path(args.output_prefix)
    outputs = [
        stem.with_suffix(f".{suffix}")
        for suffix in ("json", "csv", "md", "png", "pdf")
    ]
    existing = [str(path) for path in outputs if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite existing outputs: {existing}")
    stem.parent.mkdir(parents=True, exist_ok=True)

    cache = Path(args.cache)
    (
        case,
        label,
        features,
        baseline,
        cache_metadata,
        population_metadata,
    ) = load_official_split(
        cache,
        args.case_min,
        args.case_max,
        split_name=args.official_split,
        v21_primary_domain=args.v21_primary_domain,
        v21_mask_path=Path(args.v21_mask) if args.v21_mask else None,
    )

    models: list[dict[str, Any]] = []
    for spec in MODEL_SPECS:
        print(f"scoring {spec['display']} ({spec['tag']})", flush=True)
        if spec["cached"]:
            prediction = baseline
            provenance = population_metadata["baseline_provenance"]
        else:
            prediction, provenance = score_model(
                spec["tag"], features, args.threads
            )
            if provenance["features"] != cache_metadata["model_features"]:
                raise RuntimeError(
                    f"cached feature order does not match {spec['tag']}"
                )
        models.append(summarize_model(
            case,
            label,
            prediction,
            spec,
            provenance,
            None,
            args.n_bins,
            args.case_min,
            args.case_max,
        ))
        if not spec["cached"]:
            del prediction

    if len({model["n_pairs"] for model in models}) != 1:
        raise RuntimeError(f"models do not share the official {args.official_split} rows")
    selection = population_metadata["selection"]
    payload = json_clean({
        "title": f"Four-model calibration on official {args.official_split} rows",
        "design": {
            "source": "frozen half-shear response-pair cache",
            "cache": str(cache.resolve()),
            "cache_source_catalogue": cache_metadata["source_catalogue"],
            "case_window": [args.case_min, args.case_max],
            "population": (
                f"official random {args.official_split} rows inside fitting cases, "
                "intersected with the V2.1 primary domain"
                if args.v21_primary_domain else
                f"official random {args.official_split} rows inside fitting cases"
            ),
            "official_split": args.official_split,
            "official_train_value": bool(args.official_split == "training"),
            "v21_primary_domain": bool(args.v21_primary_domain),
            "v21_domain": selection["v21_domain"],
            "split_status": (
                "rows used to fit every plotted emulator"
                if args.official_split == "training" else
                "internal validation used during model training; not an external holdout"
            ),
            "random_state": selection["random_state"],
            "test_size": selection["test_size"],
            "n_official_training_rows_before_domain": (
                selection["n_official_training_rows_before_domain"]
            ),
            "n_official_validation_rows_before_domain": (
                selection["n_official_validation_rows_before_domain"]
            ),
            "binning": "20 pooled prediction quantiles separately for each frozen model",
            "labels_used_in_binning": False,
            "residual_definition": "half-shear pair label minus model prediction",
            "uncertainty_unit": "rendered simulation case",
            "histogram": "V2.2 prediction density only",
            "constgold_opened": False,
            "coherent_anchor_truth_opened": False,
        },
        "selection_checks": selection,
        "n_pairs": models[0]["n_pairs"],
        "n_cases": models[0]["n_cases"],
        "models": models,
    })

    with stem.with_suffix(".json").open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    flatten(models).to_csv(stem.with_suffix(".csv"), index=False)
    stem.with_suffix(".md").write_text(markdown(payload), encoding="utf-8")
    plot_overlay(
        models,
        stem,
        suptitle=(
            f"Half-shear official {args.official_split} pairs"
            + (" in the V2.1 primary domain" if args.v21_primary_domain else "")
            + " (cases 40–199): "
            "residual versus prediction\n"
            "20 equal-population bins per model; error bars are one SEM "
            "across 160 rendered cases"
        ),
    )
    print(markdown(payload), flush=True)


if __name__ == "__main__":
    main()
