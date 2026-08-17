#!/usr/bin/env python3
"""Combine retained original anchors and V2-complement anchors by population weight."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


MODEL_SHA = "3cf70b6e74ad382f3ec59c6e8a2d0a5b9b0615d4c4677c7a71dd2344cbf35553"
LABEL = "v2_reweighted_vector_fixed_from_v22_trial15"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def stat(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    if len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("stat requires at least two finite values")
    return {
        "mean": float(values.mean()),
        "case_sd": float(values.std(ddof=1)),
        "case_sem": float(values.std(ddof=1) / np.sqrt(len(values))),
        "n_cases": int(len(values)),
    }


def clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def block_root(prefix: str, case: int) -> Path:
    start = case // 100 * 100
    return Path(f"{prefix}{start}-{start + 99}")


def check_score(frame: pd.DataFrame, case: int, label: str) -> None:
    if frame.empty or frame.case.nunique() != 1 or int(frame.case.iloc[0]) != case:
        raise RuntimeError(f"case {case}: invalid score case coverage")
    if frame.input_index.duplicated().any():
        raise RuntimeError(f"case {case}: duplicate score keys")
    if set(frame.model_tag) != {label} or set(frame.model_sha256) != {MODEL_SHA}:
        raise RuntimeError(f"case {case}: candidate identity drifted")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-parts", required=True)
    parser.add_argument("--complement-parts", required=True)
    parser.add_argument("--original-base-prefix", required=True)
    parser.add_argument("--complement-base-prefix", required=True)
    parser.add_argument("--case-min", type=int, default=400)
    parser.add_argument("--case-max", type=int, default=899)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")

    rows = []
    for case in range(args.case_min, args.case_max + 1):
        old_score_path = Path(args.original_parts) / f"case{case}.feather"
        comp_score_path = Path(args.complement_parts) / f"case{case}.feather"
        old_score = pd.read_feather(old_score_path)
        comp_score = pd.read_feather(comp_score_path)
        check_score(old_score, case, LABEL)
        check_score(comp_score, case, LABEL)
        old_base = block_root(args.original_base_prefix, case)
        comp_base = block_root(args.complement_base_prefix, case)
        old_manifest = pd.read_feather(old_base / f"anchors_case{case}.feather")
        comp_manifest = pd.read_feather(comp_base / f"anchors_case{case}.feather")
        if old_manifest["index"].duplicated().any() or comp_manifest["index"].duplicated().any():
            raise RuntimeError(f"case {case}: duplicate manifest keys")
        count_columns = [
            "n_parent_v2", "n_parent_original_domain", "n_parent_complement"
        ]
        counts = {}
        for column in count_columns:
            values = comp_manifest[column].unique()
            if len(values) != 1:
                raise RuntimeError(f"case {case}: nonconstant {column}")
            counts[column] = int(values[0])
        if counts["n_parent_original_domain"] + counts["n_parent_complement"] != counts["n_parent_v2"]:
            raise RuntimeError(f"case {case}: strata do not partition V2")
        old_weight = counts["n_parent_original_domain"] / len(old_manifest)
        comp_weight = counts["n_parent_complement"] / len(comp_manifest)
        old_effective = old_weight * len(old_score)
        comp_effective = comp_weight * len(comp_score)
        total_effective = old_effective + comp_effective
        if total_effective <= 0:
            raise RuntimeError(f"case {case}: zero effective detected population")

        def mean(frame: pd.DataFrame, column: str) -> float:
            value = float(frame[column].mean())
            if not np.isfinite(value):
                raise RuntimeError(f"case {case}: non-finite {column}")
            return value

        combined = {}
        for column in ("R_blend_truth", "prediction_model", "gap_model"):
            old_mean = mean(old_score, column)
            comp_mean = mean(comp_score, column)
            combined[column] = (
                old_effective * old_mean + comp_effective * comp_mean
            ) / total_effective
            combined[f"original_{column}"] = old_mean
            combined[f"complement_{column}"] = comp_mean
        rows.append({
            "case": case,
            **counts,
            "n_original_anchors": len(old_manifest),
            "n_complement_anchors": len(comp_manifest),
            "n_original_measured": len(old_score),
            "n_complement_measured": len(comp_score),
            "original_inverse_sampling_weight": old_weight,
            "complement_inverse_sampling_weight": comp_weight,
            "estimated_original_detected_population": old_effective,
            "estimated_complement_detected_population": comp_effective,
            **combined,
        })

    case_frame = pd.DataFrame(rows)
    metrics = [
        "R_blend_truth", "prediction_model", "gap_model",
        "original_R_blend_truth", "original_prediction_model", "original_gap_model",
        "complement_R_blend_truth", "complement_prediction_model", "complement_gap_model",
    ]
    payload = {
        "schema_version": 1,
        "kind": "population-weighted full-V2 coherent-anchor supplement",
        "candidate": {"label": LABEL, "sha256": MODEL_SHA},
        "domain": {
            "full_v2": "18 < primary r < 26 and 0.3 < primary Re < 1.5 arcsec",
            "original_stratum": "historical coherent-anchor primary_mask (V2.1 intrinsic domain)",
            "complement_stratum": "full V2 minus original stratum",
        },
        "design": {
            "original_anchor_measurements_retained": True,
            "supplement_rerendered_separately": True,
            "combination": "within each case, expand each measured stratum by n_parent/n_sparse_anchors, then combine; cases are the uncertainty unit",
            "selection_or_refit_on_anchor_truth": False,
        },
        "case_window": [args.case_min, args.case_max],
        "population": {
            "n_parent_v2": int(case_frame.n_parent_v2.sum()),
            "n_parent_original_domain": int(case_frame.n_parent_original_domain.sum()),
            "n_parent_complement": int(case_frame.n_parent_complement.sum()),
            "original_parent_fraction": float(
                case_frame.n_parent_original_domain.sum() / case_frame.n_parent_v2.sum()
            ),
            "n_original_sparse_anchors": int(case_frame.n_original_anchors.sum()),
            "n_complement_sparse_anchors": int(case_frame.n_complement_anchors.sum()),
            "n_original_measured": int(case_frame.n_original_measured.sum()),
            "n_complement_measured": int(case_frame.n_complement_measured.sum()),
        },
        "results": {name: stat(case_frame[name].to_numpy(float)) for name in metrics},
        "artifacts": {
            "case_table": str(output.with_suffix(".cases.csv")),
            "original_parts": str(Path(args.original_parts).resolve()),
            "complement_parts": str(Path(args.complement_parts).resolve()),
        },
    }
    case_frame.to_csv(output.with_suffix(".cases.csv"), index=False)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(clean(payload), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps(clean(payload), indent=2, sort_keys=True), flush=True)
    print("ANCHOR_V2_SUPPLEMENT_ANALYSIS_DONE", flush=True)


if __name__ == "__main__":
    main()
