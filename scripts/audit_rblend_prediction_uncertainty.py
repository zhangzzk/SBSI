#!/usr/bin/env python3
"""Case-bootstrap uncertainty of frozen V3 R_blend mean predictions.

The emulator predicts a scalar response for each primary--secondary pair.  The
quantity entering SBSI closure is instead the sum over pairs for each primary,
averaged over the selected primaries.  This audit preserves that estimand by
accumulating each case's pair-prediction sum and number of distinct primaries,
then resampling whole cases and re-pooling those sufficient statistics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import xgboost as xgb


FEATURES = [
    "Re_input_p_scaled",
    "Re_input_s_scaled",
    "r_input_p_scaled",
    "r_input_s_scaled",
    "sersic_n_input_p",
    "sersic_n_input_s",
    "distance_scaled",
]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-cache", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--constgold-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n-boot", type=int, default=100_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260824)
    return parser.parse_args(argv)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def bootstrap_mean(
    sums: np.ndarray,
    counts: np.ndarray,
    *,
    n_boot: int,
    seed: int,
    chunk: int = 10_000,
) -> dict[str, float | int | list[float]]:
    sums = np.asarray(sums, dtype=np.float64)
    counts = np.asarray(counts, dtype=np.int64)
    if sums.ndim != 1 or counts.shape != sums.shape or len(sums) < 2:
        raise ValueError("case sums/counts must be matching vectors with at least two cases")
    if not np.isfinite(sums).all() or np.any(counts <= 0):
        raise ValueError("case sufficient statistics must be finite with positive counts")
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot, dtype=np.float64)
    for start in range(0, n_boot, chunk):
        stop = min(start + chunk, n_boot)
        index = rng.integers(0, len(sums), size=(stop - start, len(sums)))
        draws[start:stop] = sums[index].sum(axis=1) / counts[index].sum(axis=1)
    mean = float(sums.sum() / counts.sum())
    case_means = sums / counts
    return {
        "n_cases": int(len(sums)),
        "n_objects": int(counts.sum()),
        "mean_prediction": mean,
        "case_mean_sd": float(case_means.std(ddof=1)),
        "case_mean_sem": float(case_means.std(ddof=1) / np.sqrt(len(sums))),
        "case_bootstrap_standard_error": float(draws.std(ddof=1)),
        "case_bootstrap_relative_standard_error": float(draws.std(ddof=1) / abs(mean)),
        "case_bootstrap_95_interval": [
            float(np.quantile(draws, 0.025)),
            float(np.quantile(draws, 0.975)),
        ],
    }


def summarize_windows(
    case_ids: np.ndarray,
    sums: np.ndarray,
    counts: np.ndarray,
    *,
    n_boot: int,
    seed: int,
) -> dict[str, dict[str, float | int | list[float]]]:
    windows = {
        "all_available_cases_0_199": (0, 199),
        "development_cases_0_39": (0, 39),
        "fit_cases_40_199": (40, 199),
    }
    result = {}
    for offset, (name, (first, last)) in enumerate(windows.items()):
        use = (case_ids >= first) & (case_ids <= last)
        if int(use.sum()) != last - first + 1:
            raise RuntimeError(f"{name} does not contain every registered case")
        summary = bootstrap_mean(
            sums[use], counts[use], n_boot=n_boot, seed=seed + offset
        )
        summary["case_range_inclusive"] = [first, last]
        result[name] = summary
    return result


def training_prediction_sufficient_statistics(cache: Path, model: Path):
    with (cache / "metadata.json").open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    arrays = metadata["arrays"]
    case = np.load(cache / arrays["case"], mmap_mode="r")
    primary = np.load(cache / arrays["input_index"], mmap_mode="r")
    x_scaled = np.load(cache / arrays["x_scaled"], mmap_mode="r")
    if len(case) != len(primary) or len(case) != len(x_scaled):
        raise RuntimeError("source-cache arrays differ in length")
    if x_scaled.ndim != 2 or x_scaled.shape[1] != len(FEATURES):
        raise RuntimeError("unexpected source-cache feature shape")
    if metadata.get("model_features") != FEATURES:
        raise RuntimeError("source-cache feature order differs from the frozen model contract")
    standardization = metadata["source_standardization"]
    target_mean = float(standardization["mean"])
    target_scale = float(standardization["std"])

    booster = xgb.Booster()
    booster.load_model(model)
    if booster.feature_names != FEATURES:
        raise RuntimeError("frozen emulator feature names differ from source cache")
    n_trees = int(booster.num_boosted_rounds())

    unique_cases = np.unique(case).astype(np.int64)
    if not np.array_equal(unique_cases, np.arange(200, dtype=np.int64)):
        raise RuntimeError("expected exactly source cases 0--199")
    case_sums = np.empty(len(unique_cases), dtype=np.float64)
    case_counts = np.empty(len(unique_cases), dtype=np.int64)
    case_pair_counts = np.empty(len(unique_cases), dtype=np.int64)

    for position, case_id in enumerate(unique_cases):
        indices = np.flatnonzero(case == case_id)
        if len(indices) == 0 or indices[-1] - indices[0] + 1 != len(indices):
            raise RuntimeError(f"case {case_id} rows are not one contiguous block")
        slc = slice(int(indices[0]), int(indices[-1]) + 1)
        matrix = xgb.DMatrix(np.asarray(x_scaled[slc], dtype=np.float32), feature_names=FEATURES)
        standardized = booster.predict(matrix, iteration_range=(0, n_trees))
        prediction = standardized.astype(np.float64) * target_scale + target_mean
        if not np.isfinite(prediction).all():
            raise RuntimeError(f"case {case_id} has non-finite predictions")
        case_sums[position] = prediction.sum(dtype=np.float64)
        case_counts[position] = len(np.unique(primary[slc]))
        case_pair_counts[position] = len(prediction)
        print(
            f"case {case_id:03d}: pairs={len(prediction):,} "
            f"primaries={case_counts[position]:,} "
            f"mean_summed_Rblend={case_sums[position] / case_counts[position]:.8f}",
            flush=True,
        )
    return metadata, unique_cases, case_sums, case_counts, case_pair_counts, n_trees


def constgold_sufficient_statistics(path: Path):
    with path.open(encoding="utf-8") as handle:
        result = json.load(handle)
    per_case = result["per_case"]
    case_ids = np.array(sorted(map(int, per_case)), dtype=np.int64)
    counts = np.array([per_case[str(case)]["n_rows"] for case in case_ids], dtype=np.int64)
    means = np.array(
        [per_case[str(case)]["blend_response"] for case in case_ids], dtype=np.float64
    )
    if not np.isclose(means @ counts / counts.sum(), result["blend_response"], rtol=0, atol=1e-14):
        raise RuntimeError("ConstGold per-case predictions do not reproduce the pooled mean")
    return result, case_ids, means * counts, counts


def main(argv=None):
    args = parse_args(argv)
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    if args.n_boot < 2:
        raise ValueError("--n-boot must be at least 2")

    metadata, cases, sums, counts, pair_counts, n_trees = (
        training_prediction_sufficient_statistics(args.source_cache, args.model)
    )
    training = summarize_windows(
        cases,
        sums,
        counts,
        n_boot=args.n_boot,
        seed=args.bootstrap_seed,
    )
    for name, summary in training.items():
        first, last = summary["case_range_inclusive"]
        use = (cases >= first) & (cases <= last)
        summary["n_pairs"] = int(pair_counts[use].sum())

    constgold_source, cg_cases, cg_sums, cg_counts = constgold_sufficient_statistics(
        args.constgold_result
    )
    constgold = bootstrap_mean(
        cg_sums,
        cg_counts,
        n_boot=args.n_boot,
        seed=args.bootstrap_seed + 10,
    )
    constgold["case_range_inclusive"] = [int(cg_cases.min()), int(cg_cases.max())]
    constgold["population"] = "exact registered V3 ConstGold evaluation rows"

    payload = {
        "estimand": (
            "mean per-primary R_blend: sum frozen emulator pair predictions per primary, "
            "then pool over primaries; uncertainty resamples whole simulation cases"
        ),
        "uncertainty_scope": (
            "case-sampling uncertainty conditional on one frozen emulator checkpoint; "
            "does not include training-seed or model-form uncertainty"
        ),
        "bootstrap_replicates": int(args.n_boot),
        "bootstrap_seed": int(args.bootstrap_seed),
        "model": str(args.model),
        "model_sha256": sha256(args.model),
        "model_trees": n_trees,
        "source_cache": str(args.source_cache),
        "source_cache_metadata_sha256": sha256(args.source_cache / "metadata.json"),
        "source_cache_case_window": metadata["case_window"],
        "source_cache_official_random_row_split": metadata["official_random_row_split"],
        "training_catalogue_predictions": training,
        "constgold": constgold,
        "constgold_source": str(args.constgold_result),
        "constgold_source_sha256": sha256(args.constgold_result),
        "constgold_pooled_mean_guard": float(constgold_source["blend_response"]),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"training_catalogue_predictions": training, "constgold": constgold}, indent=2))


if __name__ == "__main__":
    main()
