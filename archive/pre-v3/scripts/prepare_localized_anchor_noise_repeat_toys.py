"""Freeze 20 outcome-blind, typical panel-E-tail scenes for noise repeats.

The diagnostic panel is deliberately balanced between anchors whose dominant
secondary is compact (Re < 0.4 arcsec) and the noncompact complement.  Within
each stratum, scenes are medoids of ten clusters in rank-scaled latent/model
feature space after excluding marginal 2-percent tails.  No measured response,
residual, or result from the noiseless toy experiment enters the selection.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


KEY = ["case", "input_index"]
SELECTION_FEATURES = [
    "scene_prediction",
    "primary_mag",
    "log10_primary_size",
    "log10_primary_sersic_n",
    "dominant_secondary_mag",
    "log10_dominant_secondary_size",
    "log10_dominant_secondary_sersic_n",
    "dominant_distance",
    "log10_flux_ratio",
    "top_abs_fraction",
    "log1p_n_pairs",
]


def rank_design(frame: pd.DataFrame, feature_columns: list[str],
                central_quantile: float) -> tuple[pd.DataFrame, np.ndarray]:
    """Return the central candidates and their within-stratum percentile ranks."""
    if not 0.0 <= central_quantile < 0.5:
        raise ValueError("central_quantile must be in [0, 0.5)")
    if missing := set(feature_columns) - set(frame):
        raise KeyError(f"selection frame lacks {sorted(missing)}")
    finite = np.isfinite(frame[feature_columns].to_numpy(float)).all(axis=1)
    local = frame.loc[finite].copy()
    raw_ranks = local[feature_columns].rank(method="average")
    # Mid-percentile ranks make the two marginal exclusions symmetric: with
    # 100 unique values and q=0.02, exactly values 0--1 and 98--99 are removed.
    ranks = (raw_ranks - 0.5) / len(local)
    central = (
        (ranks >= float(central_quantile))
        & (ranks <= 1.0 - float(central_quantile))
    ).all(axis=1)
    local = local.loc[central].copy()
    ranks = ranks.loc[central]
    if local.empty:
        raise RuntimeError("central typical-scene candidate pool is empty")
    return local, ranks.to_numpy(float)


def select_stratum(frame: pd.DataFrame, feature_columns: list[str], n_select: int,
                    central_quantile: float, seed: int,
                    excluded_cases: set[int]) -> tuple[pd.DataFrame, dict]:
    """Select cluster medoids while keeping catalogue cases distinct."""
    candidates, design = rank_design(frame, feature_columns, central_quantile)
    if len(candidates) < n_select:
        raise RuntimeError(
            f"only {len(candidates)} central candidates for {n_select} selections"
        )
    fit = KMeans(
        n_clusters=int(n_select), random_state=int(seed), n_init=50,
        algorithm="lloyd",
    ).fit(design)
    candidates = candidates.copy()
    candidates["selection_cluster"] = fit.labels_.astype(np.int16)
    candidates["selection_distance"] = np.sqrt(np.sum(
        (design - fit.cluster_centers_[fit.labels_]) ** 2, axis=1,
    ))

    selected = []
    used_cases = set(int(value) for value in excluded_cases)
    cluster_sizes = {}
    for cluster in range(n_select):
        local = candidates.loc[candidates.selection_cluster == cluster].copy()
        cluster_sizes[str(cluster)] = int(len(local))
        local = local.sort_values(
            ["selection_distance", *KEY], kind="mergesort",
        )
        available = local.loc[~local.case.astype(int).isin(used_cases)]
        if available.empty:
            raise RuntimeError(
                f"cluster {cluster} has no candidate from an unused catalogue case"
            )
        chosen = available.iloc[0]
        selected.append(chosen)
        used_cases.add(int(chosen.case))
    output = pd.DataFrame(selected).sort_values(
        "selection_cluster", kind="mergesort"
    ).reset_index(drop=True)
    audit = {
        "n_input": int(len(frame)),
        "n_central_candidates": int(len(candidates)),
        "cluster_sizes": cluster_sizes,
        "selected_cases": [int(value) for value in output.case],
    }
    return output, audit


def select_typical_scenes(frame: pd.DataFrame, n_per_stratum: int = 10,
                          central_quantile: float = 0.02,
                          seed: int = 20260813) -> tuple[pd.DataFrame, dict]:
    """Return balanced compact/noncompact feature-space medoids."""
    required = {*KEY, "compact_dominant_secondary", "n_deployed_pairs",
                "dominant_secondary_size", *SELECTION_FEATURES}
    if missing := required - set(frame):
        raise KeyError(f"joined tail frame lacks {sorted(missing)}")
    chosen = []
    audits = {}
    used_cases: set[int] = set()
    for offset, (name, compact) in enumerate((
        ("compact", True), ("noncompact", False),
    )):
        local = frame.loc[
            frame.compact_dominant_secondary.astype(bool) == compact
        ].copy()
        selected, audit = select_stratum(
            local, SELECTION_FEATURES, n_per_stratum, central_quantile,
            seed + offset, used_cases,
        )
        selected["selection_stratum"] = name
        chosen.append(selected)
        audits[name] = audit
        used_cases.update(int(value) for value in selected.case)
    output = pd.concat(chosen, ignore_index=True)
    output.insert(0, "scene_id", np.arange(len(output), dtype=np.int16))
    if output.duplicated(KEY).any() or output.case.duplicated().any():
        raise RuntimeError("typical-scene selection did not preserve unique keys/cases")
    return output, audits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tail-manifest", required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--n-per-stratum", type=int, default=10)
    parser.add_argument("--central-quantile", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--output-feather", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()
    for path in (args.output_feather, args.output_json):
        if os.path.exists(path):
            raise FileExistsError(f"refusing existing output {path}")

    tail = pd.read_feather(args.tail_manifest)
    features = pd.read_feather(
        args.features, columns=[*KEY, *SELECTION_FEATURES],
    )
    joined = tail.merge(
        features, on=KEY, how="left", validate="one_to_one",
        suffixes=("", "_feature"),
    )
    feature_prediction = joined.pop("scene_prediction_feature").to_numpy(float)
    prediction_error = float(np.max(np.abs(
        joined.scene_prediction.to_numpy(float) - feature_prediction
    )))
    if prediction_error > 2.0e-12:
        raise RuntimeError(
            f"tail/feature scene predictions differ by {prediction_error:.3g}"
        )
    selected, audits = select_typical_scenes(
        joined, args.n_per_stratum, args.central_quantile, args.seed,
    )
    keep = [
        "scene_id", *KEY, "selection_stratum", "selection_cluster",
        "selection_distance", "compact_dominant_secondary",
        "dominant_secondary_size", "n_deployed_pairs", *SELECTION_FEATURES,
    ]
    selected = selected[keep].copy()
    Path(args.output_feather).parent.mkdir(parents=True, exist_ok=True)
    selected.to_feather(args.output_feather)
    payload = {
        "design": (
            "outcome-blind feature-space medoids from the frozen panel-E tail; "
            "ten compact and ten noncompact dominant-secondary scenes"
        ),
        "selection_features": SELECTION_FEATURES,
        "selection_uses_measured_response_or_residual": False,
        "n_per_stratum": int(args.n_per_stratum),
        "central_quantile_excluded_each_side": float(args.central_quantile),
        "kmeans_seed": int(args.seed),
        "n_scenes": int(len(selected)),
        "all_cases_distinct": bool(~selected.case.duplicated().any()),
        "tail_manifest": os.path.abspath(args.tail_manifest),
        "source_features": os.path.abspath(args.features),
        "max_scene_prediction_replay_error": prediction_error,
        "strata": audits,
        "selected_keys": selected[["scene_id", *KEY, "selection_stratum"]]
        .to_dict(orient="records"),
    }
    with open(args.output_json, "x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(selected[[
        "scene_id", *KEY, "selection_stratum", "scene_prediction",
        "n_deployed_pairs", "dominant_secondary_size",
    ]].to_string(index=False))
    print("LOCALIZED_ANCHOR_NOISE_REPEAT_MANIFEST_DONE", flush=True)


if __name__ == "__main__":
    main()
