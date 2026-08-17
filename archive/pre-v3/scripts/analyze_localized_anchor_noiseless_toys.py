"""Combine localized-anchor noiseless toy cases and quantify the gap decomposition."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


KEY = ["case", "input_index"]


def add_decomposition(frame: pd.DataFrame) -> pd.DataFrame:
    """Add matched model, individual, and coherent residual identities."""
    out = frame.copy()
    for suffix, coherent, individual in (
        ("g1", "R11_coherent", "R11_individual_sum"),
        ("trace", "R_trace_coherent", "R_trace_individual_sum"),
    ):
        out[f"additivity_gap_{suffix}"] = out[coherent] - out[individual]
        out[f"individual_truth_minus_model_{suffix}"] = (
            out[individual] - out.R_model_sum
        )
        out[f"coherent_truth_minus_model_{suffix}"] = (
            out[coherent] - out.R_model_sum
        )
        out[f"decomposition_replay_{suffix}"] = (
            out[f"coherent_truth_minus_model_{suffix}"]
            - out[f"additivity_gap_{suffix}"]
            - out[f"individual_truth_minus_model_{suffix}"]
        )
    out["toy_minus_original_coherent_g1"] = (
        out.R11_coherent - out.R_original_coherent_g1
    )
    out["toy_minus_original_model_gap_g1"] = (
        out.coherent_truth_minus_model_g1
        - out.original_truth_minus_model_g1
    )
    return out


def finite_stat(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        raise ValueError("statistic needs at least two finite cases")
    sem = float(values.std(ddof=1) / np.sqrt(len(values)))
    mean = float(values.mean())
    return {
        "mean": mean, "case_sem": sem, "n_cases": int(len(values)),
        "t": float(mean / sem) if sem else float("inf"),
        "case_values": values.tolist(),
    }


def case_stat(frame: pd.DataFrame, column: str) -> dict:
    values = frame.groupby("case", sort=True)[column].mean().to_numpy(float)
    return finite_stat(values)


def correlations(frame: pd.DataFrame) -> dict:
    x = frame.original_truth_minus_model_g1.to_numpy(float)
    y = frame.coherent_truth_minus_model_g1.to_numpy(float)
    row_rho = stats.spearmanr(x, y).statistic
    case = frame.groupby("case", sort=True)[[
        "original_truth_minus_model_g1", "coherent_truth_minus_model_g1",
    ]].mean()
    case_rho = stats.spearmanr(
        case.original_truth_minus_model_g1,
        case.coherent_truth_minus_model_g1,
    ).statistic
    return {
        "row_spearman_original_vs_toy": float(row_rho),
        "case_mean_spearman_original_vs_toy": float(case_rho),
    }


def summarize_group(frame: pd.DataFrame) -> dict:
    columns = [
        "R_model_sum", "R_original_coherent_g1",
        "original_truth_minus_model_g1",
        "R11_coherent", "R11_individual_sum",
        "coherent_truth_minus_model_g1",
        "individual_truth_minus_model_g1", "additivity_gap_g1",
        "R_trace_coherent", "R_trace_individual_sum",
        "coherent_truth_minus_model_trace",
        "individual_truth_minus_model_trace", "additivity_gap_trace",
        "toy_minus_original_coherent_g1",
        "toy_minus_original_model_gap_g1",
    ]
    return {
        "n_anchors": int(len(frame)), "n_cases": int(frame.case.nunique()),
        "mean_deployed_pairs": float(frame.n_deployed_pairs.mean()),
        "statistics": {column: case_stat(frame, column) for column in columns},
        "correlations": correlations(frame),
    }


def markdown(payload: dict) -> str:
    lines = [
        "# Localized coherent-anchor noiseless toy decomposition",
        "",
        "The toys use the exact latent primary and exact deployed V2.2 neighbours "
        "of every frozen panel-E-tail anchor. Pixel noise, detection, and measured-"
        "centroid motion are absent. Positions are fixed and the primary is never "
        "sheared.",
        "",
        f"- Selected anchors: `{payload['coverage']['n_manifest']:,}` across "
        f"`{payload['coverage']['n_cases']}` cases.",
        f"- Complete matched toys: `{payload['coverage']['n_success']:,}` "
        f"(`{payload['coverage']['success_fraction']:.2%}`).",
        f"- Shear amplitude: `{payload['design']['g']}`; stamp: "
        f"`{payload['design']['stamp']}` pixels.",
        "",
        "The exact identity is `coherent - model = (coherent - individual sum) "
        "+ (individual sum - model)`. Positive residual means V2.2 underpredicts.",
        "",
        "| population | original coherent-model g1 | toy coherent-model g1 | "
        "toy coherent-individual g1 | toy individual-model g1 | trace coherent-individual |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in (
        "all_tail", "compact_dominant_secondary", "noncompact_dominant_secondary"
    ):
        group = payload["groups"][name]
        s = group["statistics"]
        def cell(column: str) -> str:
            item = s[column]
            return f"{item['mean']:+.6f} ± {item['case_sem']:.6f}"
        lines.append(
            f"| {name.replace('_', ' ')} (n={group['n_anchors']:,}) | "
            f"{cell('original_truth_minus_model_g1')} | "
            f"{cell('coherent_truth_minus_model_g1')} | "
            f"{cell('additivity_gap_g1')} | "
            f"{cell('individual_truth_minus_model_g1')} | "
            f"{cell('additivity_gap_trace')} |"
        )
    lines.extend([
        "",
        "Uncertainties are SEMs across rendered catalogue cases; anchors are not "
        "treated as independent replacements for cases.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--cases", type=int, nargs="+", required=True)
    ap.add_argument("--output-anchor", required=True)
    ap.add_argument("--output-pair", required=True)
    ap.add_argument("--output-json", required=True)
    ap.add_argument("--output-md", required=True)
    ap.add_argument("--min-success-fraction", type=float, default=0.9)
    args = ap.parse_args()
    for path in (
        args.output_anchor, args.output_pair, args.output_json, args.output_md
    ):
        if os.path.exists(path):
            raise FileExistsError(f"refusing existing output {path}")

    input_dir = Path(args.input_dir)
    anchor_parts = []
    pair_parts = []
    audits = []
    for case in args.cases:
        anchor_path = input_dir / f"anchors_case{case}.feather"
        pair_path = input_dir / f"pairs_case{case}.feather"
        json_path = input_dir / f"audit_case{case}.json"
        for path in (anchor_path, pair_path, json_path):
            if not path.exists():
                raise FileNotFoundError(path)
        anchor_parts.append(pd.read_feather(anchor_path))
        pair_parts.append(pd.read_feather(pair_path))
        with open(json_path, encoding="utf-8") as handle:
            audits.append(json.load(handle))
    anchors = pd.concat(anchor_parts, ignore_index=True)
    pairs = pd.concat(pair_parts, ignore_index=True)
    if anchors.duplicated(KEY).any():
        raise RuntimeError("duplicate toy anchor key")
    if pairs.duplicated([*KEY, "secondary_index"]).any():
        raise RuntimeError("duplicate toy pair key")

    manifest = pd.read_feather(args.manifest)
    manifest = manifest.loc[manifest.case.isin(args.cases)].copy()
    expected = manifest[KEY].sort_values(KEY).reset_index(drop=True)
    observed = anchors[KEY].sort_values(KEY).reset_index(drop=True)
    if not expected.equals(observed):
        raise RuntimeError("toy outputs do not cover the exact manifest keys")
    success = anchors.success.astype(bool)
    success_fraction = float(success.mean())
    if success_fraction < args.min_success_fraction:
        raise RuntimeError(
            f"toy success fraction {success_fraction:.2%} below "
            f"{args.min_success_fraction:.2%}"
        )
    complete = add_decomposition(anchors.loc[success].copy())
    replay_columns = ["decomposition_replay_g1", "decomposition_replay_trace"]
    replay_max = float(np.max(np.abs(complete[replay_columns].to_numpy(float))))
    if replay_max > 2.0e-12:
        raise RuntimeError(f"toy decomposition does not close: {replay_max:.3g}")
    expected_pairs = int(complete.n_deployed_pairs.sum())
    if len(pairs) != expected_pairs:
        raise RuntimeError(
            f"pair rows {len(pairs):,} != successful deployed sum "
            f"{expected_pairs:,}"
        )

    groups = {
        "all_tail": np.ones(len(complete), dtype=bool),
        "compact_dominant_secondary": complete.compact_dominant_secondary.to_numpy(bool),
        "noncompact_dominant_secondary": ~complete.compact_dominant_secondary.to_numpy(bool),
    }
    first_audit = audits[0]
    if len({(item["g"], item["stamp"]) for item in audits}) != 1:
        raise RuntimeError("toy shard design settings differ")
    payload = {
        "design": {
            "description": first_audit["design"],
            "g": float(first_audit["g"]), "stamp": int(first_audit["stamp"]),
            "pixel_noise": 0.0, "primary_sheared": False,
            "positions_sheared": False,
            "scene_members": first_audit["scene_members"],
            "primary_estimand": "fixed g1=0.02 matching localized coherent anchors",
            "rotation_control": "0.5*(R11+R22) response trace",
            "uncertainty_unit": "rendered catalogue case",
        },
        "coverage": {
            "n_manifest": int(len(manifest)), "n_cases": int(manifest.case.nunique()),
            "n_success": int(success.sum()), "n_failure": int((~success).sum()),
            "success_fraction": success_fraction,
            "failure_types": {
                str(name): int(count) for name, count in
                anchors.loc[~success, "error_type"].value_counts().items()
            },
            "n_pair_rows": int(len(pairs)),
        },
        "decomposition_replay_max_abs": replay_max,
        "groups": {
            name: summarize_group(complete.loc[mask])
            for name, mask in groups.items()
        },
        "artifacts": {
            "anchor_feather": os.path.abspath(args.output_anchor),
            "pair_feather": os.path.abspath(args.output_pair),
            "report_md": os.path.abspath(args.output_md),
        },
    }
    Path(args.output_anchor).parent.mkdir(parents=True, exist_ok=True)
    complete.to_feather(args.output_anchor)
    pairs.to_feather(args.output_pair)
    with open(args.output_json, "x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    Path(args.output_md).write_text(markdown(payload), encoding="utf-8")
    print(markdown(payload))
    print("LOCALIZED_ANCHOR_NOISELESS_TOY_ANALYSIS_DONE", flush=True)


if __name__ == "__main__":
    main()
