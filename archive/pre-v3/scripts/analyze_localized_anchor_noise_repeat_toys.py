"""Analyze the frozen 20-scene common-noise additivity experiment."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


KEY = ["case", "input_index"]
STAT_COLUMNS = [
    "R11_coherent", "R11_individual_sum", "additivity_gap_g1",
    "coherent_truth_minus_model_g1", "individual_truth_minus_model_g1",
    "R_trace_coherent", "R_trace_individual_sum", "additivity_gap_trace",
    "coherent_truth_minus_model_trace", "individual_truth_minus_model_trace",
]


def draw_stat(values: np.ndarray, confidence: float = 0.95) -> dict:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        raise ValueError("noise statistic requires at least two finite draws")
    mean = float(values.mean())
    sd = float(values.std(ddof=1))
    sem = float(sd / np.sqrt(len(values)))
    critical = float(stats.t.ppf(0.5 + confidence / 2.0, len(values) - 1))
    half = critical * sem
    return {
        "mean": mean, "noise_sd": sd, "noise_sem": sem,
        "n_draws": int(len(values)), "confidence": float(confidence),
        "ci_low": float(mean - half), "ci_high": float(mean + half),
        "ci_halfwidth": float(half),
    }


def scene_summary(draws: pd.DataFrame, row: pd.Series, sesoi: float,
                  target_halfwidth: float) -> tuple[dict, dict[str, dict]]:
    """Summarize technical repeats without promoting them to scene replicates."""
    success = draws.success.astype(bool)
    good = draws.loc[success].copy()
    stats_by_column = {
        column: draw_stat(good[column].to_numpy(float))
        for column in STAT_COLUMNS
    }
    primary = stats_by_column["additivity_gap_trace"]
    required_n = int(np.ceil(
        (1.96 * primary["noise_sd"] / float(target_halfwidth)) ** 2
    ))
    summary = {
        "scene_id": int(row.scene_id), "case": int(row.case),
        "input_index": int(row.input_index),
        "selection_stratum": str(row.selection_stratum),
        "compact_dominant_secondary": bool(row.compact_dominant_secondary),
        "dominant_secondary_size": float(row.dominant_secondary_size),
        "n_deployed_pairs": int(row.n_deployed_pairs),
        "R_model_sum": float(row.scene_prediction),
        "n_requested": int(len(draws)), "n_success": int(success.sum()),
        "success_fraction": float(success.mean()),
        "additivity_gap_trace": primary["mean"],
        "additivity_gap_trace_noise_sem": primary["noise_sem"],
        "additivity_gap_trace_ci_low": primary["ci_low"],
        "additivity_gap_trace_ci_high": primary["ci_high"],
        "additivity_gap_trace_ci_halfwidth": primary["ci_halfwidth"],
        "equivalent_within_sesoi": bool(
            primary["ci_low"] > -float(sesoi)
            and primary["ci_high"] < float(sesoi)
        ),
        "significantly_positive": bool(primary["ci_low"] > 0.0),
        "significantly_negative": bool(primary["ci_high"] < 0.0),
        "target_ci_halfwidth_met": bool(
            primary["ci_halfwidth"] <= float(target_halfwidth)
        ),
        "estimated_draws_for_target_halfwidth": max(2, required_n),
    }
    for column in (
        "coherent_truth_minus_model_trace",
        "individual_truth_minus_model_trace",
        "additivity_gap_g1",
    ):
        summary[column] = stats_by_column[column]["mean"]
        summary[f"{column}_noise_sem"] = stats_by_column[column]["noise_sem"]
    return summary, stats_by_column


def scene_group_stat(frame: pd.DataFrame, column: str) -> dict:
    values = frame[column].to_numpy(float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        raise ValueError("scene group needs at least two fixed scenes")
    return {
        "mean_of_scene_means": float(values.mean()),
        "scene_sd": float(values.std(ddof=1)),
        "scene_sem_descriptive": float(values.std(ddof=1) / np.sqrt(len(values))),
        "n_scenes": int(len(values)),
    }


def summarize_group(frame: pd.DataFrame) -> dict:
    return {
        "n_scenes": int(len(frame)),
        "additivity_gap_trace": scene_group_stat(frame, "additivity_gap_trace"),
        "coherent_truth_minus_model_trace": scene_group_stat(
            frame, "coherent_truth_minus_model_trace"
        ),
        "individual_truth_minus_model_trace": scene_group_stat(
            frame, "individual_truth_minus_model_trace"
        ),
        "median_primary_noise_sem": float(
            frame.additivity_gap_trace_noise_sem.median()
        ),
        "max_primary_ci_halfwidth": float(
            frame.additivity_gap_trace_ci_halfwidth.max()
        ),
        "n_equivalent_within_sesoi": int(frame.equivalent_within_sesoi.sum()),
        "n_significantly_positive": int(frame.significantly_positive.sum()),
        "n_significantly_negative": int(frame.significantly_negative.sum()),
        "n_target_precision_met": int(frame.target_ci_halfwidth_met.sum()),
    }


def markdown(payload: dict, scenes: pd.DataFrame) -> str:
    design = payload["design"]
    coverage = payload["coverage"]
    lines = [
        "# Localized anchor common-noise repetition test",
        "",
        "Twenty outcome-blind feature-space medoids from the frozen panel-E tail "
        "were tested: ten compact and ten noncompact dominant-secondary scenes. "
        "Each noise realization is a matched block across coherent, individual-"
        "neighbour, antithetic, and g1/g2 arms.",
        "",
        f"- Setup: `g={design['g']}`, `{design['stamp']}` pixels, Gaussian pixel "
        f"RMS `{design['pixel_rms']}`, `{design['nreal']}` requested draws per scene.",
        f"- Successful matched blocks: `{coverage['n_success']:,}` / "
        f"`{coverage['n_requested']:,}` (`{coverage['success_fraction']:.2%}`).",
        f"- Primary practical-null band: `|coherent-individual trace| < "
        f"{design['sesoi']:.3f}`; target 95% CI half-width "
        f"`<= {design['target_ci_halfwidth']:.3f}`.",
        "",
        "Repeated noise draws are technical replicates for one scene. The table "
        "therefore reports within-scene noise SEMs; it does not treat 400 draws as "
        "400 independent anchor scenes.",
        "",
        "| id | stratum | case:index | Nnbr | noiseless add trace | noisy add trace | "
        "coherent-model trace | individual-model trace | 95% status |",
        "|---:|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in scenes.sort_values("scene_id").itertuples(index=False):
        if row.equivalent_within_sesoi:
            status = "equivalent"
        elif row.significantly_positive:
            status = "positive"
        elif row.significantly_negative:
            status = "negative"
        else:
            status = "inconclusive"
        lines.append(
            f"| {row.scene_id} | {row.selection_stratum} | "
            f"{row.case}:{row.input_index} | {row.n_deployed_pairs} | "
            f"{row.noiseless_additivity_gap_trace:+.5f} | "
            f"{row.additivity_gap_trace:+.5f} ± "
            f"{row.additivity_gap_trace_noise_sem:.5f} | "
            f"{row.coherent_truth_minus_model_trace:+.5f} ± "
            f"{row.coherent_truth_minus_model_trace_noise_sem:.5f} | "
            f"{row.individual_truth_minus_model_trace:+.5f} ± "
            f"{row.individual_truth_minus_model_trace_noise_sem:.5f} | {status} |"
        )
    lines.extend([
        "",
        "Group means below are descriptive means of the deliberately selected "
        "fixed scenes, not population estimates for the full tail.",
        "",
        "| group | scenes | mean add trace | scene SEM | equivalent | + / - | "
        "target precision |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for name in ("all", "compact", "noncompact"):
        group = payload["groups"][name]
        stat = group["additivity_gap_trace"]
        lines.append(
            f"| {name} | {group['n_scenes']} | "
            f"{stat['mean_of_scene_means']:+.6f} | "
            f"{stat['scene_sem_descriptive']:.6f} | "
            f"{group['n_equivalent_within_sesoi']} | "
            f"{group['n_significantly_positive']} / "
            f"{group['n_significantly_negative']} | "
            f"{group['n_target_precision_met']} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--noiseless-results", required=True)
    parser.add_argument("--nreal", type=int, default=400)
    parser.add_argument("--g", type=float, default=0.02)
    parser.add_argument("--stamp", type=int, default=48)
    parser.add_argument("--pixel-rms", type=float, default=0.312)
    parser.add_argument("--sesoi", type=float, default=0.02)
    parser.add_argument("--target-ci-halfwidth", type=float, default=0.01)
    parser.add_argument("--min-success-fraction", type=float, default=0.8)
    parser.add_argument("--output-scenes", required=True)
    parser.add_argument("--output-pairs", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args()
    for path in (
        args.output_scenes, args.output_pairs, args.output_json, args.output_md,
    ):
        if os.path.exists(path):
            raise FileExistsError(f"refusing existing output {path}")

    manifest = pd.read_feather(args.manifest).sort_values("scene_id")
    noiseless = pd.read_feather(args.noiseless_results)
    noiseless = noiseless[[
        *KEY, "additivity_gap_trace", "coherent_truth_minus_model_trace",
        "individual_truth_minus_model_trace",
    ]].rename(columns={
        "additivity_gap_trace": "noiseless_additivity_gap_trace",
        "coherent_truth_minus_model_trace": (
            "noiseless_coherent_truth_minus_model_trace"
        ),
        "individual_truth_minus_model_trace": (
            "noiseless_individual_truth_minus_model_trace"
        ),
    })
    manifest = manifest.merge(noiseless, on=KEY, how="left", validate="one_to_one")
    if manifest.noiseless_additivity_gap_trace.isna().any():
        raise RuntimeError("selected scene lacks a successful noiseless reference")

    input_dir = Path(args.input_dir)
    scene_rows = []
    all_pair_draws = []
    audits = []
    replay_max = 0.0
    for row in manifest.itertuples(index=False):
        scene_id = int(row.scene_id)
        draw_path = input_dir / f"draws_scene{scene_id:02d}.feather"
        pair_path = input_dir / f"pairs_scene{scene_id:02d}.feather"
        audit_path = input_dir / f"audit_scene{scene_id:02d}.json"
        for path in (draw_path, pair_path, audit_path):
            if not path.exists():
                raise FileNotFoundError(path)
        draws = pd.read_feather(draw_path)
        pairs = pd.read_feather(pair_path)
        with open(audit_path, encoding="utf-8") as handle:
            audit = json.load(handle)
        audits.append(audit)
        if len(draws) != args.nreal:
            raise RuntimeError(
                f"scene {scene_id} has {len(draws)} draws, expected {args.nreal}"
            )
        if int(audit["scene_id"]) != scene_id:
            raise RuntimeError("scene audit id mismatch")
        good = draws.loc[draws.success.astype(bool)]
        local_replay = (
            good.coherent_truth_minus_model_trace
            - good.additivity_gap_trace
            - good.individual_truth_minus_model_trace
        )
        replay_max = max(
            replay_max, float(np.max(np.abs(local_replay.to_numpy(float))))
        )
        summary, _ = scene_summary(
            draws, pd.Series(row._asdict()), args.sesoi,
            args.target_ci_halfwidth,
        )
        for column in (
            "noiseless_additivity_gap_trace",
            "noiseless_coherent_truth_minus_model_trace",
            "noiseless_individual_truth_minus_model_trace",
        ):
            summary[column] = float(getattr(row, column))
        if summary["success_fraction"] < args.min_success_fraction:
            raise RuntimeError(
                f"scene {scene_id} success {summary['success_fraction']:.2%} "
                f"below {args.min_success_fraction:.2%}"
            )
        scene_rows.append(summary)
        all_pair_draws.append(pairs)
    scenes = pd.DataFrame(scene_rows).sort_values("scene_id").reset_index(drop=True)
    pair_draws = pd.concat(all_pair_draws, ignore_index=True)
    pair_groups = ["scene_id", *KEY, "secondary_index"]
    pair_summary = pair_draws.groupby(pair_groups, sort=True).agg(
        distance=("distance", "first"),
        R_model_pair=("R_model_pair", "first"),
        n_draws=("R_trace_individual", "size"),
        R11_individual=("R11_individual", "mean"),
        R22_individual=("R22_individual", "mean"),
        R_trace_individual=("R_trace_individual", "mean"),
        R_trace_individual_noise_sd=("R_trace_individual", "std"),
    ).reset_index()
    pair_summary["R_trace_individual_noise_sem"] = (
        pair_summary.R_trace_individual_noise_sd
        / np.sqrt(pair_summary.n_draws)
    )
    if replay_max > 2.0e-12:
        raise RuntimeError(f"noise-repeat decomposition does not close: {replay_max}")

    masks = {
        "all": np.ones(len(scenes), dtype=bool),
        "compact": scenes.compact_dominant_secondary.to_numpy(bool),
        "noncompact": ~scenes.compact_dominant_secondary.to_numpy(bool),
    }
    total_requested = int(sum(item["n_requested"] for item in audits))
    total_success = int(sum(item["n_success"] for item in audits))
    payload = {
        "design": {
            "description": audits[0]["design"],
            "g": float(args.g), "stamp": int(args.stamp),
            "pixel_rms": float(args.pixel_rms), "nreal": int(args.nreal),
            "primary_endpoint": "coherent-minus-individual response trace",
            "sesoi": float(args.sesoi),
            "target_ci_halfwidth": float(args.target_ci_halfwidth),
            "uncertainty_structure": (
                "noise draws are technical replicates within scene; scene means "
                "are the units for descriptive between-scene summaries"
            ),
            "selection": (
                "outcome-blind typical medoids, balanced 10 compact/10 noncompact"
            ),
        },
        "coverage": {
            "n_scenes": int(len(scenes)),
            "n_requested": total_requested, "n_success": total_success,
            "n_failure": total_requested - total_success,
            "success_fraction": float(total_success / total_requested),
            "failure_types": {
                str(name): int(count) for name, count in pd.Series([
                    failure["error_type"] for audit in audits
                    for failure in audit["failures"]
                ]).value_counts().items()
            },
        },
        "decomposition_replay_max_abs": replay_max,
        "groups": {
            name: summarize_group(scenes.loc[mask])
            for name, mask in masks.items()
        },
        "precision": {
            "all_scenes_target_halfwidth_met": bool(
                scenes.target_ci_halfwidth_met.all()
            ),
            "max_estimated_draws_for_target_halfwidth": int(
                scenes.estimated_draws_for_target_halfwidth.max()
            ),
            "scenes_needing_more_draws": scenes.loc[
                ~scenes.target_ci_halfwidth_met,
                ["scene_id", "estimated_draws_for_target_halfwidth"],
            ].to_dict(orient="records"),
        },
        "artifacts": {
            "scene_summary_feather": os.path.abspath(args.output_scenes),
            "pair_summary_feather": os.path.abspath(args.output_pairs),
            "report_md": os.path.abspath(args.output_md),
        },
    }
    Path(args.output_scenes).parent.mkdir(parents=True, exist_ok=True)
    scenes.to_feather(args.output_scenes)
    pair_summary.to_feather(args.output_pairs)
    with open(args.output_json, "x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    Path(args.output_md).write_text(markdown(payload, scenes), encoding="utf-8")
    print(markdown(payload, scenes))
    print("LOCALIZED_ANCHOR_NOISE_REPEAT_ANALYSIS_DONE", flush=True)


if __name__ == "__main__":
    main()
