"""Analyze a constgold q3 self/deployed/other/additivity experiment."""
from __future__ import annotations

import argparse
import json
import math
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.analyze_v22_matched_decomposition import (  # noqa: E402
    case_mean_sem, flow_mean_sem, total_mean_sem,
)


COMPONENTS = [
    ("Delta_flow", "Flow - self truth", "#0072B2"),
    ("Delta_deployed", "Emulator - deployed truth", "#D55E00"),
    ("Contribution_other", "Omitted local sources", "#CC79A7"),
    ("Contribution_additivity", "Non-additivity", "#009E73"),
    ("Model_minus_truth", "Total model - truth", "#000000"),
]
TARGETS = (0.0067, 0.015, 0.030)
Z_ALPHA_PLUS_POWER = 1.96 + 0.84  # two-sided alpha=.05, 80% power


def power_summary(frame: pd.DataFrame, column: str) -> dict:
    case_values = frame.groupby("case", sort=True)[column].mean().to_numpy(float)
    sd = float(case_values.std(ddof=1))
    n = len(case_values)
    return {
        "between_case_sd": sd,
        "current_n_cases": n,
        "current_sem": float(sd / np.sqrt(n)),
        "current_80pct_mde": float(Z_ALPHA_PLUS_POWER * sd / np.sqrt(n)),
        "targets": {
            f"{target:.4f}": {
                "cases_for_1sigma_sem": int(math.ceil((sd / target) ** 2)),
                "cases_for_80pct_power": int(math.ceil(
                    (Z_ALPHA_PLUS_POWER * sd / target) ** 2
                )),
            }
            for target in TARGETS
        },
    }


def multiplicative_bias_summary(frame: pd.DataFrame, flow_cols: list[str]) -> dict:
    truth_mean = float(frame["R_total_truth"].mean())
    blend_mean = float(frame["R_blend_model"].mean())
    seed_values = np.asarray([
        truth_mean / (float(frame[column].mean()) + blend_mean) - 1.0
        for column in flow_cols
    ])
    ensemble = frame[flow_cols].mean(axis=1)
    per_case = pd.DataFrame({
        "case": frame["case"], "truth": frame["R_total_truth"],
        "model": ensemble + frame["R_blend_model"],
    }).groupby("case", sort=True).mean()
    case_values = (per_case["truth"] / per_case["model"] - 1.0).to_numpy(float)
    seed_sem = float(seed_values.std(ddof=1) / np.sqrt(len(seed_values)))
    case_sem = float(case_values.std(ddof=1) / np.sqrt(len(case_values)))
    return {
        "mean": float(seed_values.mean()), "seed_sem": seed_sem,
        "case_sem": case_sem, "quadrature_sem": float(np.hypot(seed_sem, case_sem)),
        "seed_values": seed_values.tolist(), "case_values": case_values.tolist(),
        "definition": "mean(R_total_truth) / mean(R_flow + R_blend_model) - 1",
    }


def make_plot(result: dict, prefix: str) -> None:
    plt.rcParams.update({
        "font.family": "sans-serif", "font.size": 8.5,
        "axes.labelsize": 9, "axes.titlesize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    fig, ax = plt.subplots(figsize=(6.3, 3.6))
    x = np.arange(len(COMPONENTS))
    for i, (name, _, color) in enumerate(COMPONENTS):
        stats = result["components"][name]
        error = stats.get("quadrature_sem", stats["case_sem"])
        ax.errorbar(i, 100 * stats["mean"], yerr=100 * error, fmt="o",
                    color=color, capsize=3, markersize=5)
    ax.axhline(0, color="0.35", linestyle="--", linewidth=0.8)
    ax.set_xticks(x, [label for _, label, _ in COMPONENTS], rotation=24, ha="right")
    ax.set_ylabel("Model - truth contribution\n(response points)")
    ax.set_title(
        f"Constgold V2.2 q3 local-scene decomposition: "
        f"{result['n_rows']:,} anchors, {result['n_cases']} cases"
    )
    ax.text(
        0.01, 0.98,
        "Error bars: +/-1 case SEM; flow and total also include flow-seed SEM",
        transform=ax.transAxes, ha="left", va="top", fontsize=7, color="0.35",
    )
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    Path(prefix).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(prefix + ".pdf", bbox_inches="tight")
    fig.savefig(prefix + ".png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--truth", required=True)
    ap.add_argument("--flow", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--plot-prefix")
    args = ap.parse_args()
    if os.path.exists(args.output):
        raise FileExistsError(f"refusing to overwrite {args.output}")

    truth = pd.read_feather(args.truth)
    flow = pd.read_feather(args.flow)
    keys = ["case", "input_index"]
    if truth.duplicated(keys).any() or flow.duplicated(keys).any():
        raise RuntimeError("duplicate decomposition key")
    frame = truth.merge(flow, on=keys, how="inner", validate="one_to_one")
    if len(frame) != len(truth) or len(frame) != len(flow):
        raise RuntimeError(
            f"truth/flow mismatch: truth={len(truth)} flow={len(flow)} both={len(frame)}"
        )
    flow_cols = sorted(
        (c for c in frame if c.startswith("R_flow_s")),
        key=lambda c: int(re.search(r"(\d+)$", c).group(1)),
    )
    if len(flow_cols) != 16:
        raise RuntimeError(f"expected 16 V2.2 flow seeds, got {flow_cols}")
    need = [
        "R_total_truth", "R_self_truth", "R_deployed_truth", "R_other_truth",
        "R_blend_model", "Delta_deployed", "Contribution_other",
        "Contribution_additivity", "Delta_add_raw",
        *[f"R_{mode}_null" for mode in ("total", "self", "deployed", "other")],
        *flow_cols,
    ]
    finite = np.isfinite(frame[need].to_numpy(float)).all(axis=1)
    excluded = int((~finite).sum())
    frame = frame.loc[finite].copy()
    if frame.empty or frame["case"].nunique() < 2:
        raise RuntimeError("need finite anchors in at least two cases")

    frame["Delta_flow"] = frame[flow_cols].mean(axis=1) - frame["R_self_truth"]
    frame["Model_minus_truth"] = (
        frame[flow_cols].mean(axis=1) + frame["R_blend_model"]
        - frame["R_total_truth"]
    )
    frame["Component_sum"] = (
        frame["Delta_flow"] + frame["Delta_deployed"]
        + frame["Contribution_other"] + frame["Contribution_additivity"]
    )
    closure = np.abs(frame["Component_sum"] - frame["Model_minus_truth"])
    if float(closure.max()) > 5e-7:
        raise RuntimeError(f"component closure failed: max={closure.max():.3e}")

    flow_delta = frame[flow_cols].subtract(frame["R_self_truth"].to_numpy(float), axis=0)
    flow_stats = flow_mean_sem(pd.concat([frame[["case"]], flow_delta], axis=1), flow_cols)
    total_stats = total_mean_sem(frame, flow_cols)
    m_stats = multiplicative_bias_summary(frame, flow_cols)
    components = {
        "Delta_flow": flow_stats,
        "Delta_deployed": case_mean_sem(frame, "Delta_deployed"),
        "Contribution_other": case_mean_sem(frame, "Contribution_other"),
        "Contribution_additivity": case_mean_sem(frame, "Contribution_additivity"),
        "Model_minus_truth": total_stats,
    }
    truth_terms = {
        column: case_mean_sem(frame, column)
        for column in [
            "R_total_truth", "R_self_truth", "R_deployed_truth", "R_other_truth",
            "R_blend_model", "Delta_add_raw", "R_total_null", "R_self_null",
            "R_deployed_null", "R_other_null",
        ]
    }
    power = {name: power_summary(frame, name) for name, _, _ in COMPONENTS}

    n_cases = int(frame["case"].nunique())
    is_pilot = n_cases <= 8
    result = {
        "pilot": is_pilot, "evaluation_only": True,
        "n_rows": len(frame), "n_excluded_nonfinite": excluded,
        "n_cases": n_cases,
        "components": components, "truth_terms": truth_terms,
        "multiplicative_bias": m_stats,
        "max_float_closure_error": float(closure.max()),
        "power": {
            "method": (
                "between-case SD; normal approximation; two-sided alpha=.05 and 80% power; "
                "flow-seed uncertainty reported separately and not folded into case-count projection"
            ),
            "z_alpha_plus_power": Z_ALPHA_PLUS_POWER,
            "terms": power,
        },
        "reference_constgold_q3_pair_weighted": {
            "note": (
                "original localization weighting differs from this unique-anchor "
                + ("sparse pilot" if is_pilot else "decomposition experiment")
            ),
            "R_sim": 0.7087, "R_flow": 0.4952, "R_blend": 0.1801,
            "model_minus_truth_approx": -0.0334,
            "m_percent_approx": 4.97,
        },
    }
    print(
        f"q3 finite anchors={len(frame):,}, cases={frame['case'].nunique()}, "
        f"flow seeds={len(flow_cols)}, closure={closure.max():.2e}"
    )
    for name, label, _ in COMPONENTS:
        stats = components[name]
        error = stats.get("quadrature_sem", stats["case_sem"])
        print(f"  {label:<28} {stats['mean']:+.6f} +/- {error:.6f}")
    print(
        f"  {'m = Rtruth/Rmodel - 1':<28} {100*m_stats['mean']:+.3f}% "
        f"+/- {100*m_stats['quadrature_sem']:.3f}%"
    )
    print("  cases needed for 80% power at response effect 0.0067:")
    for name, label, _ in COMPONENTS:
        n = power[name]["targets"]["0.0067"]["cases_for_80pct_power"]
        print(f"    {label:<28} {n}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    if args.plot_prefix:
        make_plot(result, args.plot_prefix)
        print(f"wrote {args.plot_prefix}.pdf and {args.plot_prefix}.png")
    print(f"wrote {args.output}")
    print("V22_CONSTGOLD_Q3_DECOMP_ANALYZE_DONE", flush=True)


if __name__ == "__main__":
    main()
