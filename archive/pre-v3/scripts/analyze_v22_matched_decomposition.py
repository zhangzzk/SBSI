"""Summarize the matched V2.2 flow/emulator/additivity decomposition."""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TRUTH_TERMS = [
    "R_total_truth", "R_self_truth", "R_neighbour_coherent_truth",
    "R_pair_truth", "R_blend_model", "Delta_emu",
    "Delta_add_self_neighbour", "Delta_add_neighbours", "Delta_add",
    "R_total_null", "R_self_null", "R_neighbour_coherent_null", "R_pair_null",
]

FLUX_AXES = {
    "near": ("logflux_abs_near_0_1", 'Near: 0--1"'),
    "mid": ("logflux_abs_mid_1_3", 'Mid: 1--3"'),
    "far": ("logflux_abs_far_3_10", 'Far: 3--10"'),
}
CONTRIBUTIONS = [
    ("Delta_flow", "Flow", "#0072B2", "o"),
    ("Delta_emu", "Pair emulator", "#D55E00", "s"),
    ("Contribution_additivity", "Additivity", "#009E73", "^"),
]
PLOT_SERIES = [*CONTRIBUTIONS, ("Model_minus_truth", "Total", "#000000", "D")]


def case_mean_sem(frame: pd.DataFrame, column: str) -> dict:
    values = frame.groupby("case", sort=True)[column].mean().to_numpy(float)
    sem = values.std(ddof=1) / np.sqrt(len(values)) if len(values) > 1 else np.nan
    return {"mean": float(values.mean()), "case_sem": float(sem),
            "n_cases": len(values), "case_values": values.tolist()}


def flow_mean_sem(frame: pd.DataFrame, flow_cols: list[str]) -> dict:
    seed_values = np.asarray([float(frame[c].mean()) for c in flow_cols])
    ensemble = frame[flow_cols].mean(axis=1).to_numpy(float)
    case_values = pd.DataFrame({"case": frame["case"], "value": ensemble}).groupby(
        "case", sort=True,
    )["value"].mean().to_numpy(float)
    seed_sem = seed_values.std(ddof=1) / np.sqrt(len(seed_values))
    case_sem = (case_values.std(ddof=1) / np.sqrt(len(case_values))
                if len(case_values) > 1 else np.nan)
    return {
        "mean": float(seed_values.mean()), "seed_sem": float(seed_sem),
        "case_sem": float(case_sem), "quadrature_sem": float(np.hypot(seed_sem, case_sem)),
        "seed_values": seed_values.tolist(), "case_values": case_values.tolist(),
        "n_cases": len(case_values),
    }


def total_mean_sem(frame: pd.DataFrame, flow_cols: list[str]) -> dict:
    seed_values = np.asarray([
        float((frame[c] + frame["R_blend_model"] - frame["R_total_truth"]).mean())
        for c in flow_cols
    ])
    ensemble = (
        frame[flow_cols].mean(axis=1) + frame["R_blend_model"] - frame["R_total_truth"]
    ).to_numpy(float)
    case_values = pd.DataFrame({"case": frame["case"], "value": ensemble}).groupby(
        "case", sort=True,
    )["value"].mean().to_numpy(float)
    seed_sem = seed_values.std(ddof=1) / np.sqrt(len(seed_values))
    case_sem = (case_values.std(ddof=1) / np.sqrt(len(case_values))
                if len(case_values) > 1 else np.nan)
    return {
        "mean": float(seed_values.mean()), "seed_sem": float(seed_sem),
        "case_sem": float(case_sem), "quadrature_sem": float(np.hypot(seed_sem, case_sem)),
        "seed_values": seed_values.tolist(), "case_values": case_values.tolist(),
        "n_cases": len(case_values),
    }


def flux_edges(frame: pd.DataFrame, edges_from: str | None) -> tuple[
        dict[str, list[float]], dict[str, str], str]:
    if edges_from:
        with open(edges_from, encoding="utf-8") as handle:
            source = json.load(handle)
        edges = source["flux_bin_edges"]
        methods = source.get("flux_bin_methods", {key: "quartiles" for key in FLUX_AXES})
        return ({key: [float(v) for v in edges[key]] for key in FLUX_AXES},
                {key: str(methods[key]) for key in FLUX_AXES}, edges_from)
    edges = {}
    methods = {}
    for key, (column, _) in FLUX_AXES.items():
        values = frame[column].to_numpy(float)
        values = values[np.isfinite(values)]
        quartiles = np.quantile(values, [0.25, 0.5, 0.75])
        if len(np.unique(quartiles)) < 3 and np.any(values == 0.0):
            positive = values[values > 0.0]
            if len(positive) < 30:
                raise RuntimeError(f"too few positive {key} flux rows for zero+tertiles")
            edges[key] = [0.0, *np.quantile(positive, [1 / 3, 2 / 3]).tolist()]
            methods[key] = "zero_plus_positive_tertiles"
        else:
            edges[key] = quartiles.tolist()
            methods[key] = "quartiles_with_zero_folded_into_first_bin"
    return edges, methods, "derived from this pilot before final-case outcomes"


def add_flux_summary(result: dict, frame: pd.DataFrame, flow_cols: list[str],
                     edges_from: str | None) -> None:
    edges, methods, edge_source = flux_edges(frame, edges_from)
    result["flux_bin_edges"] = edges
    result["flux_bin_methods"] = methods
    result["flux_edge_source"] = edge_source
    result["flux_bins"] = {}
    for key, (column, label) in FLUX_AXES.items():
        cut = np.asarray(edges[key], float)
        index = np.digitize(frame[column].to_numpy(float), cut, right=True)
        labels = (["zero", "p1", "p2", "p3"]
                  if methods[key] == "zero_plus_positive_tertiles"
                  else ["q1", "q2", "q3", "q4"])
        axis = {"column": column, "label": label, "method": methods[key], "bins": []}
        for q in range(4):
            subset = frame.loc[index == q]
            if subset.empty:
                raise RuntimeError(f"empty {key} flux bin q{q + 1}")
            terms = {
                name: case_mean_sem(subset, name)
                for name in ("Delta_emu", "Contribution_additivity")
            }
            shifted_flow = subset[flow_cols].subtract(
                subset["R_self_truth"].to_numpy(float), axis=0,
            )
            flow_subset = pd.concat([subset[["case"]], shifted_flow], axis=1)
            terms["Delta_flow"] = flow_mean_sem(flow_subset, flow_cols)
            terms["Model_minus_truth"] = total_mean_sem(subset, flow_cols)
            axis["bins"].append({
                "bin": q + 1,
                "label": labels[q],
                "low": None if q == 0 else float(cut[q - 1]),
                "high": None if q == 3 else float(cut[q]),
                "n_rows": len(subset), "n_cases": int(subset["case"].nunique()),
                "terms": terms,
            })
        low = frame.loc[index == 0]
        high = frame.loc[index == 3]
        contrasts = {}
        for name in ("Delta_emu", "Contribution_additivity", "Model_minus_truth"):
            low_case = low.groupby("case", sort=True)[name].mean()
            high_case = high.groupby("case", sort=True)[name].mean()
            paired = high_case.to_frame("high").join(low_case.to_frame("low"), how="inner")
            values = (paired["high"] - paired["low"]).to_numpy(float)
            contrasts[name] = {
                "mean": float(values.mean()),
                "case_sem": float(values.std(ddof=1) / np.sqrt(len(values))),
                "case_values": values.tolist(),
            }
        low_flow = low[flow_cols].subtract(low["R_self_truth"].to_numpy(float), axis=0)
        high_flow = high[flow_cols].subtract(high["R_self_truth"].to_numpy(float), axis=0)
        seed_values = np.asarray([
            float(high_flow[c].mean() - low_flow[c].mean()) for c in flow_cols
        ])
        low_case = pd.DataFrame({
            "case": low["case"], "value": low_flow.mean(axis=1),
        }).groupby("case", sort=True)["value"].mean()
        high_case = pd.DataFrame({
            "case": high["case"], "value": high_flow.mean(axis=1),
        }).groupby("case", sort=True)["value"].mean()
        paired = high_case.to_frame("high").join(low_case.to_frame("low"), how="inner")
        case_values = (paired["high"] - paired["low"]).to_numpy(float)
        seed_sem = seed_values.std(ddof=1) / np.sqrt(len(seed_values))
        case_sem = case_values.std(ddof=1) / np.sqrt(len(case_values))
        contrasts["Delta_flow"] = {
            "mean": float(seed_values.mean()), "seed_sem": float(seed_sem),
            "case_sem": float(case_sem),
            "quadrature_sem": float(np.hypot(seed_sem, case_sem)),
            "seed_values": seed_values.tolist(), "case_values": case_values.tolist(),
        }
        axis["q4_minus_q1"] = contrasts
        result["flux_bins"][key] = axis


def make_plot(result: dict, prefix: str) -> None:
    plt.rcParams.update({
        "font.family": "sans-serif", "font.size": 8,
        "axes.labelsize": 9, "axes.titlesize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.fontsize": 8, "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    fig, axes = plt.subplots(1, 4, figsize=(10.2, 2.85),
                             gridspec_kw={"width_ratios": [0.9, 1, 1, 1]},
                             sharey=True)
    global_terms = {
        "Delta_flow": result["Delta_flow"],
        "Delta_emu": result["truth_terms"]["Delta_emu"],
        "Contribution_additivity": result["contribution_additivity"],
        "Model_minus_truth": result["model_minus_truth"],
    }
    for j, (name, label, color, marker) in enumerate(PLOT_SERIES):
        stats = global_terms[name]
        error = stats.get("quadrature_sem", stats["case_sem"])
        axes[0].errorbar(j, 100.0 * stats["mean"], yerr=100.0 * error,
                         fmt=marker, color=color, capsize=2.5, markersize=5)
    axes[0].set_xticks(range(4), [x[1] for x in PLOT_SERIES], rotation=35, ha="right")
    axes[0].set_title("Global")
    axes[0].set_ylabel(r"Contribution to model $-$ truth response (points)")
    for ax, key in zip(axes[1:], FLUX_AXES):
        bins = result["flux_bins"][key]["bins"]
        x = np.arange(1, 5)
        offsets = (-0.15, -0.05, 0.05, 0.15)
        for offset, (name, label, color, marker) in zip(offsets, PLOT_SERIES):
            stats = [item["terms"][name] for item in bins]
            y = 100.0 * np.asarray([s["mean"] for s in stats])
            err = 100.0 * np.asarray([
                s.get("quadrature_sem", s["case_sem"]) for s in stats
            ])
            ax.errorbar(x + offset, y, yerr=err, fmt=marker + "-", color=color,
                        linewidth=1.0, capsize=2, markersize=4, label=label)
        ax.set_xticks(x, [item["label"] for item in bins])
        ax.set_xlabel("Absolute neighbour flux bin")
        ax.set_title(result["flux_bins"][key]["label"])
    for ax in axes:
        ax.axhline(0.0, color="0.35", linewidth=0.8, linestyle="--", zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
    axes[-1].legend(frameon=False, loc="best")
    subtitle = (f"Matched fresh V2.2-domain scenes; n={result['n_rows']:,} anchors, "
                f"{result['n_cases']} independent cases; errors: case SEM"
                + (" + flow-seed SEM" if result["Delta_flow"]["seed_sem"] > 0 else ""))
    fig.suptitle(subtitle, fontsize=8.5, y=1.02)
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
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--edges-from",
                    help="pilot JSON supplying pre-registered flux quartile cut points")
    ap.add_argument("--plot-prefix")
    args = ap.parse_args()
    if os.path.exists(args.output):
        raise FileExistsError(f"refusing to overwrite {args.output}")

    truth = pd.read_feather(args.truth)
    flow = pd.read_feather(args.flow)
    if truth.duplicated(["case", "input_index"]).any() \
            or flow.duplicated(["case", "input_index"]).any():
        raise RuntimeError("duplicate decomposition key")
    frame = truth.merge(flow, on=["case", "input_index"], how="inner", validate="one_to_one")
    if len(frame) != len(truth) or len(frame) != len(flow):
        raise RuntimeError(f"truth/flow key mismatch: truth={len(truth)} flow={len(flow)} both={len(frame)}")
    flow_cols = sorted(
        (c for c in frame if c.startswith("R_flow_s")),
        key=lambda c: int(re.search(r"(\d+)$", c).group(1)),
    )
    if len(flow_cols) != 16:
        raise RuntimeError(f"expected 16 flow seeds, got {flow_cols}")
    finite_columns = list(dict.fromkeys([
        *TRUTH_TERMS, *flow_cols,
        "R_blend_model", "Delta_add",
        *(column for column, _ in FLUX_AXES.values()),
        *(c for c in frame if c.startswith("R_pair_truth_pair")
          or c.startswith("R_pair_truth_rank")
          or c.startswith("R_pair_null_rank")),
    ]))
    finite = np.isfinite(frame[finite_columns].to_numpy(float)).all(axis=1)
    n_input_rows = len(frame)
    n_excluded_nonfinite = int((~finite).sum())
    frame = frame.loc[finite].copy()
    if frame.empty:
        raise RuntimeError("no finite matched decomposition rows")
    result = {
        "pilot": bool(args.pilot), "n_rows": len(frame),
        "n_input_rows": n_input_rows,
        "n_excluded_nonfinite": n_excluded_nonfinite,
        "n_cases": int(frame["case"].nunique()), "truth_terms": {},
    }
    print(f"matched finite anchors={len(frame):,}/{n_input_rows:,} "
          f"(excluded {n_excluded_nonfinite}), cases={frame['case'].nunique()}, flow seeds=16")
    for column in TRUTH_TERMS:
        result["truth_terms"][column] = case_mean_sem(frame, column)
        r = result["truth_terms"][column]
        print(f"  {column:<30} {r['mean']:+.6f} +/- {r['case_sem']:.6f} caseSEM")

    flow_delta = frame[flow_cols].subtract(frame["R_self_truth"].to_numpy(float), axis=0)
    result["Delta_flow"] = flow_mean_sem(
        pd.concat([frame[["case"]], flow_delta], axis=1), flow_cols,
    )
    d = result["Delta_flow"]
    print(
        f"  {'Delta_flow':<30} {d['mean']:+.6f} +/- {d['case_sem']:.6f} case "
        f"+/- {d['seed_sem']:.6f} seed (quad {d['quadrature_sem']:.6f})"
    )

    probes = sorted(c for c in frame if c.startswith("R_pair_truth_pair"))
    result["pair_probes"] = {}
    for column in probes:
        result["pair_probes"][column] = case_mean_sem(frame, column)
    if len(probes) >= 2:
        difference = frame[probes[0]].to_numpy(float) - frame[probes[1]].to_numpy(float)
        temp = pd.DataFrame({"case": frame["case"], "difference": difference})
        per_case = temp.groupby("case", sort=True)["difference"].mean().to_numpy(float)
        result["pair_probe_difference"] = {
            "mean": float(difference.mean()),
            "case_sem": float(per_case.std(ddof=1) / np.sqrt(len(per_case))),
            "case_values": per_case.tolist(),
        }
        pdiff = result["pair_probe_difference"]
        print(f"  pair0-minus-pair1            {pdiff['mean']:+.6f} +/- "
              f"{pdiff['case_sem']:.6f} caseSEM")

    ranks = sorted(
        (c for c in frame if c.startswith("R_pair_truth_rank")),
        key=lambda c: int(c.rsplit("rank", 1)[1]),
    )
    result["pair_rank_components"] = {
        column: case_mean_sem(frame, column) for column in ranks
    }
    if ranks:
        print(f"  exact pair-rank components  {len(ranks)} (summed, not averaged)")

    frame["Delta_flow"] = frame[flow_cols].mean(axis=1) - frame["R_self_truth"]
    frame["Contribution_additivity"] = -frame["Delta_add"]
    frame["Model_minus_truth"] = (
        frame["Delta_flow"] + frame["Delta_emu"] + frame["Contribution_additivity"]
    )
    direct = frame[flow_cols].mean(axis=1) + frame["R_blend_model"] - frame["R_total_truth"]
    closure_error = np.abs(frame["Model_minus_truth"].to_numpy(float) - direct.to_numpy(float))
    max_closure_error = float(closure_error.max())
    if max_closure_error > 5.0e-7:
        raise RuntimeError(
            f"decomposition fails direct model-minus-truth closure: max={max_closure_error:.3e}"
        )
    result["max_float_closure_error"] = max_closure_error
    result["contribution_additivity"] = case_mean_sem(frame, "Contribution_additivity")
    result["model_minus_truth"] = total_mean_sem(frame, flow_cols)
    add_flux_summary(result, frame, flow_cols, args.edges_from)

    with open(args.output, "x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    if args.plot_prefix:
        make_plot(result, args.plot_prefix)
        print(f"wrote {args.plot_prefix}.pdf and {args.plot_prefix}.png")
    print(f"wrote {args.output}")
    print("V22_MATCHED_DECOMP_ANALYZE_DONE", flush=True)


if __name__ == "__main__":
    main()
