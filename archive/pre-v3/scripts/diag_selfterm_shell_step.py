"""Does the flow's self-response error STEP at its own crowding-shell radii?

WORKLOG 2026-08-13l found that `self_term = S_h - R_flow^h` has a large,
sign-flipping, exactly-replicating profile in `dominant_distance`, with two of
the three largest reversals straddling 3.0" and 7.0" -- the edges of the flow's
near (0-3") and far (3-7") crowding shells.  A neighbour crossing such an edge
changes the flow's INPUT discontinuously while the true response varies smoothly
across it, so a genuine aperture artefact should show up as a STEP pinned to the
edge.

Two-window testing would be cheating: the same profile already reverses between
4.5" and 5.3", where no shell edge exists, so "there is a jump near 3"" is not
evidence by itself.  This script therefore fine-bins the WHOLE range at a uniform
width and treats every interior edge identically, so the candidate edges compete
against a real null distribution of jumps.

Predeclared reading, fixed before the numbers were seen:

  * the shell hypothesis PASSES only if the jumps at exactly 3.00" and 7.00" rank
    among the largest |t| of all interior edges;
  * if edges unrelated to any shell boundary produce jumps of the same size, the
    hypothesis FAILS and the profile is telling us about a physical scale (or
    about the pair-selection geometry) rather than about the flow's aperture.

The jump statistic is PAIRED BY CASE -- for each case, (mean above edge) minus
(mean below edge), then averaged over cases -- so case-level common modes cancel
and the SEM is the spread of that per-case difference.  Case is the uncertainty
unit throughout.

Two distances are tested.  `dominant_distance` is the distance of the
largest-|R_pair| neighbour, which is what the carrier rule keys on;
`closest_pair_distance` is the nearest neighbour, whose shell membership is what
actually moves flux between the flow's near and far sums.  The mechanism predicts
a step in BOTH, and most sharply in the second.

Uses no constgold `R_sim` and no `R_blend`: the pair table supplies geometry and
the split only, never a measured quantity.  Nothing is fitted, no correction is
applied.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import pyarrow.feather as pf

SBSI_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SBSI_ROOT not in sys.path:
    sys.path.insert(0, SBSI_ROOT)

from scripts.localize_v22_shared_gap import candidate_rules  # noqa: E402

KEY = ["case", "input_index"]
SEEDS = [501, 502, 503, *range(505, 518)]
RULE_COLUMNS = ["top_abs_fraction", "dominant_response",
                "dominant_to_runner_up_abs_response"]
SHELL_EDGES = [3.0, 7.0]


def case_profile(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    """Per (case, bin) mean of `column`; index (case, bin)."""
    return frame.groupby(["case", "_bin"], sort=True)[column].mean()


def jump_table(frame: pd.DataFrame, edges: np.ndarray, column: str,
               min_cases: int) -> list[dict]:
    """Paired-by-case jump across every interior edge."""
    profile = case_profile(frame, column).unstack("_bin")
    rows = []
    for i in range(len(edges) - 2):
        lo, hi = i, i + 1
        if lo not in profile.columns or hi not in profile.columns:
            continue
        paired = profile[[lo, hi]].dropna()
        if len(paired) < min_cases:
            continue
        diff = (paired[hi] - paired[lo]).to_numpy(float)
        sem = float(diff.std(ddof=1) / np.sqrt(diff.size))
        rows.append({
            "edge": float(edges[i + 1]),
            "below_center": float(0.5 * (edges[i] + edges[i + 1])),
            "above_center": float(0.5 * (edges[i + 1] + edges[i + 2])),
            "jump": float(diff.mean()),
            "jump_sem": sem,
            "t": float(diff.mean() / sem) if sem > 0 else float("nan"),
            "n_cases": int(diff.size),
            "n_rows_below": int((frame._bin == lo).sum()),
            "n_rows_above": int((frame._bin == hi).sum()),
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--half-selfresp", required=True)
    ap.add_argument("--constgold-pairs", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--case-min", type=int, default=40)
    ap.add_argument("--case-max", type=int, default=139)
    ap.add_argument("--development-max", type=int, default=89,
                    help="cases <= this are the replication half; the profile "
                         "is reported on BOTH halves independently")
    ap.add_argument("--distances", nargs="+",
                    default=["closest_pair_distance", "dominant_distance"])
    ap.add_argument("--bin-width", type=float, default=0.25)
    ap.add_argument("--range", type=float, nargs=2, default=[0.5, 10.0])
    ap.add_argument("--min-cases", type=int, default=10)
    ap.add_argument("--columns", nargs="+", default=["self_term"],
                    help="quantities to step-test; pass 'S_h R_flow_h' as the "
                         "decisive control -- an aperture artefact must step in "
                         "R_flow_h while the TRUTH S_h crosses smoothly")
    ap.add_argument("--rule",
                    default="top_fraction:>0.7 AND dominant_response:positive")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    if os.path.exists(args.output):
        raise SystemExit(f"REFUSING to overwrite {args.output}")

    flow_cols = [f"R_flow_s{seed}" for seed in args.seeds]
    half = pf.read_table(args.half_selfresp,
                         columns=KEY + ["r_sim_self"] + flow_cols,
                         memory_map=True).to_pandas()
    half = half[half.case.between(args.case_min, args.case_max)]

    pairs = pf.read_table(args.constgold_pairs,
                          columns=KEY + RULE_COLUMNS + list(args.distances),
                          memory_map=True).to_pandas()
    pairs = pairs[pairs.case.between(args.case_min, args.case_max)]

    frame = half.merge(pairs, on=KEY, how="inner", validate="one_to_one")
    finite = np.isfinite(
        frame[["r_sim_self"] + flow_cols + RULE_COLUMNS].to_numpy(float)
    ).all(axis=1)
    frame = frame.loc[finite].copy()
    frame["S_h"] = frame.r_sim_self.astype(np.float64)
    frame["R_flow_h"] = frame[flow_cols].to_numpy(np.float64).mean(axis=1)
    frame["self_term"] = frame.S_h - frame.R_flow_h

    conditions, _ = candidate_rules()
    selected = np.ones(len(frame), dtype=bool)
    for name in (p.strip() for p in args.rule.split("AND")):
        if name not in conditions:
            raise KeyError(f"unknown condition {name!r}")
        selected &= conditions[name].apply(frame)
    frame["selected"] = selected

    edges = np.arange(args.range[0],
                      args.range[1] + 0.5 * args.bin_width, args.bin_width)
    payload = {
        "design": (
            "uniform fine bins over the whole range so every interior edge is "
            "treated identically; jump is paired by case; predeclared reading is "
            "that the shell hypothesis passes only if 3.00\" and 7.00\" rank "
            "among the largest |t| of all edges"
        ),
        "rule": args.rule, "seeds": list(args.seeds),
        "bin_width": args.bin_width, "range": list(args.range),
        "shell_edges": SHELL_EDGES,
        "n_rows": int(len(frame)),
        "distances": {},
    }

    halves = {
        "first_half": frame[frame.case <= args.development_max],
        "second_half": frame[frame.case > args.development_max],
        "all_cases": frame,
    }

    for distance in args.distances:
        block = {}
        for label, sub in halves.items():
            sub = sub.copy()
            values = sub[distance].to_numpy(float)
            code = np.full(len(sub), -1, dtype=int)
            inside = np.isfinite(values) & (values >= edges[0]) & (values < edges[-1])
            code[inside] = np.digitize(values[inside], edges) - 1
            sub["_bin"] = code
            sub = sub[sub._bin >= 0]
            per_column = {}
            for column in args.columns:
                rows = jump_table(sub, edges, column, args.min_cases)
                if not rows:
                    raise RuntimeError(
                        f"no usable edges for {distance}/{label}/{column}")
                t = np.array([abs(r["t"]) for r in rows])
                order = np.argsort(-t)
                for rank, idx in enumerate(order, start=1):
                    rows[idx]["abs_t_rank"] = int(rank)
                per_column[column] = {
                    "n_edges": len(rows),
                    "edges": rows,
                    "shell_ranks": {
                        f"{e:.2f}": next((r["abs_t_rank"] for r in rows
                                          if abs(r["edge"] - e) < 1e-9), None)
                        for e in SHELL_EDGES
                    },
                    "max_abs_t": float(t.max()),
                    "median_abs_t": float(np.median(t)),
                }
            block[label] = per_column
        values = frame[distance].to_numpy(float)
        inside = np.isfinite(values) & (values >= edges[0]) & (values < edges[-1])
        code = np.full(len(frame), -1, dtype=int)
        code[inside] = np.digitize(values[inside], edges) - 1
        binned = frame.assign(_bin=code).loc[code >= 0]
        grouped = binned.groupby("_bin")
        counts = grouped.size().reindex(range(len(edges) - 1))
        profiles = {c: grouped[c].mean().reindex(range(len(edges) - 1))
                    for c in args.columns}
        block["profile_all_cases"] = [
            {"bin_center": float(0.5 * (edges[i] + edges[i + 1])),
             "n_rows": int(counts.iloc[i]) if np.isfinite(counts.iloc[i]) else 0,
             **{c: float(profiles[c].iloc[i]) for c in args.columns}}
            for i in range(len(edges) - 1)
            if np.isfinite(counts.iloc[i]) and counts.iloc[i] > 0
        ]
        print(f"\n  --- {distance}: bin means around the shell edges ---")
        print(f"    {'center':>7s} {'n_rows':>8s} "
              + " ".join(f"{c:>10s}" for c in args.columns))
        for row in block["profile_all_cases"]:
            near = any(abs(row["bin_center"] - e) < 0.6 for e in SHELL_EDGES)
            if near:
                print(f"    {row['bin_center']:7.3f} {row['n_rows']:8,d} "
                      + " ".join(f"{row[c]:+10.5f}" for c in args.columns))
        payload["distances"][distance] = block

        print(f"\n########## {distance} ##########")
        for label in ("first_half", "second_half"):
            for column in args.columns:
                b = block[label][column]
                print(f"\n--- {label} / {column}: {b['n_edges']} interior "
                      f"edges, median |t| = {b['median_abs_t']:.2f}, "
                      f"max |t| = {b['max_abs_t']:.2f}")
                print("  shell-edge ranks: " + ", ".join(
                    f"{k}\" -> rank {v}" for k, v in b["shell_ranks"].items()))
                top = sorted(b["edges"], key=lambda r: -abs(r["t"]))[:6]
                shown = {r["edge"] for r in top}
                extra = [x for x in b["edges"]
                         if any(abs(x["edge"] - e) < 1e-9 for e in SHELL_EDGES)
                         and x["edge"] not in shown]
                print(f"    {'edge':>7s} {'jump':>9s} {'sem':>8s} {'t':>7s} "
                      f"{'rank':>5s}")
                for r in top + extra:
                    mark = "  <== SHELL" if any(
                        abs(r["edge"] - e) < 1e-9 for e in SHELL_EDGES) else ""
                    print(f"    {r['edge']:7.2f} {r['jump']:+9.5f} "
                          f"{r['jump_sem']:8.5f} {r['t']:+7.2f} "
                          f"{r['abs_t_rank']:5d}{mark}")

    def clean(obj):
        if isinstance(obj, dict):
            return {k: clean(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [clean(v) for v in obj]
        if isinstance(obj, float) and not np.isfinite(obj):
            return None
        return obj

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "x", encoding="utf-8") as handle:
        json.dump(clean(payload), handle, indent=1, sort_keys=True,
                  allow_nan=False)
        handle.write("\n")
    print("\nWROTE", args.output)
    print("SELFTERM_SHELL_STEP_DONE", flush=True)


if __name__ == "__main__":
    main()
