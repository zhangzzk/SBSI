"""Is the V2.2 blend gap an AMPLITUDE effect (large `R_blend`) or a SHAPE effect?

The carrier rule that localises the gap -- `top_fraction:>0.7` /
`ratio:>20`, both with `dominant_response:positive` -- is a CONCENTRATION
statistic: it says one pair supplies most of the predicted response.  It says
nothing about how LARGE that response is.  A lone faint neighbour at 8" is
"dominant"; a large total `R_blend` built from several comparable neighbours is
explicitly NOT selected.  So "the gap is carried by dominant neighbours" does not
by itself imply "the model fails at large `R_blend`", and the two readings imply
different fixes.

This separates them on the coherent-anchor data, where truth is a direct
measurement and no constgold quantity is involved at all:

    gap = V2.2 prediction - g=0.02 sim truth        (negative = model too low)

binned by the model's own predicted amplitude `|prediction|`, in deciles.  Within
each amplitude bin the dominance split is applied unchanged, so:

  * if the failure is AMPLITUDE, the gap grows with `|prediction|` and the
    dominance contrast collapses once amplitude is held fixed;
  * if the failure is SHAPE, the dominance contrast survives at fixed amplitude.

Both the absolute gap and the FRACTIONAL gap (gap / mean |truth| in the bin) are
reported, because "the model fails more at large `R_blend`" and "the numbers are
simply bigger there" are different claims and only the fractional column
distinguishes them.

Bin edges come from the earliest block (the replication window) and are applied
unchanged to the rest, so the binning cannot be tuned on the reported number.
Case is the uncertainty unit.  Nothing is fitted and no correction is applied.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
import pyarrow.feather as pf

KEY = ["case", "input_index"]
PRED = "R_blend_lsst_r_extnbr_v22"
DOM_COLUMNS = ["case", "anchor_index", "dominant_to_runner_up_abs_response",
               "dominant_abs_response", "R_abs_sum"]


def stat(values: np.ndarray) -> dict:
    values = np.asarray(values, float)
    n = values.size
    sd = float(values.std(ddof=1)) if n > 1 else float("nan")
    return {"mean": float(values.mean()),
            "case_sem": sd / np.sqrt(n) if n > 1 else float("nan"),
            "n_cases": int(n)}


def load_concat(paths, columns, what):
    frames = []
    for path in paths:
        if not os.path.exists(path):
            raise SystemExit(f"missing {what}: {path}")
        frames.append(pf.read_table(path, columns=columns,
                                    memory_map=True).to_pandas())
    return pd.concat(frames, ignore_index=True)


def group(frame: pd.DataFrame, mask: np.ndarray) -> dict:
    local = frame.loc[mask]
    if local.empty or local.case.nunique() < 2:
        return {"gap": {"mean": float("nan"), "case_sem": float("nan"),
                        "n_cases": int(local.case.nunique())},
                "n_rows": int(len(local)), "fraction": float(mask.mean())}
    by_case = local.groupby("case", sort=True)[["gap", "truth", "prediction"]].mean()
    out = {"gap": stat(by_case.gap.to_numpy(float)),
           "mean_truth": float(by_case.truth.mean()),
           "mean_prediction": float(by_case.prediction.mean()),
           "n_rows": int(len(local)), "fraction": float(mask.mean())}
    denom = abs(out["mean_truth"])
    out["fractional_gap"] = (out["gap"]["mean"] / denom
                             if denom > 1e-12 else float("nan"))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--g002", required=True, nargs="+")
    ap.add_argument("--dominance", required=True, nargs="+")
    ap.add_argument("--ratio-threshold", type=float, default=20.0)
    ap.add_argument("--case-min", type=int, default=400)
    ap.add_argument("--case-max", type=int, default=899)
    ap.add_argument("--edge-window", type=int, nargs=2, default=[400, 499],
                    help="cases whose |prediction| deciles define the bin edges")
    ap.add_argument("--n-bins", type=int, default=10)
    ap.add_argument("--min-tail-fraction", type=float, default=0.05,
                    help="bins whose dominance tail is rarer than this are "
                         "EXCLUDED from the within-amplitude aggregate: their "
                         "contrast is built from a handful of anchors and "
                         "otherwise swamps the weighted mean")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    if os.path.exists(args.output):
        raise SystemExit(f"REFUSING to overwrite {args.output}")

    response = load_concat(args.g002, KEY + ["R_blend_truth", PRED], "responses")
    response = response[response.case.between(args.case_min, args.case_max)]
    dom = load_concat(args.dominance, DOM_COLUMNS, "dominance tables").rename(
        columns={"anchor_index": "input_index"})
    dom = dom[dom.case.between(args.case_min, args.case_max)]

    before = len(response)
    frame = response.merge(dom, on=KEY, how="inner", validate="one_to_one")
    coverage = len(frame) / before
    if coverage < 0.90:
        raise RuntimeError(f"dominance join covers only {coverage:.3%}")

    frame["prediction"] = frame[PRED].astype(np.float64)
    frame["truth"] = frame.R_blend_truth.astype(np.float64)
    frame["gap"] = frame.prediction - frame.truth
    frame["abs_prediction"] = frame.prediction.abs()
    tail = (frame.dominant_to_runner_up_abs_response.to_numpy(float)
            > args.ratio_threshold)
    frame["_tail"] = tail

    lo, hi = args.edge_window
    edge_rows = frame[frame.case.between(lo, hi)]
    qs = np.linspace(0.0, 1.0, args.n_bins + 1)[1:-1]
    edges = np.unique(np.quantile(edge_rows.abs_prediction.to_numpy(float), qs))
    code = np.digitize(frame.abs_prediction.to_numpy(float), edges)
    frame["_bin"] = code

    overall = group(frame, np.ones(len(frame), bool))
    tail_all = group(frame, tail)
    bulk_all = group(frame, ~tail)
    payload = {
        "design": (
            "anchor gap = V2.2 prediction - g=0.02 sim truth, binned by the "
            "model's own |prediction|; bin edges from the edge window applied "
            "unchanged elsewhere; dominance split applied within each bin; case "
            "is the uncertainty unit; negative gap = model too low"
        ),
        "ratio_threshold": args.ratio_threshold,
        "edge_window": [lo, hi],
        "edges": [float(e) for e in edges],
        "join_coverage": float(coverage),
        "n_rows": int(len(frame)),
        "all": overall, "tail": tail_all, "bulk": bulk_all,
        "bins": [],
    }

    print(f"rows={len(frame):,}  coverage={coverage:.3%}")
    print(f"ALL   gap={overall['gap']['mean']:+.5f} "
          f"+- {overall['gap']['case_sem']:.5f}  "
          f"frac_gap={overall['fractional_gap']:+.3%}")
    print(f"tail  gap={tail_all['gap']['mean']:+.5f} "
          f"+- {tail_all['gap']['case_sem']:.5f}  "
          f"frac_gap={tail_all['fractional_gap']:+.3%}  "
          f"({tail_all['fraction']:.1%} of anchors)")
    print(f"bulk  gap={bulk_all['gap']['mean']:+.5f} "
          f"+- {bulk_all['gap']['case_sem']:.5f}  "
          f"frac_gap={bulk_all['fractional_gap']:+.3%}")

    print(f"\n{'bin':>5s} {'med|pred|':>10s} {'n_rows':>8s} {'truth':>9s} "
          f"{'gap':>9s} {'sem':>8s} {'frac_gap':>9s} {'tail%':>6s} "
          f"{'gap_tail':>9s} {'gap_bulk':>9s} {'contrast':>9s}")
    contrasts, weights = [], []
    for b in sorted(set(code.tolist())):
        sub = frame[frame._bin == b]
        allb = group(sub, np.ones(len(sub), bool))
        t = group(sub, sub._tail.to_numpy())
        u = group(sub, ~sub._tail.to_numpy())
        contrast = t["gap"]["mean"] - u["gap"]["mean"]
        entry = {"bin": int(b),
                 "median_abs_prediction": float(sub.abs_prediction.median()),
                 "all": allb, "tail": t, "bulk": u,
                 "tail_minus_bulk": float(contrast)}
        payload["bins"].append(entry)
        entry["counted_in_aggregate"] = bool(
            np.isfinite(contrast) and t["fraction"] >= args.min_tail_fraction)
        if entry["counted_in_aggregate"]:
            contrasts.append(contrast)
            weights.append(len(sub))
        print(f"{b:5d} {entry['median_abs_prediction']:10.4f} "
              f"{allb['n_rows']:8,d} {allb['mean_truth']:+9.4f} "
              f"{allb['gap']['mean']:+9.5f} {allb['gap']['case_sem']:8.5f} "
              f"{allb['fractional_gap']:+9.2%} {t['fraction']:6.1%} "
              f"{t['gap']['mean']:+9.5f} {u['gap']['mean']:+9.5f} "
              f"{contrast:+9.5f}")

    uncond = tail_all["gap"]["mean"] - bulk_all["gap"]["mean"]
    within = float(np.average(contrasts, weights=weights))
    payload["min_tail_fraction"] = args.min_tail_fraction
    payload["bins_in_aggregate"] = [b["bin"] for b in payload["bins"]
                                    if b["counted_in_aggregate"]]
    payload["unconditional_tail_minus_bulk"] = float(uncond)
    payload["within_amplitude_tail_minus_bulk"] = within
    payload["surviving_fraction"] = float(within / uncond) if uncond else None
    print(f"\naggregate over bins {payload['bins_in_aggregate']} "
          f"(tail occupancy >= {args.min_tail_fraction:.0%})")
    print(f"dominance contrast: unconditional = {uncond:+.5f}   "
          f"within-amplitude = {within:+.5f}   "
          f"surviving = {within / uncond:.1%}")

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
    print("ANCHOR_GAP_BY_AMPLITUDE_DONE", flush=True)


if __name__ == "__main__":
    main()
