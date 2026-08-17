"""Is the flow's self-response error driven by neighbour flux?

WORKLOG 2026-08-13k established DIRECTLY on half-shear truth (no constgold
`R_sim`, no `R_blend`) that

    self_term = S_h - R_flow^h        (positive = flow UNDER-predicts)

is strongly structured by the frozen carrier rule: -0.0345 +- 0.0060 inside the
selected group, +0.0089 +- 0.0025 outside, contrast -0.0433 with 16/16 seeds
agreeing on sign.  Two candidate mechanisms were left open, and the first is
testable with data already on disk:

  (a) the flow conditions on MEASURED inputs (`SN`, `r_input_p`, `Re_input_p`),
      and a bright close neighbour contaminates them, so the flow is handed a
      galaxy that looks brighter/larger than it is and returns too large a self
      response;
  (b) something specific to BlendEMU's dominant-pair regime.

If (a) is the mechanism, `self_term` should track `nbr_flux_near` -- which is
already a column in the half-shear dump -- and the carrier rule's contrast
should LARGELY COLLAPSE once `nbr_flux_near` is held fixed, because the rule
would then just be a coarse proxy for "bright close neighbour".  If the contrast
SURVIVES at fixed neighbour flux, the carrier rule is carrying information that
neighbour brightness alone does not, and (a) is at best partial.

Design, fixed before looking at the validation numbers:

* Bin edges are the deciles of `nbr_flux_near` computed on the DEVELOPMENT
  window (cases <= 89) and then applied unchanged to the validation window, so
  the binning cannot be tuned on the number being reported.
* Rows with non-positive or non-finite `nbr_flux_near` (isolated anchors) form
  their own group rather than being forced into a quantile.
* The headline is the WITHIN-BIN selected-minus-complement contrast, averaged
  over bins with a common weight, next to the unconditional contrast.  A ratio
  near 0 means neighbour flux explains the split; near 1 means it does not.
* `SN` and `Re_input_p` are run through the identical machinery as CONTROLS, so
  a generic faint-end or small-size trend cannot be misread as a neighbour
  effect.

Case is the uncertainty unit throughout.  Seeds: 16 (e-response quantity, per
AGENTS.md).  Nothing is fitted, no rule is re-selected, no correction is applied,
and constgold is opened only to supply the SPLIT -- never a measured quantity.
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
SPLIT_VARIABLES = ["nbr_flux_near", "SN", "Re_input_p"]


def stat(values: np.ndarray) -> dict:
    values = np.asarray(values, float)
    n = values.size
    sd = float(values.std(ddof=1)) if n > 1 else float("nan")
    return {"mean": float(values.mean()), "case_sem": sd / np.sqrt(n) if n > 1
            else float("nan"), "n_cases": int(n)}


def case_stat(frame: pd.DataFrame, mask: np.ndarray, column: str) -> dict:
    local = frame.loc[mask]
    if local.empty or local.case.nunique() < 2:
        return {"mean": float("nan"), "case_sem": float("nan"),
                "n_cases": int(local.case.nunique()), "n_rows": int(len(local))}
    out = stat(local.groupby("case", sort=True)[column].mean().to_numpy(float))
    out["n_rows"] = int(len(local))
    return out


def decile_edges(values: np.ndarray, n_bins: int) -> np.ndarray:
    """Interior edges only; monotone-safe against ties."""
    qs = np.linspace(0.0, 1.0, n_bins + 1)[1:-1]
    edges = np.quantile(values, qs)
    return np.unique(edges)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--half-selfresp", required=True)
    ap.add_argument("--constgold-pairs", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--case-min", type=int, default=40)
    ap.add_argument("--case-max", type=int, default=139)
    ap.add_argument("--development-max", type=int, default=89)
    ap.add_argument("--n-bins", type=int, default=10)
    ap.add_argument("--rule",
                    default="top_fraction:>0.7 AND dominant_response:positive")
    ap.add_argument("--variables", nargs="+", default=SPLIT_VARIABLES,
                    help="split variables read from the half-shear dump")
    ap.add_argument("--pair-variables", nargs="*", default=[],
                    help="split variables read from the constgold pair table; "
                         "these describe the SPLIT geometry, never a measured "
                         "quantity")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    if os.path.exists(args.output):
        raise SystemExit(f"REFUSING to overwrite {args.output}")

    flow_cols = [f"R_flow_s{seed}" for seed in args.seeds]
    half = pf.read_table(
        args.half_selfresp,
        columns=KEY + ["r_sim_self"] + args.variables + flow_cols,
        memory_map=True).to_pandas()
    half = half[half.case.between(args.case_min, args.case_max)]
    if half.duplicated(KEY).any():
        raise RuntimeError("duplicate half-shear keys")

    pair_extra = [c for c in args.pair_variables if c not in RULE_COLUMNS]
    pairs = pf.read_table(args.constgold_pairs,
                          columns=KEY + RULE_COLUMNS + pair_extra,
                          memory_map=True).to_pandas()
    pairs = pairs[pairs.case.between(args.case_min, args.case_max)]
    if pairs.duplicated(KEY).any():
        raise RuntimeError("duplicate constgold pair keys")

    frame = half.merge(pairs, on=KEY, how="inner", validate="one_to_one")
    finite = np.isfinite(
        frame[["r_sim_self"] + flow_cols + RULE_COLUMNS].to_numpy(float)
    ).all(axis=1)
    frame = frame.loc[finite].copy()
    frame["self_term"] = (frame.r_sim_self.astype(np.float64)
                          - frame[flow_cols].to_numpy(np.float64).mean(axis=1))

    conditions, _ = candidate_rules()
    names = tuple(part.strip() for part in args.rule.split("AND"))
    for name in names:
        if name not in conditions:
            raise KeyError(f"unknown condition {name!r}")
    selected = np.ones(len(frame), dtype=bool)
    for name in names:
        selected &= conditions[name].apply(frame)
    frame["selected"] = selected

    dev = frame[frame.case <= args.development_max]
    val = frame[frame.case > args.development_max].copy()

    payload = {
        "design": (
            "self_term = S_h - R_flow^h binned against candidate drivers; bin "
            "edges are development-window quantiles applied unchanged to the "
            "validation window; headline is the within-bin selected-minus-"
            "complement contrast vs the unconditional one; case is the "
            "uncertainty unit; positive self_term = flow UNDER-predicts"
        ),
        "rule": args.rule,
        "seeds": list(args.seeds),
        "n_bins": args.n_bins,
        "counts": {"matched_finite_rows": int(len(frame)),
                   "development_rows": int(len(dev)),
                   "validation_rows": int(len(val))},
        "variables": {},
    }

    uncond_s = case_stat(val, val.selected.to_numpy(), "self_term")
    uncond_c = case_stat(val, ~val.selected.to_numpy(), "self_term")
    uncond = float(uncond_s["mean"] - uncond_c["mean"])
    payload["unconditional"] = {
        "selected": uncond_s, "complement": uncond_c,
        "selected_minus_complement": uncond,
        "selected_fraction": float(val.selected.mean()),
    }
    print(f"unconditional  selected={uncond_s['mean']:+.5f}"
          f" +- {uncond_s['case_sem']:.5f}   "
          f"complement={uncond_c['mean']:+.5f} +- {uncond_c['case_sem']:.5f}   "
          f"contrast={uncond:+.5f}", flush=True)

    for variable in list(args.variables) + list(args.pair_variables):
        dev_v = dev[variable].to_numpy(float)
        val_v = val[variable].to_numpy(float)
        dev_ok = np.isfinite(dev_v) & (dev_v > 0)
        val_ok = np.isfinite(val_v) & (val_v > 0)
        edges = decile_edges(dev_v[dev_ok], args.n_bins)
        code = np.full(len(val), -1, dtype=int)
        code[val_ok] = np.digitize(val_v[val_ok], edges)
        val["_bin"] = code

        rows, contrasts, weights = [], [], []
        for b in sorted(set(code.tolist())):
            sub = val[val._bin == b]
            m = sub.selected.to_numpy()
            s = case_stat(sub, m, "self_term")
            c = case_stat(sub, ~m, "self_term")
            entry = {
                "bin": int(b),
                "label": "isolated/non-positive" if b < 0 else f"q{b + 1}",
                "variable_median": (float(np.median(sub[variable]))
                                    if len(sub) else float("nan")),
                "n_rows": int(len(sub)),
                "selected_fraction": float(m.mean()) if len(sub) else float("nan"),
                "selected": s, "complement": c,
                "all": case_stat(sub, np.ones(len(sub), bool), "self_term"),
            }
            entry["selected_minus_complement"] = float(s["mean"] - c["mean"])
            rows.append(entry)
            if np.isfinite(entry["selected_minus_complement"]):
                contrasts.append(entry["selected_minus_complement"])
                weights.append(len(sub))

        contrasts = np.asarray(contrasts, float)
        weights = np.asarray(weights, float)
        within = float((contrasts * weights).sum() / weights.sum())
        block = {
            "development_edges": [float(e) for e in edges],
            "bins": rows,
            "within_bin_contrast": within,
            "unconditional_contrast": uncond,
            "surviving_fraction": float(within / uncond),
        }
        payload["variables"][variable] = block

        print(f"\n=== split by {variable} ===")
        print(f"{'bin':>22s} {'median':>10s} {'n_rows':>9s} {'sel.frac':>8s} "
              f"{'all':>9s} {'selected':>10s} {'complement':>11s} "
              f"{'contrast':>9s}")
        for e in rows:
            print(f"{e['label']:>22s} {e['variable_median']:10.4g} "
                  f"{e['n_rows']:9,d} {e['selected_fraction']:8.4f} "
                  f"{e['all']['mean']:+9.5f} {e['selected']['mean']:+10.5f} "
                  f"{e['complement']['mean']:+11.5f} "
                  f"{e['selected_minus_complement']:+9.5f}")
        print(f"  within-bin contrast = {within:+.5f}   "
              f"unconditional = {uncond:+.5f}   "
              f"surviving = {within / uncond:.1%}")

    def clean(obj):
        """JSON has no NaN; a bin with <2 cases legitimately has none."""
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
    print("FLOW_SELFTERM_VS_NBRFLUX_DONE", flush=True)


if __name__ == "__main__":
    main()
