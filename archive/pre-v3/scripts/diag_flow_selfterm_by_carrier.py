"""Is the flow's self-response error really structured by the shared carrier rule?

WORKLOG 2026-08-13j read the flow/self term off a DIFFERENCE of two constgold
quantities:

    deficit_total - deficit_proxy = S_h - R_flow

and found it strongly opposite-signed across the frozen carrier rule
(-0.0376 selected / +0.0091 complement) while nearly cancelling globally
(-0.0030).  That reconciles 08-13c's "flow term is globally null" with a large
local structure -- but it inherits the shared-gap report's own caveat, that the
neighbour proxy "also contains selection-estimand differences and possible
non-additivity".  A difference of two constgold estimands is exactly where such an
artefact would hide.

This diagnostic removes that risk by measuring the flow term DIRECTLY, the way
08-13c defined it:

    self_term = S_h - R_flow^h

where `S_h` is the half-shear self truth and `R_flow^h` is the 16-seed ensemble
mean flow self response, both on the same `(case, input_index)` rows.  **It uses
no constgold `R_sim` and no `R_blend` at all**, so neither the blend estimand nor
any additivity assumption can enter.  The constgold pair-feature table is opened
only to evaluate the frozen rule's three input columns -- it contributes the SPLIT,
never the measured quantity.

Sign convention follows 08-13c: positive `self_term` means the flow UNDER-predicts
the self response.

Seeds: `self_term` is an e-response quantity, so the full 16-seed set is used
(501, 502, 503, 505--517; 504 does not exist), per AGENTS.md.  Per-seed values are
reported so the split can be checked for seed robustness rather than resting on
the ensemble mean alone.

Nothing is fitted, no rule is re-selected and no correction is applied.
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


def stat(values: np.ndarray) -> dict:
    values = np.asarray(values, float)
    n = values.size
    sd = float(values.std(ddof=1)) if n > 1 else float("nan")
    return {"mean": float(values.mean()), "case_sd": sd,
            "case_sem": sd / np.sqrt(n) if n > 1 else float("nan"),
            "n_cases": int(n)}


def summarize(frame: pd.DataFrame, mask: np.ndarray, column: str) -> dict:
    local = frame.loc[mask]
    by_case = local.groupby("case", sort=True)[column].mean()
    out = stat(by_case.to_numpy(float))
    out["fraction"] = float(mask.mean())
    out["n_rows"] = int(len(local))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--half-selfresp", required=True)
    ap.add_argument("--constgold-pairs", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--case-min", type=int, default=40)
    ap.add_argument("--case-max", type=int, default=139)
    ap.add_argument("--development-max", type=int, default=89)
    ap.add_argument("--rules", nargs="+",
                    default=["top_fraction:>0.7 AND dominant_response:positive",
                             "ratio:>5 AND dominant_response:positive"])
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    if os.path.exists(args.output):
        raise SystemExit(f"REFUSING to overwrite {args.output}")

    flow_cols = [f"R_flow_s{seed}" for seed in args.seeds]
    half = pf.read_table(args.half_selfresp,
                         columns=KEY + ["r_sim_self"] + flow_cols,
                         memory_map=True).to_pandas()
    half = half[half.case.between(args.case_min, args.case_max)]
    if half.duplicated(KEY).any():
        raise RuntimeError("duplicate half-shear keys")

    pairs = pf.read_table(args.constgold_pairs, columns=KEY + RULE_COLUMNS,
                          memory_map=True).to_pandas()
    pairs = pairs[pairs.case.between(args.case_min, args.case_max)]
    if pairs.duplicated(KEY).any():
        raise RuntimeError("duplicate constgold pair keys")

    n_half = len(half)
    frame = half.merge(pairs, on=KEY, how="inner", validate="one_to_one")
    finite = np.isfinite(
        frame[["r_sim_self"] + flow_cols + RULE_COLUMNS].to_numpy(float)
    ).all(axis=1)
    frame = frame.loc[finite].copy()
    coverage = len(frame) / n_half

    frame["S_h"] = frame.r_sim_self.astype(np.float64)
    frame["R_flow_h"] = frame[flow_cols].to_numpy(np.float64).mean(axis=1)
    frame["self_term"] = frame.S_h - frame.R_flow_h
    for seed in args.seeds:
        frame[f"self_term_s{seed}"] = frame.S_h - frame[f"R_flow_s{seed}"]

    conditions, _ = candidate_rules()
    dev = frame[frame.case <= args.development_max]
    val = frame[frame.case > args.development_max]

    payload = {
        "design": (
            "direct flow self-response error S_h - R_flow^h on half-shear truth, "
            "split by the frozen constgold carrier rule; uses no constgold R_sim "
            "and no R_blend, so it is immune to the neighbour-proxy's "
            "selection-estimand and non-additivity caveat; 16-seed ensemble mean; "
            "case is the uncertainty unit; positive = flow UNDER-predicts"
        ),
        "seeds": list(args.seeds),
        "counts": {
            "halfshear_rows": int(n_half),
            "matched_finite_rows": int(len(frame)),
            "coverage": float(coverage),
        },
        "windows": {
            "development": [args.case_min, args.development_max],
            "validation": [args.development_max + 1, args.case_max],
        },
        "rules": {},
    }

    for spec in args.rules:
        names = tuple(part.strip() for part in spec.split("AND"))
        for name in names:
            if name not in conditions:
                raise KeyError(f"unknown condition {name!r}")
        block = {}
        for window, sub in (("development", dev), ("validation", val),
                            ("all_cases", frame)):
            mask = np.ones(len(sub), dtype=bool)
            for name in names:
                mask &= conditions[name].apply(sub)
            selected = summarize(sub, mask, "self_term")
            complement = summarize(sub, ~mask, "self_term")
            glob = summarize(sub, np.ones(len(sub), bool), "self_term")
            entry = {"selected": selected, "complement": complement,
                     "global": glob,
                     "selected_minus_complement":
                         float(selected["mean"] - complement["mean"])}
            if window == "validation":
                per_seed = {}
                for seed in args.seeds:
                    s = summarize(sub, mask, f"self_term_s{seed}")["mean"]
                    c = summarize(sub, ~mask, f"self_term_s{seed}")["mean"]
                    per_seed[str(seed)] = {"selected": s, "complement": c,
                                           "difference": float(s - c)}
                diffs = np.array([v["difference"] for v in per_seed.values()])
                entry["per_seed"] = per_seed
                entry["per_seed_difference"] = stat(diffs)
                entry["seeds_agreeing_on_sign"] = int(
                    (np.sign(diffs) == np.sign(diffs.mean())).sum())
            block[window] = entry
        payload["rules"][spec] = block

        v = block["validation"]
        print(f"\n=== {spec} ===")
        print(f"  validation  frac={v['selected']['fraction']:.4f}")
        for nm in ("selected", "complement", "global"):
            b = v[nm]
            print(f"    {nm:11s} self_term={b['mean']:+.5f} +- {b['case_sem']:.5f}"
                  f"  ({b['n_rows']:,} rows)")
        print(f"    selected-complement = {v['selected_minus_complement']:+.5f}"
              f"   per-seed {v['per_seed_difference']['mean']:+.5f}"
              f" +- {v['per_seed_difference']['case_sem']:.5f}"
              f"   seeds agreeing on sign: "
              f"{v['seeds_agreeing_on_sign']}/{len(args.seeds)}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=1, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print("\nWROTE", args.output)
    print("FLOW_SELFTERM_BY_CARRIER_DONE", flush=True)


if __name__ == "__main__":
    main()
