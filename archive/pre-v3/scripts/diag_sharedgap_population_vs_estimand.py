"""Is the shared-carrier rule's 93%-vs-31% contrast POPULATION or ESTIMAND?

`localize_v22_shared_gap.py` reports that the frozen rule
`top_fraction:>0.7 AND dominant_response:positive` carries 92.9% of the constgold
NEIGHBOUR-PROXY deficit but only 31.2% of the constgold TOTAL deficit, and
WORKLOG 2026-08-13i read that as a second, roughly uniform component living in the
total and invisible to the anchor instrument.

That reading is not yet safe, because the two rows are not the same galaxies:

* `constgold_proxy` is evaluated on the half-shear-MATCHED frame (366,559 rows);
* `constgold_total` is evaluated on the FULL frame (735,237 rows).

`constgold_shared_fraction` is 0.4988, so the total row silently includes the ~50%
of constgold objects that have no half-shear self-truth match. AGENTS.md warns
about exactly this: a restricted list changes which model is being handicapped,
and an unchanged global mean is not evidence of an unchanged population.

The matched frame already carries `deficit_total`, so the apples-to-apples number
needs no new data.  This script evaluates, with the SAME frozen rule:

  A. proxy  deficit on the MATCHED rows   -> must reproduce the published 92.9%
  B. total  deficit on the MATCHED rows   -> the apples-to-apples number
  C. total  deficit on the FULL rows      -> must reproduce the published 31.2%

Then B-vs-C isolates the population effect and A-vs-B isolates the estimand
effect.  Nothing is fitted, no rule is re-selected, no correction is applied and
constgold stays evaluation-only.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

SBSI_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SBSI_ROOT not in sys.path:
    sys.path.insert(0, SBSI_ROOT)

from scripts.localize_v22_shared_gap import (  # noqa: E402
    CaseAggregator,
    candidate_rules,
    evaluate_rule,
    load_constgold,
)


def strip(block: dict) -> dict:
    """Keep the fields that matter for the comparison."""
    return {
        "selected_fraction": block["selected_fraction"]["mean"],
        "selected_conditional_deficit": block["selected_conditional_deficit"]["mean"],
        "selected_conditional_sem": block["selected_conditional_deficit"]["case_sem"],
        "complement_conditional_deficit":
            block["complement_conditional_deficit"]["mean"],
        "complement_conditional_sem":
            block["complement_conditional_deficit"]["case_sem"],
        "global_deficit": block["global_deficit"]["mean"],
        "global_sem": block["global_deficit"]["case_sem"],
        "selected_contribution": block["selected_contribution"]["mean"],
        "carrier_share": block["carrier_share"],
        "n_rows": block["n_rows"],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--constgold-gap", required=True)
    ap.add_argument("--constgold-pairs", required=True)
    ap.add_argument("--half-selfresp", required=True)
    ap.add_argument("--case-min", type=int, default=40)
    ap.add_argument("--case-max", type=int, default=139)
    ap.add_argument("--development-max", type=int, default=89)
    ap.add_argument("--rule", nargs="+",
                    default=["top_fraction:>0.7", "dominant_response:positive"])
    ap.add_argument("--also-rule", nargs="+",
                    default=["ratio:>5", "dominant_response:positive"],
                    help="the pilot-frozen rule, checked the same way")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    if os.path.exists(args.output):
        raise SystemExit(f"REFUSING to overwrite {args.output}")

    conditions, _ = candidate_rules()
    full, matched = load_constgold(
        args.constgold_gap, args.constgold_pairs, args.half_selfresp,
        args.case_min, args.case_max,
    )
    # validation window only, matching the published report
    full_val = full.loc[full.case > args.development_max].copy()
    matched_val = matched.loc[matched.case > args.development_max].copy()
    print(f"validation rows: full={len(full_val):,} matched={len(matched_val):,} "
          f"shared_fraction={len(matched_val) / len(full_val):.4f}", flush=True)

    payload = {
        "design": (
            "same frozen rule evaluated three ways to separate the population "
            "effect from the estimand effect; validation window only"
        ),
        "case_window": [args.case_min, args.case_max],
        "development_max": args.development_max,
        "rows": {
            "full_validation": int(len(full_val)),
            "matched_validation": int(len(matched_val)),
            "shared_fraction": float(len(matched_val) / len(full_val)),
        },
        "rules": {},
    }

    for label, rule in (("chosen", tuple(args.rule)),
                        ("pilot_frozen", tuple(args.also_rule))):
        agg_proxy = CaseAggregator(matched_val, "deficit_proxy")
        agg_total_matched = CaseAggregator(matched_val, "deficit_total")
        agg_total_full = CaseAggregator(full_val, "deficit_total")
        a = evaluate_rule(rule, conditions, matched_val, agg_proxy)
        b = evaluate_rule(rule, conditions, matched_val, agg_total_matched)
        c = evaluate_rule(rule, conditions, full_val, agg_total_full)
        if a is None or b is None or c is None:
            raise RuntimeError(f"rule {rule} is empty or full in some block")
        block = {
            "rule": list(rule),
            "A_proxy_on_matched": strip(a),
            "B_total_on_matched": strip(b),
            "C_total_on_full": strip(c),
            "estimand_effect_A_minus_B_carrier_share":
                float(a["carrier_share"] - b["carrier_share"]),
            "population_effect_B_minus_C_carrier_share":
                float(b["carrier_share"] - c["carrier_share"]),
        }
        payload["rules"][label] = block
        print(f"\n=== {label}: {' AND '.join(rule)} ===")
        print(f"{'':26s} {'frac':>7s} {'selected':>10s} {'complement':>11s} "
              f"{'global':>9s} {'share':>8s} {'n_rows':>9s}")
        for name, blk in (("A proxy  on MATCHED", a), ("B total  on MATCHED", b),
                          ("C total  on FULL   ", c)):
            s = strip(blk)
            print(f"{name:26s} {s['selected_fraction']:7.4f} "
                  f"{s['selected_conditional_deficit']:+10.5f} "
                  f"{s['complement_conditional_deficit']:+11.5f} "
                  f"{s['global_deficit']:+9.5f} {s['carrier_share']:8.1%} "
                  f"{s['n_rows']:9,d}")
        print(f"  estimand effect (A-B) = "
              f"{block['estimand_effect_A_minus_B_carrier_share']:+.1%} share")
        print(f"  population effect (B-C) = "
              f"{block['population_effect_B_minus_C_carrier_share']:+.1%} share")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=1, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print("\nWROTE", args.output)
    print("SHAREDGAP_POPULATION_VS_ESTIMAND_DONE", flush=True)


if __name__ == "__main__":
    main()
