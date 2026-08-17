"""Coherent-anchor V2.2 gap at |g|=0.02, split by single-pair response dominance.

The paired script ``analyze_anchor_amplitude_by_dominance.py`` compared the
already-rendered `g=0.02` cases 400--499 against their `g=0.05` partners.  Its
predeclared amplitude gate FAILED for the tail (the tail deficit did not shrink
at `g=0.02`) but the BULK moved from `+0.00024 +- 0.00230` at `g=0.05` to
`-0.00688 +- 0.00492` at `g=0.02` -- only 1.4 sigma, so the carrier split at
constgold's amplitude is unresolved.  Response noise scales like `1/g`, which is
why 100 cases at `g=0.02` carry roughly twice the SEM of 100 cases at `g=0.05`.

This script scores the EXTENDED `g=0.02` block on its own.  No `g=0.05` partner
is used or needed: the quantity is ``gap = V2.2 prediction - g=0.02 sim truth``,
and the prediction side is a deterministic function of the input field, so extra
unpaired `g=0.02` cases buy SEM directly.

Predeclared reading, fixed before the extension was rendered:

* if the NON-TAIL (bulk) gap at `g=0.02` is consistent with zero within two case
  SEM, the coherent-anchor instrument stays TAIL-dominated at constgold's
  amplitude, and it does not reproduce constgold's bulk-carried deficit (78% of
  the constgold gap sits outside the dominance tail).  The anchor line then
  measures a different effect and stops being a proxy for the constgold gap.
* if the bulk gap at `g=0.02` is negative at more than two case SEM and carries a
  majority of the total contribution, the anchor instrument DOES reproduce
  constgold's bulk-dominated carrier once run at matching amplitude, and the
  earlier anchor/constgold divergence was an artefact of measuring anchors at
  `g=0.05`.

Nothing is fitted, no correction is applied and constgold is not opened.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy import stats

KEY = ["case", "input_index"]
PRED = "R_blend_lsst_r_extnbr_v22"
DOM_COLUMNS = ["case", "anchor_index", "dominant_to_runner_up_abs_response",
               "dominant_abs_response", "R_abs_sum"]


def stat(values: np.ndarray) -> dict:
    values = np.asarray(values, float)
    sd = float(values.std(ddof=1)) if values.size > 1 else float("nan")
    return {
        "mean": float(values.mean()),
        "case_sd": sd,
        "case_sem": sd / np.sqrt(values.size) if values.size > 1 else float("nan"),
        "n_cases": int(values.size),
    }


def group_summary(frame: pd.DataFrame, mask: np.ndarray) -> dict:
    local = frame.loc[mask]
    if not len(local):
        raise RuntimeError("empty dominance group")
    case = local.groupby("case", sort=True)[["gap", "truth", "prediction"]].mean()
    gaps = case.gap.to_numpy(float)
    out = {
        "n_rows": int(len(local)),
        "fraction": float(mask.mean()),
        "gap": stat(gaps),
        "truth": stat(case.truth.to_numpy(float)),
        "prediction": stat(case.prediction.to_numpy(float)),
    }
    test = stats.ttest_1samp(gaps, 0.0)
    out["p_gap_zero"] = float(test.pvalue)
    out["t_gap_zero"] = float(test.statistic)
    # share of the global truth-minus-model shortfall carried by this group
    out["contribution"] = float(out["fraction"] * out["gap"]["mean"])
    return out


def load_concat(paths: list[str], columns: list[str], label: str) -> pd.DataFrame:
    parts = [pd.read_feather(path, columns=columns) for path in paths]
    frame = pd.concat(parts, ignore_index=True)
    if frame.duplicated(columns[:2]).any():
        raise RuntimeError(f"duplicate keys after concatenating {label}")
    return frame


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--g002", required=True, nargs="+",
                    help="one or more anchorblend g=0.02 response feathers")
    ap.add_argument("--dominance", required=True, nargs="+",
                    help="matching per-anchor dominance tables")
    ap.add_argument("--ratio-threshold", type=float, default=20.0)
    ap.add_argument("--scan", type=float, nargs="+",
                    default=[5.0, 10.0, 20.0, 40.0, 100.0])
    ap.add_argument("--case-min", type=int, default=400)
    ap.add_argument("--case-max", type=int, default=899)
    ap.add_argument("--replicate-window", type=int, nargs=2, default=[400, 499],
                    help="case window whose result must replicate the published "
                         "100-case number; reported separately as a control")
    ap.add_argument("--restrict-keys", default=None,
                    help="optional feather whose (case,input_index) keys the "
                         "anchor set is inner-joined down to.  Used ONLY by the "
                         "replication gate, to reproduce the published PAIRED "
                         "number: the published analysis inner-joins against the "
                         "g=0.05 partner and so silently drops anchors that the "
                         "g=0.02 render has and the g=0.05 render does not.  The "
                         "production run is deliberately unpaired and must not "
                         "set this.")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    if os.path.exists(args.output):
        raise SystemExit(f"REFUSING to overwrite {args.output}")

    response = load_concat(args.g002, KEY + ["R_blend_truth", PRED], "responses")
    response = response[response.case.between(args.case_min, args.case_max)]

    restricted_fraction = 1.0
    if args.restrict_keys:
        keys = pd.read_feather(args.restrict_keys, columns=KEY)
        keys = keys[keys.case.between(args.case_min, args.case_max)].drop_duplicates()
        before = len(response)
        response = response.merge(keys, on=KEY, how="inner", validate="one_to_one")
        restricted_fraction = len(response) / before
        print(f"restricted to {args.restrict_keys}: {before:,} -> {len(response):,} "
              f"anchors ({restricted_fraction:.4%})")

    dom = load_concat(args.dominance, DOM_COLUMNS, "dominance tables").rename(
        columns={"anchor_index": "input_index"})
    dom = dom[dom.case.between(args.case_min, args.case_max)]

    before = len(response)
    frame = response.merge(dom, on=KEY, how="inner", validate="one_to_one")
    dominance_coverage = len(frame) / before
    if dominance_coverage < 0.90:
        raise RuntimeError(
            f"dominance join covers only {dominance_coverage:.3%} of anchors")

    frame["prediction"] = frame[PRED]
    frame["truth"] = frame.R_blend_truth
    frame["gap"] = frame.prediction - frame.truth

    ratio = frame.dominant_to_runner_up_abs_response.to_numpy(float)
    tail = ratio > args.ratio_threshold
    cases = np.sort(frame.case.unique())

    payload = {
        "design": (
            "unpaired coherent-anchor central-neighbour response at |g|=0.02, "
            "split by the frozen single-pair response-dominance ratio; "
            "gap = V2.2 prediction minus sim truth; case is the uncertainty unit"
        ),
        "predeclared_readings": {
            "tail_dominated": "bulk gap within two case SEM of zero",
            "bulk_dominated": "bulk gap below -2 case SEM and carrying the "
                              "majority of the total contribution",
        },
        "case_window": [args.case_min, args.case_max],
        "n_cases": int(cases.size),
        "case_span": [int(cases.min()), int(cases.max())],
        "ratio_threshold": args.ratio_threshold,
        "n_anchors": int(len(frame)),
        "dominance_join_coverage": float(dominance_coverage),
        "restrict_keys": args.restrict_keys,
        "restricted_fraction": float(restricted_fraction),
        "all": group_summary(frame, np.ones(len(frame), bool)),
        "tail": group_summary(frame, tail),
        "rest": group_summary(frame, ~tail),
        "threshold_scan": {},
        "replicate_control": {},
    }
    total = payload["all"]["gap"]["mean"]
    if total != 0:
        for name in ("tail", "rest"):
            payload[name]["carrier_share"] = float(
                payload[name]["contribution"] / total)

    for threshold in args.scan:
        mask = ratio > threshold
        payload["threshold_scan"][f"gt_{threshold:g}"] = {
            "removed_fraction": float(mask.mean()),
            "kept_gap": group_summary(frame, ~mask)["gap"],
            "removed_gap": group_summary(frame, mask)["gap"],
        }

    low, high = args.replicate_window
    window = frame.case.between(low, high).to_numpy(bool)
    if window.any():
        control = frame.loc[window]
        control_ratio = control.dominant_to_runner_up_abs_response.to_numpy(float)
        payload["replicate_control"] = {
            "window": [low, high],
            "all": group_summary(control, np.ones(len(control), bool)),
            "tail": group_summary(control, control_ratio > args.ratio_threshold),
            "rest": group_summary(control, control_ratio <= args.ratio_threshold),
        }

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=1, sort_keys=True, allow_nan=False)
        handle.write("\n")
    for name in ("all", "tail", "rest"):
        block = payload[name]
        print(f"{name:5s} frac={block['fraction']:.4f} "
              f"gap={block['gap']['mean']:+.5f}+-{block['gap']['case_sem']:.5f} "
              f"t={block['t_gap_zero']:+.2f} "
              f"contribution={block['contribution']:+.5f} "
              f"share={block.get('carrier_share', float('nan')):+.3f}")
    print("WROTE", args.output)
    print("ANCHOR_G002_GAP_BY_DOMINANCE_DONE", flush=True)


if __name__ == "__main__":
    main()
