"""Split the paired coherent-anchor amplitude test by single-pair response dominance.

08-12o showed that removing `max|R_pair| / runner-up|R_pair| > 20` closes the whole
coherent-anchor gap at `g=0.05`, while 08-13a showed the identical cut removes only
~22% of the constgold response deficit.  The two experiments differ in shear
amplitude: constgold is antithetic at `|g|=0.02`, the anchor block at `|g|=0.05`.

This diagnostic reruns the already-rendered `g=0.02` / `g=0.05` paired anchor
comparison (08-12h) *inside* the frozen dominance groups, on exact
`(case, input_index)` keys.  Nothing is fitted, no correction is applied and
constgold is not opened.

Predeclared reading, fixed before the numbers were seen:

* if the TAIL gap at `g=0.02` is smaller in absolute value than half its `g=0.05`
  value, the anchor localization is amplitude-driven -- a finite-difference
  artefact of measuring a dominant close companion across `+-0.05` -- and it does
  not transfer to constgold's `|g|=0.02` regime;
* if the tail gap is consistent with its `g=0.05` value within two paired case
  SEM, the tail deficit is real at constgold's amplitude and the anchor/constgold
  disagreement must live on the constgold side of the comparison instead.

The complementary NON-tail statistic is reported with the same two readings, since
the global `g=0.02` gap is about twice the `g=0.05` gap and something must carry
that growth.
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
    case = local.groupby("case", sort=True)[
        ["gap_g002", "gap_g005", "gap_g002_minus_g005",
         "truth_g002", "truth_g005", "prediction"]
    ].mean()
    out = {
        "n_rows": int(len(local)),
        "fraction": float(mask.mean()),
        "gap_g002": stat(case.gap_g002.to_numpy(float)),
        "gap_g005": stat(case.gap_g005.to_numpy(float)),
        "gap_g002_minus_g005": stat(case.gap_g002_minus_g005.to_numpy(float)),
        "truth_g002": stat(case.truth_g002.to_numpy(float)),
        "truth_g005": stat(case.truth_g005.to_numpy(float)),
        "prediction": stat(case.prediction.to_numpy(float)),
    }
    paired = stats.ttest_1samp(case.gap_g002_minus_g005.to_numpy(float), 0.0)
    out["paired_p_g002_minus_g005"] = float(paired.pvalue)
    for amplitude in ("g002", "g005"):
        out[f"contribution_{amplitude}"] = float(
            out["fraction"] * out[f"gap_{amplitude}"]["mean"]
        )
    low = abs(out["gap_g002"]["mean"])
    high = abs(out["gap_g005"]["mean"])
    sem = out["gap_g002_minus_g005"]["case_sem"]
    out["gates"] = {
        "amplitude_driven_abs_g002_below_half_g005": bool(low < 0.5 * high),
        "persistent_within_2sem": bool(
            abs(out["gap_g002_minus_g005"]["mean"]) <= 2.0 * sem
        ),
    }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--g002", required=True)
    ap.add_argument("--g005", required=True)
    ap.add_argument("--dominance", required=True,
                    help="per-anchor dominance table with the response-ratio column")
    ap.add_argument("--ratio-threshold", type=float, default=20.0)
    ap.add_argument("--scan", type=float, nargs="+",
                    default=[5.0, 10.0, 20.0, 40.0, 100.0])
    ap.add_argument("--case-min", type=int, default=400)
    ap.add_argument("--case-max", type=int, default=499)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    if os.path.exists(args.output):
        raise SystemExit(f"REFUSING to overwrite {args.output}")

    columns = KEY + ["R_blend_truth", PRED]
    low = pd.read_feather(args.g002, columns=columns)
    high = pd.read_feather(args.g005, columns=columns)
    low = low[low.case.between(args.case_min, args.case_max)]
    high = high[high.case.between(args.case_min, args.case_max)]
    if low.duplicated(KEY).any() or high.duplicated(KEY).any():
        raise RuntimeError("duplicate response keys")
    frame = low.merge(high, on=KEY, suffixes=("_g002", "_g005"), validate="one_to_one")
    coverage = min(len(frame) / len(low), len(frame) / len(high))
    if coverage < 0.95:
        raise RuntimeError(f"common response coverage below 95%: {coverage:.3%}")

    replay = float(np.max(np.abs(frame[f"{PRED}_g002"] - frame[f"{PRED}_g005"])))
    if replay > 2e-7:
        raise RuntimeError(f"paired V2.2 predictions differ by {replay:.3e}")

    dom = pd.read_feather(
        args.dominance,
        columns=["case", "anchor_index", "dominant_to_runner_up_abs_response",
                 "dominant_abs_response", "R_abs_sum"],
    ).rename(columns={"anchor_index": "input_index"})
    dom = dom[dom.case.between(args.case_min, args.case_max)]
    if dom.duplicated(KEY).any():
        raise RuntimeError("duplicate dominance keys")

    before = len(frame)
    frame = frame.merge(dom, on=KEY, how="inner", validate="one_to_one")
    dominance_coverage = len(frame) / before
    if dominance_coverage < 0.90:
        raise RuntimeError(
            f"dominance join covers only {dominance_coverage:.3%} of paired anchors"
        )

    frame["prediction"] = frame[f"{PRED}_g005"]
    frame["truth_g002"] = frame.R_blend_truth_g002
    frame["truth_g005"] = frame.R_blend_truth_g005
    frame["gap_g002"] = frame.prediction - frame.truth_g002
    frame["gap_g005"] = frame.prediction - frame.truth_g005
    frame["gap_g002_minus_g005"] = frame.gap_g002 - frame.gap_g005

    ratio = frame.dominant_to_runner_up_abs_response.to_numpy(float)
    tail = ratio > args.ratio_threshold
    payload = {
        "design": (
            "paired coherent-anchor central neighbour response at |g|=0.02 and "
            "|g|=0.05 on exact (case,input_index) keys, split by the frozen "
            "single-pair response-dominance ratio; gap = V2.2 prediction minus "
            "sim truth; case is the uncertainty unit"
        ),
        "predeclared_readings": {
            "amplitude_driven": "abs(gap_g002) < 0.5 * abs(gap_g005) in the tail",
            "persistent": "gap_g002 - gap_g005 within two paired case SEM",
        },
        "case_window": [args.case_min, args.case_max],
        "ratio_threshold": args.ratio_threshold,
        "n_common": int(len(frame)),
        "response_common_coverage": float(coverage),
        "dominance_join_coverage": float(dominance_coverage),
        "prediction_replay_max_abs": replay,
        "all": group_summary(frame, np.ones(len(frame), bool)),
        "tail": group_summary(frame, tail),
        "rest": group_summary(frame, ~tail),
        "threshold_scan": {},
    }
    for threshold in args.scan:
        mask = ratio > threshold
        kept = ~mask
        payload["threshold_scan"][f"gt_{threshold:g}"] = {
            "removed_fraction": float(mask.mean()),
            "kept_gap_g002": group_summary(frame, kept)["gap_g002"],
            "kept_gap_g005": group_summary(frame, kept)["gap_g005"],
        }

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=1, sort_keys=True, allow_nan=False)
        handle.write("\n")
    for name in ("all", "tail", "rest"):
        block = payload[name]
        print(f"{name:5s} frac={block['fraction']:.4f} "
              f"gap_g005={block['gap_g005']['mean']:+.5f}+-{block['gap_g005']['case_sem']:.5f} "
              f"gap_g002={block['gap_g002']['mean']:+.5f}+-{block['gap_g002']['case_sem']:.5f} "
              f"diff={block['gap_g002_minus_g005']['mean']:+.5f}"
              f"+-{block['gap_g002_minus_g005']['case_sem']:.5f} "
              f"gates={block['gates']}")
    print("WROTE", args.output)
    print("ANCHOR_AMPLITUDE_BY_DOMINANCE_DONE", flush=True)


if __name__ == "__main__":
    main()
