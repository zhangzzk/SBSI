"""Same-object, distance-resolved decomposition of the V2.2 constgold gap.

WORKLOG 2026-08-13b compared the constgold total gap with the half-shear self gap
in fixed nearest-neighbour distance bins, but only CASE-paired: the two
detection-conditioned populations are not the same galaxies, so a per-bin
difference could be a population artefact.

This diagnostic repeats the comparison on EXACT ``(case, input_index)`` keys, and
splits the residual into the three terms of the 2026-08-11a identity

    R_flow^c + R_blend - R_sim
        = (R_flow^h - S_h)          flow-vs-half-shear-self-truth
        + (R_flow^c - R_flow^h)     extraction/realisation term
        + [R_blend - (R_sim - S_h)] blend/estimand term

evaluated inside the frozen nearest-annotated-distance groups.  Nothing is fitted
and no correction is applied; constgold is read for evaluation only.

Sign convention: the reported ``deficit`` columns are truth-minus-model, i.e.
``R_sim - (R_flow^c + R_blend)`` and its per-term pieces, so a POSITIVE number
means the model is too low.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
import pyarrow.feather as pf
from scipy import stats

KEYS = ["case", "input_index"]
SEEDS = [501, 502, 503, *range(505, 518)]


def stat(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=float)
    n = values.size
    sd = float(np.std(values, ddof=1)) if n > 1 else float("nan")
    return {
        "mean": float(np.mean(values)),
        "case_sd": sd,
        "case_sem": sd / np.sqrt(n) if n > 1 else float("nan"),
        "n_cases": int(n),
    }


def distance_masks(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    neighbored = frame.neighbored.to_numpy(bool)
    distance = frame.distance.to_numpy(float)
    return {
        "not_flagged": ~neighbored,
        "flagged_0_1_arcsec": neighbored & (distance >= 0.0) & (distance < 1.0),
        "flagged_1_2_arcsec": neighbored & (distance >= 1.0) & (distance < 2.0),
        "flagged_2_3p01_arcsec": neighbored & (distance >= 2.0) & (distance < 3.01),
    }


TERMS = {
    # truth-minus-model pieces; they add up to `total_deficit`
    "self_term": "S_h - R_flow^h  (flow vs direct half-shear self truth)",
    "extraction_term": "R_flow^h - R_flow^c  (constgold vs half-shear flow extraction)",
    "blend_term": "(R_sim - S_h) - R_blend  (blend estimand vs emulator)",
    "total_deficit": "R_sim - (R_flow^c + R_blend)",
}


def summarize(frame: pd.DataFrame, mask: np.ndarray) -> dict:
    local = frame.loc[mask]
    by_case = local.groupby("case", sort=True).agg(
        R_sim=("R_sim", "mean"),
        R_flow_c=("R_flow_c", "mean"),
        R_flow_h=("R_flow_h", "mean"),
        S_h=("S_h", "mean"),
        R_blend=("R_blend", "mean"),
        blend_demand=("blend_demand", "mean"),
        self_term=("self_term", "mean"),
        extraction_term=("extraction_term", "mean"),
        blend_term=("blend_term", "mean"),
        total_deficit=("total_deficit", "mean"),
        n_pairs=("n_pairs", "mean"),
    )
    out = {
        "n_rows": int(len(local)),
        "fraction": float(np.mean(mask)),
        "levels": {
            name: stat(by_case[name].to_numpy(float))
            for name in ("R_sim", "R_flow_c", "R_flow_h", "S_h", "R_blend",
                         "blend_demand", "n_pairs")
        },
        "deficits": {
            name: stat(by_case[name].to_numpy(float)) for name in TERMS
        },
    }
    demand = out["levels"]["blend_demand"]["mean"]
    emu = out["levels"]["R_blend"]["mean"]
    out["blend_underprediction_percent"] = (
        float(100.0 * (demand / emu - 1.0)) if emu != 0 else float("nan")
    )
    # contribution of this group to the global truth-minus-model deficit
    out["contribution"] = {
        name: float(out["fraction"] * out["deficits"][name]["mean"]) for name in TERMS
    }
    return out


def block(frame: pd.DataFrame) -> dict:
    masks = distance_masks(frame)
    out = {name: summarize(frame, mask) for name, mask in masks.items()}
    out["global"] = summarize(frame, np.ones(len(frame), dtype=bool))
    # paired 2-3 vs 1-2 contrast on the blend term, the quantity 08-13b could not
    # isolate because its two populations were not the same objects
    rows = []
    for name in ("flagged_1_2_arcsec", "flagged_2_3p01_arcsec"):
        values = frame.loc[masks[name]].groupby("case", sort=True).blend_term.mean()
        rows.append(values.rename(name))
    paired = pd.concat(rows, axis=1, join="inner")
    contrast = (paired["flagged_2_3p01_arcsec"].to_numpy(float)
                - paired["flagged_1_2_arcsec"].to_numpy(float))
    test = stats.ttest_1samp(contrast, popmean=0.0)
    out["blend_term_contrast_2_3_minus_1_2"] = {
        "gap": stat(contrast),
        "paired_t": float(test.statistic),
        "paired_p": float(test.pvalue),
    }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--constgold-features", required=True,
                    help="results/v22_constgold_gap_features_c40-139.feather")
    ap.add_argument("--half-selfresp", required=True,
                    help="half-shear per-object self truth + per-seed R_flow")
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--case-min", type=int, default=40)
    ap.add_argument("--case-max", type=int, default=139)
    ap.add_argument("--development-max", type=int, default=89)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    if os.path.exists(args.output):
        raise SystemExit(f"REFUSING to overwrite {args.output}")

    const = pf.read_table(
        args.constgold_features,
        columns=KEYS + ["R_sim", "R_flow", "R_blend", "neighbored", "distance",
                        "n_pairs"],
        memory_map=True,
    ).to_pandas()
    const = const[(const.case >= args.case_min) & (const.case <= args.case_max)]
    if const.duplicated(KEYS).any():
        raise RuntimeError("duplicate keys in constgold features")
    const = const.rename(columns={"R_flow": "R_flow_c"})

    flow_cols = [f"R_flow_s{seed}" for seed in args.seeds]
    half = pf.read_table(
        args.half_selfresp,
        columns=KEYS + ["r_sim_self"] + flow_cols,
        memory_map=True,
    ).to_pandas()
    half = half[(half.case >= args.case_min) & (half.case <= args.case_max)]
    if half.duplicated(KEYS).any():
        raise RuntimeError("duplicate keys in half-shear self response")
    half["R_flow_h"] = half[flow_cols].to_numpy(dtype=np.float64).mean(axis=1)
    half["S_h"] = half["r_sim_self"].astype(np.float64)
    half = half[KEYS + ["R_flow_h", "S_h"]]

    n_const, n_half = len(const), len(half)
    frame = const.merge(half, on=KEYS, how="inner", validate="one_to_one")
    finite = np.isfinite(frame[["R_sim", "R_flow_c", "R_blend", "R_flow_h",
                                "S_h"]].to_numpy(float)).all(axis=1)
    frame = frame.loc[finite].copy()

    frame["blend_demand"] = frame.R_sim - frame.S_h
    frame["self_term"] = frame.S_h - frame.R_flow_h
    frame["extraction_term"] = frame.R_flow_h - frame.R_flow_c
    frame["blend_term"] = frame.blend_demand - frame.R_blend
    frame["total_deficit"] = frame.R_sim - (frame.R_flow_c + frame.R_blend)

    identity = np.abs(
        frame.self_term + frame.extraction_term + frame.blend_term
        - frame.total_deficit
    ).max()
    if identity > 1e-9:
        raise RuntimeError(f"decomposition identity failed at {identity:g}")

    dev = frame[frame.case <= args.development_max]
    val = frame[frame.case > args.development_max]

    payload = {
        "design": (
            "exact (case,input_index) intersection of the V2.2 constgold domain "
            "and the fresh half-shear self-response dump; frozen nearest "
            "annotated distance bins; 16-seed mean R_flow on both sides; "
            "case is the uncertainty unit; truth-minus-model sign"
        ),
        "terms": TERMS,
        "seeds": list(args.seeds),
        "counts": {
            "constgold_rows": int(n_const),
            "halfshear_rows": int(n_half),
            "matched_finite_rows": int(len(frame)),
            "constgold_coverage": float(len(frame) / n_const),
            "halfshear_coverage": float(len(frame) / n_half),
        },
        "identity_max_abs_error": float(identity),
        "development_window": [args.case_min, args.development_max],
        "validation_window": [args.development_max + 1, args.case_max],
        "development": block(dev),
        "validation": block(val),
        "all_cases": block(frame),
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as handle:
        json.dump(payload, handle, indent=1, sort_keys=True)
    print(json.dumps(payload["validation"]["global"]["deficits"], indent=1))
    print("WROTE", args.output)


if __name__ == "__main__":
    main()
