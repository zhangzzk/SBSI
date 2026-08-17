"""Does one common intrinsic cut close half-shear against constgold?

This is a simulation-truth diagnostic, not a model fit.  It independently
measures, on the same case window,

  * R_flow from the random-direction flow half-shear (g=0 -> +0.05),
  * summed R_blend from the BlendEMU response half-shear (g=0 -> +0.2), and
  * R_total from constgold's antithetic +/-0.02 render.

The ``common_lsst10`` arm calls exactly one shared truth-level mask for all
three catalogues.  The ``current`` arm reproduces the present mismatch:
flow/constgold use the historical primary + 5-arcsec-or-isolated cut, while the
R_blend catalogue uses the extended-neighbour 10-arcsec regression cuts.

No measured quantity enters either population mask.  Detection/ngmix
availability is applied afterwards because a response cannot be formed
without a shape; its attrition is printed separately.

FIREWALL: constgold is read only for the final diagnostic truth.  Nothing is
trained, tuned, or written into a model/target.
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.feather as pf
import pyarrow.ipc as ipc

from sbs_shear import domain
from sbs_shear.population import (EXTENDED_PAIR_CUTS, LSST_PAIR_CUTS, describe, pair_mask,
                                  primary_mask)


CAT = "/project/ls-gruen/users/zekang.zhang/sbsi_catalogues"
SIM = "/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876"
CONST = "/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant"
RESULTS = "/home/z/Zekang.Zhang/SBSI/results"
DUMPS = "/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v21_domain_dumps"

KEY = ["case", "input_index"]
COMMON = "common_lsst10"
CURRENT = "current"
COMMON_EXT = "common_extnbr10"
ARMS = (CURRENT, COMMON, COMMON_EXT)


def packed_key(case, index):
    return (np.asarray(case, np.int64) << 40) | np.asarray(index, np.int64)


def in_case_window(frame, lo, hi):
    case = frame["case"].to_numpy(np.int64)
    return (case >= lo) & (case <= hi)


def current_flow_mask(frame):
    """Historical flow/constgold truth cut: V2.1 plus 5 arcsec or isolated."""
    p = primary_mask(frame)
    distance = frame["distance"].to_numpy(float)
    neighbored = frame["neighbored"].fillna(False).astype(bool).to_numpy()
    return p & (((distance > 0.0) & (distance < 5.0)) | ~neighbored)


def current_rblend_mask(frame):
    """Current V2.1 emulator's extended-secondary, 10-arcsec support."""
    return pair_mask(frame, cuts=EXTENDED_PAIR_CUTS)


def masks_for_flow_or_const(frame):
    return {
        CURRENT: current_flow_mask(frame),
        COMMON: pair_mask(frame, cuts=LSST_PAIR_CUTS),
        COMMON_EXT: pair_mask(frame, cuts=EXTENDED_PAIR_CUTS),
    }


def masks_for_rblend(frame):
    extended = current_rblend_mask(frame)
    return {
        CURRENT: extended,
        COMMON: pair_mask(frame, cuts=LSST_PAIR_CUTS),
        COMMON_EXT: extended,
    }


def read_snc(path, lo, hi):
    d = pf.read_table(path, columns=["case", "input_index", "ngmix0_g1", "ngmix0_g2"],
                      memory_map=True).to_pandas()
    d = d[in_case_window(d, lo, hi)].copy()
    d["key"] = packed_key(d["case"], d["input_index"])
    if d["key"].duplicated().any():
        raise RuntimeError("g=0 SNC lookup contains duplicate keys")
    d = d.sort_values("key")
    return (d["key"].to_numpy(np.int64), d["ngmix0_g1"].to_numpy(float),
            d["ngmix0_g2"].to_numpy(float))


def exact_lookup(keys, lookup):
    lk, e1, e2 = lookup
    pos = np.searchsorted(lk, keys)
    clip = np.clip(pos, 0, max(len(lk) - 1, 0))
    hit = (pos < len(lk)) & (lk[clip] == keys)
    return hit, e1[clip], e2[clip]


def finish_primary_rows(parts, reduction):
    if not parts:
        return pd.DataFrame(columns=[*KEY, "response"])
    d = pd.concat(parts, ignore_index=True)
    if reduction == "sum":
        return d.groupby(KEY, sort=False, as_index=False)["response"].sum()
    if reduction == "mean":
        return d.groupby(KEY, sort=False, as_index=False)["response"].mean()
    raise ValueError(reduction)


def summarize(name, arm, rows, raw, truth_cut, available):
    by_case = rows.groupby("case", sort=True)["response"].mean()
    mean = float(rows["response"].mean())
    sem = float(by_case.std(ddof=1) / np.sqrt(len(by_case)))
    print(f"{name:<10} {arm:<14} raw={raw:>12,} truth-cut={truth_cut:>12,} "
          f"usable={available:>12,} primaries={len(rows):>10,} "
          f"R={mean:+.6f} caseSEM={sem:.6f}", flush=True)
    return dict(mean=mean, sem=sem, n=len(rows), cases=by_case)


def flow_truth(path, snc_path, lo, hi):
    lookup = read_snc(snc_path, lo, hi)
    cols = ["case", "input_index", "r_input_p", "Re_input_p", "r_input_s", "Re_input_s",
            "distance", "neighbored", "detected", "gamma1_input_p", "gamma2_input_p",
            "measured_ngmix_g1", "measured_ngmix_g2"]
    parts = {arm: [] for arm in ARMS}
    counts = {arm: [0, 0] for arm in ARMS}
    raw = 0
    with ipc.open_file(path) as reader:
        for bi in range(reader.num_record_batches):
            b = pa.Table.from_batches([reader.get_batch(bi)]).select(cols).to_pandas()
            raw += len(b)
            b = b[in_case_window(b, lo, hi)]
            if not len(b):
                continue
            g1 = b["gamma1_input_p"].to_numpy(float)
            g2 = b["gamma2_input_p"].to_numpy(float)
            gm = np.hypot(g1, g2)
            availability = (b["detected"].astype(bool).to_numpy() & np.isfinite(gm) & (gm > 1e-6)
                            & np.isfinite(b["measured_ngmix_g1"].to_numpy(float))
                            & np.isfinite(b["measured_ngmix_g2"].to_numpy(float)))
            masks = masks_for_flow_or_const(b)
            keys = packed_key(b["case"], b["input_index"])
            hit, e10, e20 = exact_lookup(keys, lookup)
            e1 = b["measured_ngmix_g1"].to_numpy(float)
            e2 = b["measured_ngmix_g2"].to_numpy(float)
            response = ((e1 - e10) * g1 + (e2 - e20) * g2) / np.maximum(gm * gm, 1e-30)
            for arm, truth in masks.items():
                counts[arm][0] += int(truth.sum())
                keep = truth & availability & hit & np.isfinite(response)
                counts[arm][1] += int(keep.sum())
                if keep.any():
                    parts[arm].append(pd.DataFrame({
                        "case": b.loc[keep, "case"].to_numpy(np.int64),
                        "input_index": b.loc[keep, "input_index"].to_numpy(np.int64),
                        "response": response[keep],
                    }))
    out = {}
    for arm in ARMS:
        rows = finish_primary_rows(parts[arm], "mean")
        out[arm] = summarize("R_flow", arm, rows, raw, counts[arm][0], counts[arm][1])
    return out


def rblend_truth(path, lo, hi, shear):
    cols = ["case", "input_index", "r_input_p", "Re_input_p", "r_input_s", "Re_input_s",
            "distance", "delta_et1"]
    parts = {arm: [] for arm in ARMS}
    counts = {arm: [0, 0] for arm in ARMS}
    raw = 0
    with ipc.open_file(path) as reader:
        for bi in range(reader.num_record_batches):
            b = pa.Table.from_batches([reader.get_batch(bi)]).select(cols).to_pandas()
            raw += len(b)
            b = b[in_case_window(b, lo, hi)]
            if not len(b):
                continue
            response = b["delta_et1"].to_numpy(float) / shear
            masks = masks_for_rblend(b)
            for arm, truth in masks.items():
                counts[arm][0] += int(truth.sum())
                keep = truth & np.isfinite(response)
                counts[arm][1] += int(keep.sum())
                if keep.any():
                    parts[arm].append(pd.DataFrame({
                        "case": b.loc[keep, "case"].to_numpy(np.int64),
                        "input_index": b.loc[keep, "input_index"].to_numpy(np.int64),
                        "response": response[keep],
                    }))
    out = {}
    for arm in ARMS:
        rows = finish_primary_rows(parts[arm], "sum")
        out[arm] = summarize("R_blend", arm, rows, raw, counts[arm][0], counts[arm][1])
    return out


def constgold_truth(path, lo, hi):
    cols = ["case", "input_index", "r_input_p", "Re_input_p", "r_input_s", "Re_input_s",
            "distance", "neighbored", "applied_g1", "applied_g2",
            "measured_e1_plus", "measured_e2_plus", "measured_e1_minus", "measured_e2_minus"]
    parts = {arm: [] for arm in ARMS}
    counts = {arm: [0, 0] for arm in ARMS}
    raw = 0
    with ipc.open_file(path) as reader:
        for bi in range(reader.num_record_batches):
            b = pa.Table.from_batches([reader.get_batch(bi)]).select(cols).to_pandas()
            raw += len(b)
            b = b[in_case_window(b, lo, hi)]
            if not len(b):
                continue
            g1 = b["applied_g1"].to_numpy(float)
            g2 = b["applied_g2"].to_numpy(float)
            gm = np.hypot(g1, g2)
            de1 = b["measured_e1_plus"].to_numpy(float) - b["measured_e1_minus"].to_numpy(float)
            de2 = b["measured_e2_plus"].to_numpy(float) - b["measured_e2_minus"].to_numpy(float)
            response = (de1 * g1 + de2 * g2) / np.maximum(2.0 * gm * gm, 1e-30)
            available = np.isfinite(response) & np.isfinite(gm) & (gm > 1e-6)
            masks = masks_for_flow_or_const(b)
            for arm, truth in masks.items():
                counts[arm][0] += int(truth.sum())
                keep = truth & available
                counts[arm][1] += int(keep.sum())
                if keep.any():
                    parts[arm].append(pd.DataFrame({
                        "case": b.loc[keep, "case"].to_numpy(np.int64),
                        "input_index": b.loc[keep, "input_index"].to_numpy(np.int64),
                        "response": response[keep],
                    }))
    out = {}
    for arm in ARMS:
        rows = finish_primary_rows(parts[arm], "mean")
        out[arm] = summarize("R_total", arm, rows, raw, counts[arm][0], counts[arm][1])
        out[arm]["rows"] = rows
    return out


def score_existing_model(dump_dir, tag, total, lo, hi):
    """Re-mask the existing 16-seed constgold dumps onto each truth population.

    This is the decisive end-to-end test for the reported bias.  Unlike adding
    response means from three different simulation designs, every component
    here is evaluated on the exact same constgold object keys.
    """
    paths = sorted(glob.glob(os.path.join(dump_dir, f"{tag}_perobj_s*.feather")))
    if len(paths) != 16:
        raise RuntimeError(f"expected 16 {tag} dumps, found {len(paths)}")
    result = {arm: dict(per_seed=[], comps=[], coverage=[], seeds=[]) for arm in ARMS}
    keeps = {
        arm: total[arm]["rows"][[*KEY, "response"]].rename(columns={"response": "r_sim_raw"})
        for arm in ARMS
    }
    print("\nEND-TO-END EXISTING MODEL, re-masked on exact constgold keys")
    # Paths are outermost so adding population arms does not re-read the 16 large dumps.
    for path in paths:
        d = pf.read_table(path, columns=[*KEY, "r_sim", "R_flow", "R_blend"],
                          memory_map=True).to_pandas()
        d = d[in_case_window(d, lo, hi)]
        seed = int(re.search(r"perobj_s(\d+)", os.path.basename(path)).group(1))
        for arm in ARMS:
            keep = keeps[arm]
            m = keep.merge(d, on=KEY, how="left", validate="one_to_one")
            hit = m["r_sim"].notna().to_numpy()
            result[arm]["coverage"].append(float(hit.mean()))
            if hit.mean() < 0.999:
                raise RuntimeError(f"{arm}: dump coverage {hit.mean():.4%} < 99.9% for {path}")
            m = m[hit]
            raw_delta = float(np.max(np.abs(m["r_sim"].to_numpy(float)
                                             - m["r_sim_raw"].to_numpy(float)), initial=0.0))
            if raw_delta > 2e-6:
                raise RuntimeError(f"{arm}: dump/raw constgold response differs by {raw_delta:.3e}")
            rs = float(m["r_sim"].mean())
            rf = float(m["R_flow"].mean())
            rb = float(m["R_blend"].mean())
            result[arm]["per_seed"].append(rs / (rf + rb) - 1.0)
            result[arm]["comps"].append((rs, rf, rb))
            result[arm]["seeds"].append(seed)
    for arm in ARMS:
        keep = keeps[arm]
        per_seed = np.asarray(result[arm]["per_seed"], float)
        comps = np.asarray(result[arm]["comps"], float)
        coverage = result[arm]["coverage"]
        seeds = result[arm]["seeds"]
        per_seed = np.asarray(per_seed, float)
        comps = np.asarray(comps, float)
        mean = float(per_seed.mean())
        sem = float(per_seed.std(ddof=1) / np.sqrt(len(per_seed)))
        cmean = comps.mean(axis=0)
        print(f"  {arm:<14} N={len(keep):,} coverage={min(coverage):.4%} "
              f"R_sim={cmean[0]:.6f} R_flow={cmean[1]:.6f} R_blend={cmean[2]:.6f} "
              f"m={mean:+.3%} +/- {sem:.3%}  seeds={seeds}")
        result[arm] = dict(m=mean, sem=sem, per_seed=per_seed, comps=cmean)
    dm = result[COMMON]["per_seed"] - result[CURRENT]["per_seed"]
    print(f"  paired LSST-cut effect: {dm.mean():+.3%} +/- "
          f"{dm.std(ddof=1)/np.sqrt(len(dm)):.3%} on m")
    dm_ext = result[COMMON_EXT]["per_seed"] - result[CURRENT]["per_seed"]
    print(f"  paired extended-cut effect: {dm_ext.mean():+.3%} +/- "
          f"{dm_ext.std(ddof=1)/np.sqrt(len(dm_ext)):.3%} on m")
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--flow", default=f"{CAT}/det_meas_crowd_g0.05_val_full.feather")
    ap.add_argument("--snc", default=f"{RESULTS}/g0_lookup_c0-99.feather")
    ap.add_argument("--rblend", default=f"{SIM}/response_catalogue_train.feather")
    ap.add_argument("--constgold", default=f"{CONST}/constant_response_catalogue_train.feather")
    ap.add_argument("--dump-dir", default=DUMPS)
    ap.add_argument("--dump-tag", default="ablate_s2c_lt500_v21")
    ap.add_argument("--min-case", type=int, default=40)
    ap.add_argument("--max-case", type=int, default=99)
    ap.add_argument("--rblend-shear", type=float, default=0.2)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    if os.path.exists(args.output):
        raise SystemExit(f"REFUSING to overwrite {args.output}")

    print(f"cases {args.min_case}..{args.max_case}")
    print(f"shared arm: {describe(LSST_PAIR_CUTS)}")
    print(f"shared extended control: {describe(EXTENDED_PAIR_CUTS)}")
    print("current arm: flow/constgold V2.1 + (<5 arcsec OR isolated); "
          "R_blend V2.1 + extended secondaries <10 arcsec")
    print("Population masks use intrinsic columns only. Detection/shape availability is counted after.")
    t0 = time.time()
    flow = flow_truth(args.flow, args.snc, args.min_case, args.max_case)
    blend = rblend_truth(args.rblend, args.min_case, args.max_case, args.rblend_shear)
    total = constgold_truth(args.constgold, args.min_case, args.max_case)

    saved = {}
    print("\nTRAINING-SIM COMPONENT SUM (diagnostic only: the R_blend simulation has a "
          "half-density neighbour pool, so this is not the end-to-end model closure)")
    for arm in ARMS:
        rf, rb, rt = flow[arm]["mean"], blend[arm]["mean"], total[arm]["mean"]
        model = rf + rb
        gap = rt - model
        m = rt / model - 1.0
        common_cases = sorted(set(flow[arm]["cases"].index)
                              & set(blend[arm]["cases"].index)
                              & set(total[arm]["cases"].index))
        dc = np.array([total[arm]["cases"].loc[c] - flow[arm]["cases"].loc[c]
                       - blend[arm]["cases"].loc[c] for c in common_cases])
        gap_sem = float(dc.std(ddof=1) / np.sqrt(len(dc)))
        print(f"  {arm:<14} R_flow={rf:.6f} R_blend={rb:.6f} sum={model:.6f} "
              f"R_const={rt:.6f} gap={gap:+.6f} +/- {gap_sem:.6f} "
              f"m_equiv={m:+.3%}")
        saved[arm] = np.array([rf, rb, model, rt, gap, gap_sem, m], float)

    print("\nDIRECT SIMULATION DIFFERENCE: constgold coherent total minus flow half-shear self")
    for arm in ARMS:
        gap = total[arm]["mean"] - flow[arm]["mean"]
        print(f"  {arm:<14} {gap:+.6f}")

    model = score_existing_model(args.dump_dir, args.dump_tag, total,
                                 args.min_case, args.max_case)
    if min(abs(model[COMMON]["m"]), abs(model[COMMON_EXT]["m"])) <= 0.003:
        verdict = "CONSISTENT CUTS EXPLAIN THE MODEL DISCREPANCY"
    else:
        verdict = "CONSISTENT CUTS DO NOT EXPLAIN THE MODEL DISCREPANCY"
    print(f"\nVERDICT: {verdict}")
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    np.savez(args.output, current=saved[CURRENT], common_lsst10=saved[COMMON],
             common_extnbr10=saved[COMMON_EXT],
             fields=np.array(["R_flow", "R_blend", "sum", "R_const", "gap", "gap_sem", "m_equiv"]),
             model_m_current=model[CURRENT]["per_seed"],
             model_m_common_lsst10=model[COMMON]["per_seed"],
             model_m_common_extnbr10=model[COMMON_EXT]["per_seed"],
             model_components_current=model[CURRENT]["comps"],
             model_components_common_lsst10=model[COMMON]["comps"],
             model_components_common_extnbr10=model[COMMON_EXT]["comps"],
             min_case=args.min_case, max_case=args.max_case,
             population=describe(LSST_PAIR_CUTS))
    print(f"saved {args.output}  elapsed={time.time()-t0:.1f}s")
    print("CONSISTENT_POPULATION_CLOSURE_DONE")


if __name__ == "__main__":
    main()
