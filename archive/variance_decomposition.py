#!/usr/bin/env python3
"""Honest error budget for the certified m = R_sim/(R_flow+R_blend)-1.

Motivated by the cont.49 pipeline review (CONFIRMED finding, PIPELINE.md:154):
the quoted +/-0.25% is std/sqrt(N) over TRAINING SEEDS and captures only the
init/SGD scatter of R_flow. It omits the common-mode HARVEST sampling floor --
the variance of the global R_flow (and R_sim, R_blend) under resampling the finite
set of certification cases -- which is identical for every seed and does NOT
average down with more seeds. This script quantifies BOTH components from the
per-object dump and the per-seed CRN R_flow globals, and combines them.

It also runs the response-penalty in-sample vs out-of-sample test (finding
job_pilot_harvest.sh:29): the response penalty target spans cases c0-99, so
cases 40-99 are penalty-in-sample and 100-139 are penalty-OOS (still inside the
flow's NLL training range). If m is stable across the split, the penalty is not
overfitting the certification cases.

Inputs
------
--dump      per-object feather from validate_constant_with_blend.py --dump
            (columns: case,input_index,r_input_p,r_sim,R_flow,R_blend,neighbored,distance)
            R_flow here is per-object (mp-mm)/(2g); its case-mean is the global.
--rblend    scalar certified R_blend (default 0.1593) -- used only as a cross-check;
            the dump's own R_blend column is what enters the budget.
--seeds-glob R_flow globals across seeds, parsed from harvest logs (CRN flow-seed 12345 only).

All resampling is over CASES (the independent unit; objects within a case are
spatially correlated by construction). Parameter-free: no fitted scalars.
"""
import argparse, glob, os, re
import numpy as np
import pandas as pd


def per_case_sums(df):
    """Return arrays keyed by case: sum r_sim, sum R_flow, sum R_blend, count."""
    g = df.groupby("case")
    agg = g.agg(rs=("r_sim", "sum"), rf=("R_flow", "sum"),
                rb=("R_blend", "sum"), n=("r_sim", "size"))
    return agg


def m_from_sums(agg, cases):
    n = agg.loc[cases, "n"].to_numpy(float)
    N = n.sum()
    R_sim = agg.loc[cases, "rs"].to_numpy(float).sum() / N
    R_flow = agg.loc[cases, "rf"].to_numpy(float).sum() / N
    R_blend = agg.loc[cases, "rb"].to_numpy(float).sum() / N
    return R_sim, R_flow, R_blend, R_sim / (R_flow + R_blend) - 1.0


def bootstrap(agg, cases, n_boot=2000, seed=0, resample_rflow=True):
    """Case bootstrap. If resample_rflow=False, R_flow held at its full-sample
    value (reproduces the current harvest bootstrap that OMITS the R_flow floor)."""
    rng = np.random.default_rng(seed)
    cases = np.asarray(cases)
    rs = agg.loc[cases, "rs"].to_numpy(float)
    rf = agg.loc[cases, "rf"].to_numpy(float)
    rb = agg.loc[cases, "rb"].to_numpy(float)
    n = agg.loc[cases, "n"].to_numpy(float)
    Rf_full = rf.sum() / n.sum()
    out = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, len(cases), len(cases))
        N = n[idx].sum()
        R_sim = rs[idx].sum() / N
        R_blend = rb[idx].sum() / N
        R_flow = rf[idx].sum() / N if resample_rflow else Rf_full
        out[b] = R_sim / (R_flow + R_blend) - 1.0
    return out


def jackknife_component(agg, cases, key):
    """Delete-one-case jackknife std of the global mean of `key` (rs/rf/rb)."""
    cases = np.asarray(cases)
    v = agg.loc[cases, key].to_numpy(float)
    n = agg.loc[cases, "n"].to_numpy(float)
    tot_v, tot_n = v.sum(), n.sum()
    loo = (tot_v - v) / (tot_n - n)              # leave-one-case-out global means
    m = len(cases)
    return np.sqrt((m - 1) / m * np.sum((loo - loo.mean()) ** 2))


def parse_seed_rflow(patterns):
    """Scan harvest logs; return {seed: R_flow} for FIXED-convention CRN rows
    (R_sim=0.4534). Last occurrence per seed wins (latest harvest)."""
    rx = re.compile(r"\[s(\d+)\] GLOBAL:\s+R_sim=0\.4534\s+R_flow\(self\)=([0-9.]+)")
    vals = {}
    for pat in patterns:
        for fn in glob.glob(pat):
            try:
                for line in open(fn):
                    mobj = rx.search(line)
                    if mobj:
                        vals[int(mobj.group(1))] = float(mobj.group(2))
            except OSError:
                pass
    return vals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default="/project/ls-gruen/users/zekang.zhang/sbsi_dumps/fig2_perobj_s501_fixresp.feather")
    ap.add_argument("--rblend", type=float, default=0.1593)
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--penalty-max-case", type=int, default=100,
                    help="cases < this are response-penalty in-sample (target npz spans c0-99)")
    ap.add_argument("--seeds-glob", nargs="*",
                    default=["/home/z/Zekang.Zhang/logs/pilot_harvest_*.out"])
    args = ap.parse_args()

    print("=" * 78)
    print("HONEST ERROR BUDGET FOR CERTIFIED m  (parameter-free, case-resampled)")
    print("=" * 78)

    # ---- seed component (init/SGD scatter of R_flow) -----------------------
    seed_rf = parse_seed_rflow(args.seeds_glob)
    if seed_rf:
        rf = np.array(sorted(seed_rf.values()))
        seeds = sorted(seed_rf)
        print(f"\n[SEED COMPONENT]  N_seeds={len(rf)}  seeds={seeds}")
        print(f"  R_flow per seed: {np.array([seed_rf[s] for s in seeds]).round(4).tolist()}")
        print(f"  mean R_flow = {rf.mean():.4f}   std = {rf.std(ddof=1):.4f}   "
              f"sem = std/sqrt(N) = {rf.std(ddof=1)/np.sqrt(len(rf)):.4f}")
        Rf_seed_mean, Rf_seed_sem, Rf_seed_std = rf.mean(), rf.std(ddof=1)/np.sqrt(len(rf)), rf.std(ddof=1)
    else:
        print("\n[SEED COMPONENT]  no harvest logs parsed; skipping seed term")
        Rf_seed_mean = Rf_seed_sem = Rf_seed_std = None

    # ---- harvest floor + full budget from the per-object dump --------------
    if not os.path.exists(args.dump):
        print(f"\n[DUMP] not found yet: {args.dump}\n  -> run the fixed fig2 dump job first; rerun this script when it lands.")
        return
    df = pd.read_feather(args.dump)
    print(f"\n[DUMP] {args.dump}  rows={len(df):,}  cases={df['case'].nunique()} "
          f"({df['case'].min()}..{df['case'].max()})")
    if len(df) == 0:
        print("  EMPTY dump -> the max-rows/min-case bug bit again. Abort.")
        return
    agg = per_case_sums(df)
    allc = agg.index.to_numpy()

    R_sim, R_flow, R_blend, m = m_from_sums(agg, allc)
    print(f"\n[POINT ESTIMATE, this model]  R_sim={R_sim:.4f}  R_flow={R_flow:.4f}  "
          f"R_blend={R_blend:.4f}  ->  m = {m:+.2%}")

    s_rsim = jackknife_component(agg, allc, "rs")
    s_rflow = jackknife_component(agg, allc, "rf")
    s_rblend = jackknife_component(agg, allc, "rb")
    print(f"\n[HARVEST FLOOR — delete-one-case jackknife of each global]")
    print(f"  sigma(R_sim)  = {s_rsim:.4f}")
    print(f"  sigma(R_flow) = {s_rflow:.4f}   <-- THE OMITTED FLOOR (bootstrap holds this fixed)")
    print(f"  sigma(R_blend)= {s_rblend:.4f}")

    boot_full = bootstrap(agg, allc, args.n_boot, resample_rflow=True)
    boot_noRf = bootstrap(agg, allc, args.n_boot, resample_rflow=False)
    print(f"\n[m ERROR — case bootstrap, {args.n_boot} resamples]")
    print(f"  current harvest (R_flow FIXED) : m = {m:+.2%} +/- {boot_noRf.std():.2%}")
    print(f"  HONEST (R_flow resampled too)  : m = {m:+.2%} +/- {boot_full.std():.2%}")

    # combine seed SEM (init/SGD) with harvest floor in R_flow, propagate to m
    if Rf_seed_mean is not None:
        dm_dRf = -R_sim / (R_flow + R_blend) ** 2          # dm/dR_flow at the point
        s_m_seed = abs(dm_dRf) * Rf_seed_sem               # init/SGD term (averages down)
        s_m_floor = boot_full.std()                        # harvest floor (does NOT)
        s_m_tot = np.hypot(s_m_seed, s_m_floor)
        # significance of the residual using the honest total
        m_seed = R_sim / (Rf_seed_mean + R_blend) - 1.0
        print(f"\n[COMBINED BUDGET]")
        print(f"  m at seed-mean R_flow={Rf_seed_mean:.4f}: m = {m_seed:+.2%}")
        print(f"  sigma_m(seed init/SGD, ->0 with N) = {s_m_seed:.2%}")
        print(f"  sigma_m(harvest floor, fixed)      = {s_m_floor:.2%}")
        print(f"  sigma_m(TOTAL, honest)             = {s_m_tot:.2%}")
        print(f"  ==> residual significance = m/sigma_tot = {m_seed/s_m_tot:.2f} sigma")
        verdict = "consistent with ZERO (fluctuation)" if abs(m_seed) < 2*s_m_tot else "a DETECTED residual bias"
        print(f"  ==> {verdict} at |m| = {abs(m_seed):.2%}, total sigma = {s_m_tot:.2%}")

    # ---- response-penalty in-sample vs OOS ---------------------------------
    pmc = args.penalty_max_case
    ins = allc[allc < pmc]
    oos = allc[allc >= pmc]
    print(f"\n[RESPONSE-PENALTY IN-SAMPLE vs OOS]  (penalty target spans c0-{pmc-1})")
    if len(ins) and len(oos):
        for name, cc in [("in-sample cases {}<=c<{}".format(int(allc.min()), pmc), ins),
                         ("OOS       cases c>={}".format(pmc), oos)]:
            Rs, Rf, Rb, mm = m_from_sums(agg, cc)
            bt = bootstrap(agg, cc, args.n_boot, resample_rflow=True)
            print(f"  {name:34s} n_cases={len(cc):3d}  R_flow={Rf:.4f}  m={mm:+.2%} +/- {bt.std():.2%}")
        _, Rf_in, _, m_in = m_from_sums(agg, ins)
        _, Rf_oos, _, m_oos = m_from_sums(agg, oos)
        print(f"  delta(OOS - in-sample): R_flow {Rf_oos-Rf_in:+.4f}, m {m_oos-m_in:+.2%}")
        print(f"  -> {'STABLE (penalty not overfitting)' if abs(m_oos-m_in) < 0.005 else 'SHIFT >0.5% (possible in-sample optimism)'}")
    else:
        print(f"  need cases on both sides of {pmc}; have in={len(ins)} oos={len(oos)}")

    print("\n" + "=" * 78)


if __name__ == "__main__":
    main()
