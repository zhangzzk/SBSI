#!/usr/bin/env python
"""
cont.100: the CERTIFIED 16-seed ENSEMBLE flow closure per cut, on the constgold MATCHED sample.

The certified global m=+0.245% is <r_sim>/(<R_flow_ens>+<R_blend>)-1 where R_flow_ens is the mean
over 16 seeds (PIPELINE.md:157; harvest logs). My earlier step-2/estcmp used the SINGLE s501 dump,
whose global R_flow is ~1sigma low -> inflates the closure to +0.91%. This script rebuilds the true
per-object 16-seed ensemble R_flow (from the s501..s516 dumps) and recomputes the flow closure PER
CUT, so we see whether the per-cut flow non-closure collapses toward sub-percent under the ensemble
(as the global does), or whether real per-cut structure (bright/large mis-calibration) survives.

Matched closure = residual bias AFTER a perfect detection head (pairing cancels detection selection).
Detection-selection bias R_sel (what a detection head removes) is quoted per cut from the step-1
decomp. Full unmatched ensemble bias ~ closure_ens + R_sel/R_model. Sub-percent on gentle cuts is
reached iff closure_ens is sub-percent there.

Firewall: reads r_sim/R_flow/R_blend for VALIDATION only; trains nothing on constgold.
"""
import os, glob, numpy as np, pyarrow.feather as feather

DDIR = "/project/ls-gruen/users/zekang.zhang/sbsi_dumps"
S501 = f"{DDIR}/fig2_perobj_s501_fixresp.feather"
LEGP = "/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant/constant_shear_catalogue_0.02_train.feather"
OUT = "/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/ensemble_flow_closure.txt"
KEYMUL = np.uint64(800003)
NBOOT = 500
# step-1 decomp R_sel bias (m_cert = R_sel/R_shape, %), for combining with the closure:
RSEL_PCT = {"GLOBAL": -4.04, "acc(mag<25 & Re>0.3)": -0.63, "mag_lt24": -0.40,
            "mag_lt25": -0.72, "size_gt0.3": -3.40}

def load_key(path):
    t = feather.read_table(path, columns=["case", "input_index"])
    return (np.asarray(t["case"]).astype(np.uint64) * KEYMUL
            + np.asarray(t["input_index"]).astype(np.uint64))

def main():
    # base (seed-independent) columns from s501
    t = feather.read_table(S501, columns=["case", "input_index", "r_input_p", "r_sim", "R_blend", "neighbored"])
    case = np.asarray(t["case"]).astype("i8")
    key0 = np.asarray(t["case"]).astype(np.uint64) * KEYMUL + np.asarray(t["input_index"]).astype(np.uint64)
    mag = np.asarray(t["r_input_p"]).astype("f8")
    rsim = np.asarray(t["r_sim"]).astype("f8")
    rbl = np.asarray(t["R_blend"]).astype("f8")
    nbr = np.asarray(t["neighbored"]).astype(bool)
    N = len(key0)
    print(f"[base] s501 {N:,} matched objects; <r_sim>={np.nanmean(rsim):.4f} <R_blend>={np.nanmean(rbl):.4f}", flush=True)

    # gather all seed dumps, aligned to s501 row order (verify key; searchsorted-join if not aligned)
    seeds = sorted(int(p.split("_s")[1].split("_")[0]) for p in glob.glob(f"{DDIR}/fig2_perobj_s5*_fixresp.feather"))
    print(f"[seeds] found dumps for {seeds}", flush=True)
    o0 = np.argsort(key0); key0_s = key0[o0]
    rflow_stack = np.empty((len(seeds), N), dtype="f8")
    rf501 = None
    for si, s in enumerate(seeds):
        p = f"{DDIR}/fig2_perobj_s{s}_fixresp.feather"
        tk = feather.read_table(p, columns=["case", "input_index", "R_flow"])
        ks = np.asarray(tk["case"]).astype(np.uint64) * KEYMUL + np.asarray(tk["input_index"]).astype(np.uint64)
        rf = np.asarray(tk["R_flow"]).astype("f8")
        if len(ks) == N and np.array_equal(ks, key0):
            rflow_stack[si] = rf                              # row-aligned (deterministic first-N)
            aligned = "row-aligned"
        else:
            idx = np.clip(np.searchsorted(key0_s, ks), 0, N - 1)   # map seed rows -> base positions
            hit = key0_s[idx] == ks
            col = np.full(N, np.nan)
            col[o0[idx[hit]]] = rf[hit]
            rflow_stack[si] = col
            aligned = f"joined ({hit.mean():.1%} hit)"
        if s == 501:
            rf501 = rflow_stack[si].copy()
        print(f"  s{s}: <R_flow>={np.nanmean(rflow_stack[si]):.4f}  [{aligned}]", flush=True)

    rflow_ens = np.nanmean(rflow_stack, axis=0)              # per-object 16-seed ensemble
    print(f"[ensemble] {len(seeds)} seeds; <R_flow_ens>={np.nanmean(rflow_ens):.4f} "
          f"(s501 <R_flow>={np.nanmean(rf501):.4f})", flush=True)

    # size from the +0.02 leg
    tl = feather.read_table(LEGP, columns=["case", "input_index", "Re_input_p"])
    lkey = np.asarray(tl["case"]).astype(np.uint64) * KEYMUL + np.asarray(tl["input_index"]).astype(np.uint64)
    lre = np.asarray(tl["Re_input_p"]).astype("f8")
    ol = np.argsort(lkey); lkey_s, lre_s = lkey[ol], lre[ol]
    j = np.clip(np.searchsorted(lkey_s, key0), 0, len(lkey_s) - 1)
    hitre = lkey_s[j] == key0
    re = np.full(N, np.nan); re[hitre] = lre_s[j[hitre]]
    print(f"[join] Re hit {hitre.mean():.1%}\n", flush=True)

    CUTS = [
        ("GLOBAL",               np.ones(N, bool)),
        ("acc(mag<25 & Re>0.3)", (mag < 25) & (re > 0.3)),
        ("mag_lt24",             mag < 24),
        ("mag_lt25",             mag < 25),
        ("size_gt0.3",           re > 0.3),
        ("mag25-26",             (mag >= 25) & (mag < 26)),
        ("mag26-27(tail)",       (mag >= 26) & (mag < 27)),
        ("isolated",             ~nbr),
        ("blended",              nbr),
    ]
    good = np.isfinite(rsim) & np.isfinite(rbl) & np.isfinite(rflow_ens) & np.isfinite(rf501)
    uc_all = np.unique(case)
    rng = np.random.RandomState(2025)

    def closure(mask, rflow):
        num = rsim[mask].sum(); den = (rflow[mask] + rbl[mask]).sum()
        return num / den - 1 if den != 0 else np.nan

    hdr = (f"{'cut':22s}{'n':>11}{'Rflow_s501':>11}{'Rflow_ens':>11}"
           f"{'m_s501%':>9}{'m_ens%':>9}{'sem%':>7}{'seedSD%':>8}{'R_sel%':>8}{'m_full%':>8}")
    lines = [hdr, "-" * len(hdr)]
    for lab, cmask in CUTS:
        m = good & cmask
        rf_e = float(np.mean(rflow_ens[m])); rf_1 = float(np.mean(rf501[m]))
        m_s501 = closure(m, rf501) * 100
        m_ens = closure(m, rflow_ens) * 100
        # per-seed per-cut m -> scatter across seeds (how much the ensemble averages down)
        per_seed = np.array([closure(m, rflow_stack[si]) for si in range(len(seeds))]) * 100
        seedsd = float(np.nanstd(per_seed))
        # case bootstrap on the ensemble closure
        cs = case[m]; rs = rsim[m]; rmod = rflow_ens[m] + rbl[m]
        uc = np.unique(cs)
        sums = {int(c): (rs[cs == c].sum(), rmod[cs == c].sum()) for c in uc}
        boot = np.empty(NBOOT)
        for b in range(NBOOT):
            rc = uc[rng.randint(0, len(uc), len(uc))]
            SR = sum(sums[int(c)][0] for c in rc); SM = sum(sums[int(c)][1] for c in rc)
            boot[b] = (SR / SM - 1) if SM != 0 else np.nan
        sem = float(np.nanstd(boot) * 100)
        rsel = RSEL_PCT.get(lab, np.nan)
        m_full = m_ens + rsel if np.isfinite(rsel) else np.nan   # closure + selection (approx additive)
        lines.append(f"{lab:22s}{m.sum():>11,}{rf_1:>+11.4f}{rf_e:>+11.4f}"
                     f"{m_s501:>+9.2f}{m_ens:>+9.2f}{sem:>7.2f}{seedsd:>8.2f}"
                     f"{rsel:>+8.2f}{m_full:>+8.2f}")
        print(lines[-1], flush=True)

    lines += ["",
              "m_s501  = single-seed (s501) matched closure  = my earlier step-2/estcmp number (inflated).",
              "m_ens   = 16-seed ENSEMBLE matched closure     = residual bias AFTER a perfect detection head.",
              "seedSD  = std of the per-seed per-cut closure   (ensemble averages this down by ~1/sqrt(16)).",
              "R_sel   = detection-selection bias (step-1 decomp) = what a detection head REMOVES.",
              "m_full  = m_ens + R_sel = full unmatched ensemble bias (approx additive decomposition).",
              "SUB-PERCENT via detection head iff m_ens is sub-percent on the gentle cuts."]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w").write("\n".join(lines) + "\n")
    print(f"\nwrote {OUT}\nENSEMBLE_CLOSURE_DONE", flush=True)

if __name__ == "__main__":
    main()
