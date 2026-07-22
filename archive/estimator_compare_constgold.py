#!/usr/bin/env python
"""
cont.99: estimator comparison for the certified flow-closure (SELECTION-FREE, matched constgold),
per cut, to test the owner's point that PER-OBJECT calibration (calibrate each galaxy by its own
predicted response, then average) beats the "harness" ratio-of-means.

Matched sample = the per-object dump (fig2_perobj_s501_fixresp: r_sim = per-object matched TRUE
response, R_flow, R_blend). Re_input_p joined from the +0.02 leg for size cuts. Reading r_sim is
VALIDATION (allowed); firewall = no TRAINING on constgold.

Per cut, the certified matched bias (no selection -- pairing cancels it) computed 3 ways:
  m_ratio  = <r_sim>/<R_flow+R_blend> - 1                 ratio-of-means ("harness"; = my step-2 +0.92%)
  m_perobj = < r_sim/(R_flow+R_blend) > - 1               PER-OBJECT calibrate-then-average (owner's)
  m_permed = median( r_sim/(R_flow+R_blend) ) - 1         robust per-object (outlier-proof)
(per-object variants guard |R_model|>0.1 to avoid divide-by-~0; frac kept reported)
If m_perobj/m_permed ~ +0.245% while m_ratio ~ +0.92%, per-object IS the better (certified) estimator.
Detection bias R_sel is reported separately (from the decomp) -- that's what a detection head removes.
"""
import os, numpy as np, pyarrow.feather as feather

DUMP = "/project/ls-gruen/users/zekang.zhang/sbsi_dumps/fig2_perobj_s501_fixresp.feather"
LEGP = "/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant/constant_shear_catalogue_0.02_train.feather"
OUT = "/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/estimator_compare.txt"
KEYMUL = np.uint64(800003)
GUARD = 0.1
NBOOT = 500

def main():
    td = feather.read_table(DUMP, columns=["case", "input_index", "r_input_p", "r_sim", "R_flow", "R_blend", "neighbored"])
    case = np.asarray(td["case"]).astype("i8")
    key = (np.asarray(td["case"]).astype(np.uint64) * KEYMUL + np.asarray(td["input_index"]).astype(np.uint64))
    mag = np.asarray(td["r_input_p"]).astype("f8")
    rsim = np.asarray(td["r_sim"]).astype("f8")
    rmod = np.asarray(td["R_flow"]).astype("f8") + np.asarray(td["R_blend"]).astype("f8")
    nbr = np.asarray(td["neighbored"]).astype(bool)
    print(f"[dump] {len(key):,} matched objects; <r_sim>={np.nanmean(rsim):.4f} <R_model>={np.nanmean(rmod):.4f}", flush=True)

    tl = feather.read_table(LEGP, columns=["case", "input_index", "Re_input_p"])
    lkey = (np.asarray(tl["case"]).astype(np.uint64) * KEYMUL + np.asarray(tl["input_index"]).astype(np.uint64))
    lre = np.asarray(tl["Re_input_p"]).astype("f8")
    o = np.argsort(lkey); lkey_s, lre_s = lkey[o], lre[o]
    idx = np.clip(np.searchsorted(lkey_s, key), 0, len(lkey_s) - 1)
    hit = lkey_s[idx] == key
    re = np.full(len(key), np.nan); re[hit] = lre_s[idx[hit]]
    print(f"[join] Re hit {hit.mean():.1%}", flush=True)

    good = np.isfinite(rsim) & np.isfinite(rmod)
    CUTS = [
        ("GLOBAL",               np.ones(len(mag), bool)),
        ("acc(mag<25 & Re>0.3)", (mag < 25) & (re > 0.3)),
        ("mag_lt24",             mag < 24),
        ("mag_lt25",             mag < 25),
        ("size_gt0.3",           re > 0.3),
        ("mag25-26",             (mag >= 25) & (mag < 26)),
        ("mag26-27(tail)",       (mag >= 26) & (mag < 27)),
        ("isolated",             ~nbr),
        ("blended",              nbr),
    ]
    cases = np.unique(case)
    rng = np.random.RandomState(99)

    hdr = (f"{'cut':24s}{'n':>11}{'<r_sim>':>9}{'<Rmod>':>9}{'m_ratio%':>10}{'m_perobj%':>11}"
           f"{'m_permed%':>11}{'keep%':>7}{'sem_r%':>8}")
    lines = [hdr, "-" * len(hdr)]
    for lab, cmask in CUTS:
        m = good & cmask
        cs = case[m]; rs = rsim[m]; rm = rmod[m]
        g = np.abs(rm) > GUARD
        ratio = rs[g] / rm[g]
        m_ratio = rs.sum() / rm.sum() - 1 if rm.sum() != 0 else np.nan
        m_perobj = np.mean(ratio) - 1 if g.sum() else np.nan
        m_permed = np.median(ratio) - 1 if g.sum() else np.nan
        # case bootstrap on the ratio-of-means (the primary; per-obj sem similar order)
        uc = np.unique(cs)
        sr = {int(c): (rs[cs == c].sum(), rm[cs == c].sum()) for c in uc} if len(uc) < 250 else None
        if sr is not None:
            boot = np.empty(NBOOT)
            for b in range(NBOOT):
                rc = uc[rng.randint(0, len(uc), len(uc))]
                SR = sum(sr[int(c)][0] for c in rc); SM = sum(sr[int(c)][1] for c in rc)
                boot[b] = (SR / SM - 1) if SM != 0 else np.nan
            sem_r = np.nanstd(boot) * 100
        else:
            sem_r = np.nan
        lines.append(f"{lab:24s}{m.sum():>11,}{rs.mean():>+9.4f}{rm.mean():>+9.4f}"
                     f"{m_ratio*100:>+10.2f}{m_perobj*100:>+11.2f}{m_permed*100:>+11.2f}"
                     f"{g.mean()*100:>7.1f}{sem_r:>8.2f}")
        print(lines[-1], flush=True)

    lines += ["",
              "READING: m_ratio = ratio-of-means ('harness', my step-2). m_perobj/m_permed = calibrate each",
              "galaxy by its OWN predicted response then average (owner's proposal). All THREE are the",
              "SELECTION-FREE flow-closure (matched pairing cancels detection). If per-object ~ +0.245% while",
              "ratio ~ +0.92%, per-object is the better estimator. Detection bias R_sel (from decomp, what a",
              "detection head removes): GLOBAL -4.0%, acc -0.6%, mag_lt24 -0.4%, mag_lt25 -0.7%, size_gt0.3 -3.4%."]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w").write("\n".join(lines) + "\n")
    print(f"\nwrote {OUT}\nESTCMP_DONE", flush=True)

if __name__ == "__main__":
    main()
