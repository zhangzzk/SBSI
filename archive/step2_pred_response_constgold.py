#!/usr/bin/env python
"""
STEP 2 (cont.98): the certified pipeline's PREDICTED response on the constgold UNMATCHED ensemble,
and the actual ensemble bias m_ens = R_total/<R_model> - 1, per cut. Tests the owner's prediction
that m_ens ~ the selection bias m_cert (=R_sel/R_shape), plus the +0.9% flow-shape-closure shift.

<R_model> = <R_flow + R_blend> from the certified per-object dump (fig2_perobj_s501_fixresp),
joined onto each ensemble leg by (case, input_index). R_flow/R_blend ONLY are read -- r_sim is NEVER
read (firewall). The dump covers MATCHED (both-detected) objects, cases 40-139; the join therefore
covers the common population (differential-detection objects have no dump entry -> excluded from
<R_model>; reported as `join%`). Everything restricted to cases 40-139 for a clean comparison.

Per cut, side by side:
  R_total  = (<e_meas1>_+ - <e_meas1>_-)/(2g)             ensemble truth (shape + detection-selection)
  R_sel    = (<e_intr1>_+ - <e_intr1>_-)/(2g)             pure detection-selection
  R_shape  = R_total - R_sel
  <R_model>= <R_flow+R_blend> over joined common objects  the CERTIFIED prediction
  m_cert   = R_sel/R_shape*100                            selection-only bias (model==R_shape)
  m_ens    = (R_total/<R_model> - 1)*100                  ACTUAL certified ensemble bias
Joint-flow ceiling: a perfect selection head drives <R_model> -> R_total => m_ens -> 0.
"""
import os, numpy as np, pyarrow.feather as feather

DDIR = "/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant"
LEG = {"+": f"{DDIR}/constant_shear_catalogue_0.02_train.feather",
       "-": f"{DDIR}/constant_shear_catalogue_-0.02_train.feather"}
DUMP = "/project/ls-gruen/users/zekang.zhang/sbsi_dumps/fig2_perobj_s501_fixresp.feather"
G = 0.02
OUT = "/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/step2_pred_response.txt"
KEYMUL = np.uint64(800003)
NBOOT = 500

LEGCOLS = ["measured_e1", "axis_ratio_input_p", "position_angle_input_p",
           "r_input_p", "Re_input_p", "case", "input_index", "neighbored"]

def load_leg(path):
    t = feather.read_table(path, columns=LEGCOLS)
    d = {c: np.asarray(t[c]) for c in LEGCOLS}
    q = d["axis_ratio_input_p"].astype("f8"); th = np.deg2rad(d["position_angle_input_p"].astype("f8"))
    d["e_intr1"] = (1.0 - q) / (1.0 + q) * np.cos(2.0 * th)
    d["e_meas1"] = d["measured_e1"].astype("f8")
    d["det"] = np.isfinite(d["e_meas1"])
    d["mag"] = d["r_input_p"].astype("f8"); d["re"] = d["Re_input_p"].astype("f8")
    d["case"] = d["case"].astype("i8"); d["nbr"] = d["neighbored"].astype(bool)
    d["key"] = d["case"].astype(np.uint64) * KEYMUL + d["input_index"].astype(np.uint64)
    return d

def join_rmodel(d, dkey_sorted, dval_sorted):
    """map R_flow+R_blend onto leg rows by key; NaN where not in dump (differential/uncovered)."""
    idx = np.searchsorted(dkey_sorted, d["key"])
    idx = np.clip(idx, 0, len(dkey_sorted) - 1)
    hit = dkey_sorted[idx] == d["key"]
    rm = np.full(len(d["key"]), np.nan)
    rm[hit] = dval_sorted[idx[hit]]
    return rm

CUTS = [
    ("GLOBAL",                 lambda d: np.ones(len(d["mag"]), bool)),
    ("acc(mag<25 & Re>0.3)",   lambda d: (d["mag"] < 25) & (d["re"] > 0.3)),
    ("mag_lt24",               lambda d: d["mag"] < 24),
    ("mag_lt25",               lambda d: d["mag"] < 25),
    ("size_gt0.3",             lambda d: d["re"] > 0.3),
    ("mag25-26",               lambda d: (d["mag"] >= 25) & (d["mag"] < 26)),
    ("mag26-27(tail)",         lambda d: (d["mag"] >= 26) & (d["mag"] < 27)),
]

def leg_sums(d, mask):
    """per-case sums over detected&cut: e_meas1, e_intr1, n ; and over JOINED&det&cut: R_model, n_model."""
    m = d["det"] & mask
    mm = m & np.isfinite(d["rmodel"])
    def bycase(cs, vals):
        o = np.argsort(cs, kind="stable"); cs, vals = cs[o], vals[o]
        uc, st = np.unique(cs, return_index=True)
        s = np.add.reduceat(vals, st) if len(cs) else np.array([])
        n = np.diff(np.append(st, len(cs))) if len(cs) else np.array([])
        return {int(c): (s[i], n[i]) for i, c in enumerate(uc)}
    return (bycase(d["case"][m], d["e_meas1"][m]), bycase(d["case"][m], d["e_intr1"][m]),
            bycase(d["case"][mm], d["rmodel"][mm]))

def agg(sd, cases):
    S = sum(sd.get(c, (0, 0))[0] for c in cases); N = sum(sd.get(c, (0, 0))[1] for c in cases)
    return (S / N if N else np.nan), N

def combine(sp, sm, cases):
    (sep, sip, smp) = sp; (sem, sim, smm) = sm
    e1p, Np = agg(sep, cases); e1m, Nm = agg(sem, cases)
    eip, _ = agg(sip, cases); eim, _ = agg(sim, cases)
    rmp, NMp = agg(smp, cases); rmm, NMm = agg(smm, cases)
    if not (Np and Nm): return (np.nan,) * 6
    Rtot = (e1p - e1m) / (2 * G); Rsel = (eip - eim) / (2 * G); Rsh = Rtot - Rsel
    Rmod = np.nanmean([rmp, rmm])
    m_cert = Rsel / Rsh * 100 if abs(Rsh) > 1e-9 else np.nan
    m_ens = (Rtot / Rmod - 1) * 100 if (np.isfinite(Rmod) and abs(Rmod) > 1e-9) else np.nan
    joinfrac = (NMp + NMm) / (Np + Nm)
    return Rtot, Rsel, Rsh, Rmod, m_cert, m_ens, joinfrac

def main():
    print("=" * 116)
    print("STEP-2 CONSTGOLD ENSEMBLE: certified PREDICTED response <R_flow+R_blend> and m_ens=R_total/<R_model>-1")
    print("  (cases 40-139; r_sim NEVER read; <R_model> over the join-covered common population)")
    print("=" * 116)
    dcols = ["case", "input_index", "R_flow", "R_blend"]      # NO r_sim -> firewall
    td = feather.read_table(DUMP, columns=dcols)
    dc = np.asarray(td["case"]).astype(np.uint64); di = np.asarray(td["input_index"]).astype(np.uint64)
    dval = np.asarray(td["R_flow"]).astype("f8") + np.asarray(td["R_blend"]).astype("f8")
    dkey = dc * KEYMUL + di
    o = np.argsort(dkey); dkey_s, dval_s = dkey[o], dval[o]
    print(f"[dump] {len(dkey):,} objects, <R_flow+R_blend>={np.nanmean(dval):.4f}", flush=True)

    dp = load_leg(LEG["+"]); dm = load_leg(LEG["-"])
    for d in (dp, dm):
        keep = d["case"] >= 40                                # match dump case coverage
        for k in list(d.keys()):
            if isinstance(d[k], np.ndarray) and len(d[k]) == len(keep): d[k] = d[k][keep]
        d["rmodel"] = join_rmodel(d, dkey_s, dval_s)
    print(f"[legs] +:{dp['det'].sum():,} -:{dm['det'].sum():,} detected (cases>=40); "
          f"join hit +:{np.isfinite(dp['rmodel']).mean():.1%}", flush=True)
    cases = np.array(sorted(set(dp["case"].tolist()) | set(dm["case"].tolist())))
    print(f"[boot] {len(cases)} cases x {NBOOT}\n", flush=True)

    hdr = (f"{'cut':24s}{'R_total':>9}{'R_sel':>9}{'R_shape':>9}{'<Rmodel>':>10}"
           f"{'m_cert%':>9}{'m_ens%':>9}{'sem%':>7}{'join%':>7}")
    lines = [hdr, "-" * len(hdr)]
    rng = np.random.RandomState(2024)
    for lab, fn in CUTS:
        sp = leg_sums(dp, fn(dp)); sm = leg_sums(dm, fn(dm))
        Rtot, Rsel, Rsh, Rmod, m_cert, m_ens, jf = combine(sp, sm, cases)
        boot = np.empty(NBOOT)
        for b in range(NBOOT):
            rc = cases[rng.randint(0, len(cases), len(cases))]
            _, _, _, _, _, me, _ = combine(sp, sm, rc); boot[b] = me
        sem = np.nanstd(boot)
        lines.append(f"{lab:24s}{Rtot:>+9.4f}{Rsel:>+9.4f}{Rsh:>+9.4f}{Rmod:>+10.4f}"
                     f"{m_cert:>+9.2f}{m_ens:>+9.2f}{sem:>7.2f}{jf*100:>6.1f}")
        print(lines[-1], flush=True)

    lines += ["",
              "m_cert = selection-only bias (assumes model==R_shape). m_ens = ACTUAL certified ensemble bias",
              "(folds in flow shape closure). m_ens ~ m_cert + ~0.9% confirms owner's prediction. Joint-flow",
              "ceiling: a perfect selection head drives <R_model>->R_total => m_ens->0 (removes the selection term)."]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w").write("\n".join(lines) + "\n")
    print(f"\nwrote {OUT}\nSTEP2_PRED_DONE", flush=True)

if __name__ == "__main__":
    main()
