#!/usr/bin/env python
"""
STEP 1 (owner-directed, cont.97): evaluate the response COMPONENTS on the CERTIFIED sim
(constant-gold +/-0.02), UNMATCHED, from the average shape of each leg's own selected sample.

This is the corrected version of the cont.95/96 decomposition, which mistakenly ran on the
half-shear g=0.05 leg. Here we use the SAME sim the certified pipeline is validated on, read
UNMATCHED (each leg selects independently), so the ensemble response CONTAINS selection.

constgold shear is uniform along g1 (applied_g=(+/-0.02,0)) => projection onto the shear is
just e1. Antithetic estimator between the +0.02 and -0.02 legs:

    R_total = (<e_meas1>_{+,C} - <e_meas1>_{-,C}) / (2*0.02)    # shape + detection-selection
    R_sel   = (<e_intr1>_{+,C} - <e_intr1>_{-,C}) / (2*0.02)    # intrinsic has no shape resp => pure selection
    R_shape = R_total - R_sel                                    # cross-check vs matched r_sim=0.4534
    m_cert  = R_sel / R_shape                                    # certified bias from the MISSING selection resp

Intrinsic e (blendemu shape.py angle2e, sky_cos_sin): e1 = (1-q)/(1+q)*cos(2*theta),
theta = position_angle_input_p [deg] * pi/180.

Cuts are on TRUE properties (r_input_p, Re_input_p): a true cut is shear-INDEPENDENT, so any
R_sel it leaves is PURE detection-selection (the significant component). Gentle acceptance cut =
r_input_p<25 & Re_input_p>0.3. FIREWALL-safe: only measures truth on constgold + compares; trains nothing.
"""
import os, sys, numpy as np, pyarrow.feather as feather

DDIR = "/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant"
LEG = {"+": f"{DDIR}/constant_shear_catalogue_0.02_train.feather",
       "-": f"{DDIR}/constant_shear_catalogue_-0.02_train.feather"}
G = 0.02
OUT = "/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/decomp_constgold_ensemble.txt"
RSIM_MATCHED = 0.4534   # certified matched-pair truth (fig2 dump <r_sim>), GLOBAL cross-check for R_shape
NBOOT = 1000

COLS = ["measured_e1", "axis_ratio_input_p", "position_angle_input_p",
        "r_input_p", "Re_input_p", "case", "neighbored", "S/N"]

def load_leg(path):
    t = feather.read_table(path, columns=COLS)
    d = {c: np.asarray(t[c]) for c in COLS}
    q = d["axis_ratio_input_p"].astype("f8")
    th = np.deg2rad(d["position_angle_input_p"].astype("f8"))
    d["e_intr1"] = (1.0 - q) / (1.0 + q) * np.cos(2.0 * th)   # sky_cos_sin intrinsic e1
    d["e_meas1"] = d["measured_e1"].astype("f8")
    d["det"] = np.isfinite(d["e_meas1"])                       # detected & measured
    d["mag"] = d["r_input_p"].astype("f8")
    d["re"] = d["Re_input_p"].astype("f8")
    d["case"] = d["case"].astype("i8")
    d["nbr"] = d["neighbored"].astype(bool)
    return d

CUTS = [
    ("GLOBAL",                 lambda d: np.ones(len(d["mag"]), bool)),
    ("acc(mag<25 & Re>0.3)",   lambda d: (d["mag"] < 25) & (d["re"] > 0.3)),
    ("mag_lt24",               lambda d: d["mag"] < 24),
    ("mag_lt25",               lambda d: d["mag"] < 25),
    ("size_gt0.3",             lambda d: d["re"] > 0.3),
    ("mag24-25",               lambda d: (d["mag"] >= 24) & (d["mag"] < 25)),
    ("mag25-26",               lambda d: (d["mag"] >= 25) & (d["mag"] < 26)),
    ("mag26-27(tail)",         lambda d: (d["mag"] >= 26) & (d["mag"] < 27)),
    ("size0.2-0.3(tail)",      lambda d: (d["re"] >= 0.2) & (d["re"] < 0.3)),
    ("isolated",               lambda d: ~d["nbr"]),
    ("blended",                lambda d: d["nbr"]),
]

def per_case_sums(d, cutmask):
    """Return dict case -> [sum(e_meas1), sum(e_intr1), n] over detected & cut."""
    m = d["det"] & cutmask
    cs = d["case"][m]; em = d["e_meas1"][m]; ei = d["e_intr1"][m]
    order = np.argsort(cs, kind="stable")
    cs, em, ei = cs[order], em[order], ei[order]
    uc, start = np.unique(cs, return_index=True)
    se = np.add.reduceat(em, start) if len(cs) else np.array([])
    si = np.add.reduceat(ei, start) if len(cs) else np.array([])
    nn = np.diff(np.append(start, len(cs))) if len(cs) else np.array([])
    return {int(c): (se[i], si[i], nn[i]) for i, c in enumerate(uc)}

def combine(sp, sm, cases):
    """Point estimate R_total,R_sel,R_shape from + and - per-case sum dicts."""
    SEp = sum(sp.get(c, (0, 0, 0))[0] for c in cases); SIp = sum(sp.get(c, (0, 0, 0))[1] for c in cases); Np = sum(sp.get(c, (0, 0, 0))[2] for c in cases)
    SEm = sum(sm.get(c, (0, 0, 0))[0] for c in cases); SIm = sum(sm.get(c, (0, 0, 0))[1] for c in cases); Nm = sum(sm.get(c, (0, 0, 0))[2] for c in cases)
    if Np == 0 or Nm == 0:
        return np.nan, np.nan, np.nan, 0
    e1p, e1m = SEp / Np, SEm / Nm
    eip, eim = SIp / Np, SIm / Nm
    Rtot = (e1p - e1m) / (2 * G)
    Rsel = (eip - eim) / (2 * G)
    return Rtot, Rsel, Rtot - Rsel, int(Np + Nm)

def main():
    print("=" * 104)
    print("STEP-1 CONSTGOLD ENSEMBLE DECOMPOSITION (unmatched +/-0.02; R_total=meas, R_sel=intrinsic=selection)")
    print(f"  R_shape should cross-check against matched r_sim = {RSIM_MATCHED:.4f} on GLOBAL.")
    print("=" * 104)
    dp = load_leg(LEG["+"]); print(f"[load] +0.02: {len(dp['mag']):,} rows, {dp['det'].sum():,} detected", flush=True)
    dm = load_leg(LEG["-"]); print(f"[load] -0.02: {len(dm['mag']):,} rows, {dm['det'].sum():,} detected", flush=True)
    cases = sorted(set(dp["case"].tolist()) | set(dm["case"].tolist()))
    cases = np.array(cases); print(f"[boot] {len(cases)} cases x {NBOOT}\n", flush=True)

    hdr = f"{'cut':24s}{'n':>12}{'R_total':>10}{'R_sel':>10}{'R_shape':>10}{'m_cert%':>10}{'sem%':>8}{'z':>7}"
    lines = [hdr, "-" * len(hdr)]
    rng = np.random.RandomState(12345)
    for lab, fn in CUTS:
        sp = per_case_sums(dp, fn(dp)); sm = per_case_sums(dm, fn(dm))
        Rtot, Rsel, Rsh, n = combine(sp, sm, cases)
        # case bootstrap for m_cert = Rsel/Rshape
        boot = np.empty(NBOOT)
        for b in range(NBOOT):
            rc = cases[rng.randint(0, len(cases), len(cases))]
            _, rs, rh, _ = combine(sp, sm, rc)
            boot[b] = (rs / rh * 100) if (rh and np.isfinite(rh) and abs(rh) > 1e-9) else np.nan
        mcert = (Rsel / Rsh * 100) if (Rsh and abs(Rsh) > 1e-9) else np.nan
        sem = np.nanstd(boot); z = abs(mcert) / sem if sem > 0 else np.nan
        lines.append(f"{lab:24s}{n:>12,}{Rtot:>+10.4f}{Rsel:>+10.4f}{Rsh:>+10.4f}{mcert:>+10.2f}{sem:>8.2f}{z:>7.1f}")
        print(lines[-1], flush=True)

    lines += ["", "READING: R_shape ~ 0.45 on GLOBAL validates the setup (matches matched r_sim). R_sel = the",
              "detection-selection response the certified (shape-only) pipeline MISSES on the unmatched ensemble.",
              "m_cert = R_sel/R_shape = the certified pipeline's bias on the selection-included ensemble, per cut.",
              "On the GENTLE acceptance cut (mag<25 & Re>0.3) this is the number that decides the unified-flow question."]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nwrote {OUT}\nDECOMP_CONSTGOLD_DONE", flush=True)

if __name__ == "__main__":
    main()
