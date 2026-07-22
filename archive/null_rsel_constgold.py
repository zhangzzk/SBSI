#!/usr/bin/env python
"""
NULL TEST (cont.98): is the constgold R_sel real or a shape-noise artifact?

For each rotation-null realization, re-orient every galaxy's INTRINSIC shape by a random angle
delta(galaxy) -- the SAME delta for the same galaxy in both the +0.02 and -0.02 legs (keyed by
(case,input_index)), so the detection pattern, the selected samples, and the CRN pairing are ALL
untouched; only the recorded shape values are randomized. This is exactly "change the shape-noise
seed while holding the selection fixed". Any R_sel it yields is the PURE shape-noise floor.

Report per cut, side by side:
  real R_sel        = (<e_intr1>_+ - <e_intr1>_-)/(2*0.02)              [the measured selection response]
  boot sem          = case-cluster bootstrap sem of real R_sel          [140 independent sim cases]
  null mean/std     = mean & std of R_sel over NROT random re-orientations [the empirical noise floor]
  z_boot, z_null    = |real R_sel| / (boot sem, null std)

If real R_sel >> null std AND boot sem ~ null std, R_sel is a resolved effect, not noise.
FIREWALL-safe: simulation truth only; trains nothing.
"""
import os, numpy as np, pyarrow.feather as feather

DDIR = "/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant"
LEG = {"+": f"{DDIR}/constant_shear_catalogue_0.02_train.feather",
       "-": f"{DDIR}/constant_shear_catalogue_-0.02_train.feather"}
G = 0.02
OUT = "/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/null_rsel_constgold.txt"
NROT, NBOOT = 200, 500

COLS = ["measured_e1", "axis_ratio_input_p", "position_angle_input_p",
        "r_input_p", "Re_input_p", "case", "input_index", "neighbored"]

def load_leg(path):
    t = feather.read_table(path, columns=COLS)
    d = {c: np.asarray(t[c]) for c in COLS}
    q = d["axis_ratio_input_p"].astype("f8")
    two_th = 2.0 * np.deg2rad(d["position_angle_input_p"].astype("f8"))
    emag = (1.0 - q) / (1.0 + q)
    det = np.isfinite(d["measured_e1"].astype("f8"))
    key = (d["case"].astype(np.uint64) * np.uint64(800003) + d["input_index"].astype(np.uint64))
    return dict(A=emag * np.cos(two_th), B=emag * np.sin(two_th),   # e_intr1 = A; rotated = A*cos2d - B*sin2d
                e_intr1=emag * np.cos(two_th), det=det, key=key,
                mag=d["r_input_p"].astype("f8"), re=d["Re_input_p"].astype("f8"),
                nbr=d["neighbored"].astype(bool), case=d["case"].astype("i8"))

CUTS = [
    ("GLOBAL",               lambda d: np.ones(len(d["mag"]), bool)),
    ("acc(mag<25 & Re>0.3)", lambda d: (d["mag"] < 25) & (d["re"] > 0.3)),
    ("mag_lt25",             lambda d: d["mag"] < 25),
    ("mag26-27(tail)",       lambda d: (d["mag"] >= 26) & (d["mag"] < 27)),
]

def hash01(key_u, a, b):
    """splitmix-style per-key uniform [0,1); deterministic in key => same for a galaxy in both legs."""
    with np.errstate(over="ignore"):
        x = key_u * np.uint64(a)
        x = x ^ (x >> np.uint64(30)); x = x * np.uint64(b)
        x = x ^ (x >> np.uint64(27)); x = x * np.uint64(0x2545F4914F6CDD1D)
        x = x ^ (x >> np.uint64(32))
    return (x >> np.uint64(40)).astype(np.float64) / float(1 << 24)

def per_case_sums_intr(d, mask):
    m = d["det"] & mask
    cs = d["case"][m]; ei = d["e_intr1"][m]
    o = np.argsort(cs, kind="stable"); cs, ei = cs[o], ei[o]
    uc, st = np.unique(cs, return_index=True)
    si = np.add.reduceat(ei, st) if len(cs) else np.array([])
    nn = np.diff(np.append(st, len(cs))) if len(cs) else np.array([])
    return {int(c): (si[i], nn[i]) for i, c in enumerate(uc)}

def rsel_from_sums(sp, sm, cases):
    SIp = sum(sp.get(c, (0, 0))[0] for c in cases); Np = sum(sp.get(c, (0, 0))[1] for c in cases)
    SIm = sum(sm.get(c, (0, 0))[0] for c in cases); Nm = sum(sm.get(c, (0, 0))[1] for c in cases)
    if Np == 0 or Nm == 0: return np.nan
    return (SIp / Np - SIm / Nm) / (2 * G)

def main():
    print("=" * 96)
    print("NULL TEST for constgold R_sel: real vs shape-noise floor (CRN-preserving random re-orientation)")
    print("=" * 96)
    dp = load_leg(LEG["+"]); print(f"[load] +0.02 {dp['det'].sum():,} det", flush=True)
    dm = load_leg(LEG["-"]); print(f"[load] -0.02 {dm['det'].sum():,} det", flush=True)
    cases = np.array(sorted(set(dp["case"].tolist()) | set(dm["case"].tolist())))
    rng = np.random.RandomState(777)

    # precompute per (cut,leg) the detected&cut subset arrays (A,B,key) for the null loop
    sub = {}
    real = {}; boot = {}
    for lab, fn in CUTS:
        for tag, d in (("+", dp), ("-", dm)):
            m = d["det"] & fn(d)
            sub[(lab, tag)] = (d["A"][m], d["B"][m], d["key"][m])
        sp = per_case_sums_intr(dp, fn(dp)); sm = per_case_sums_intr(dm, fn(dm))
        real[lab] = rsel_from_sums(sp, sm, cases)
        bb = np.array([rsel_from_sums(sp, sm, cases[rng.randint(0, len(cases), len(cases))]) for _ in range(NBOOT)])
        boot[lab] = np.nanstd(bb)

    # rotation-null: NROT re-orientations; each gives one R_sel per cut
    null = {lab: np.empty(NROT) for lab, _ in CUTS}
    for r in range(NROT):
        a = np.uint64(rng.randint(1, 2**63) | 1); b = np.uint64(rng.randint(1, 2**63) | 1)
        for lab, _ in CUTS:
            vals = {}
            for tag in ("+", "-"):
                A, B, key = sub[(lab, tag)]
                u = hash01(key, a, b) * (2 * np.pi)           # 2*delta in [0,2pi)
                erot = A * np.cos(u) - B * np.sin(u)          # rotated e_intr1
                vals[tag] = erot.mean()
            null[lab][r] = (vals["+"] - vals["-"]) / (2 * G)
        if (r + 1) % 50 == 0: print(f"  null {r+1}/{NROT}", flush=True)

    hdr = f"{'cut':24s}{'real R_sel':>12}{'boot sem':>11}{'null mean':>11}{'null std':>11}{'z_boot':>8}{'z_null':>8}"
    lines = [hdr, "-" * len(hdr)]
    for lab, _ in CUTS:
        rs = real[lab]; bs = boot[lab]; nm = np.mean(null[lab]); ns = np.std(null[lab])
        zb = abs(rs) / bs if bs > 0 else np.nan
        zn = abs(rs) / ns if ns > 0 else np.nan
        lines.append(f"{lab:24s}{rs:>+12.5f}{bs:>11.5f}{nm:>+11.5f}{ns:>11.5f}{zb:>8.1f}{zn:>8.1f}")
        print(lines[-1], flush=True)
    lines += ["",
              "READING: null mean ~ 0 and null std = the shape-noise floor on R_sel at the REAL selection.",
              "boot sem ~ null std validates the case bootstrap. z_null = |real R_sel|/floor: >>1 => R_sel is",
              "a resolved effect, not a shape-noise artifact from the one realization."]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w").write("\n".join(lines) + "\n")
    print(f"\nwrote {OUT}\nNULL_RSEL_DONE", flush=True)

if __name__ == "__main__":
    main()
