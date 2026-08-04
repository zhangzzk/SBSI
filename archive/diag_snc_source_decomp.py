"""WHICH MEASUREMENT SOURCE CREATES THE -3.06% SMALL-SIZE LABEL GAP?

WHAT 03u ESTABLISHED, AND WHAT IT DID **NOT**. The pin's label (ruler `R_snc`) sits -3.06 +- 0.84%
below the scoring truth (dump `r_sim_self`) at small size. That comparison was an INNER JOIN of the
ruler against the dump, so both estimators were already evaluated on the SAME both-detected objects.
**Selection is therefore ruled out as the explanation for that number** -- the two disagree on
identical galaxies, which leaves only the catalogues that supply the shapes.

Each estimator is a difference of two ngmix shape measurements, projected on ghat and divided by g:

    R_snc        = [ eS(crowd g0.05 val_full)  -  e0(g0_lookup, raw secondaries Shapes) ] . ghat / g
    r_sim_self   = [ eS(ngmix g0.05 val)       -  e0(det_meas ngmix g0.0 TRAIN leg)     ] . ghat / g
                       ^-- sheared leg differs        ^-- unsheared leg differs

So the gap decomposes into exactly two additive pieces, and this script measures both on matched
objects:

    d_eS = [eS_crowd - eS_ngmix] . ghat / g        (disagreement in the SHEARED measurement)
    d_e0 = [e0_lookup - e0_train] . ghat / g       (disagreement in the UNSHEARED measurement)
    R_snc - r_sim_self  =  d_eS - d_e0             (exact, up to rows where any source is missing)

The sum is checked against the directly measured gap, so a decomposition that does not add up is
caught rather than reported.

WHY THIS DECIDES SOMETHING. If the whole gap is `d_e0`, the two g=0 shape sources disagree for small
galaxies and one of them is wrong -- the raw secondaries Shapes catalogue measures at DETECTED
POSITIONS, which for a marginal small object is not the same estimator as the matched train-leg
measurement. If instead it is `d_eS`, then two catalogues built from the same sheared sim disagree,
which would point at the catalogue build rather than at the SNC construction.

**This still does not say which one is CORRECT.** It says where the disagreement enters, which is the
prerequisite for asking that question of the right pair of files.

FIREWALL. Half-shear only. constgold is never read, nothing is fit, and no `m` is computed.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.feather as pf
import pyarrow.ipc as ipc

NG = ["measured_ngmix_g1", "measured_ngmix_g2"]


def read_cols(path, cols, max_case):
    parts = []
    with ipc.open_file(path) as r:
        have = set(r.schema.names)
        miss = [c for c in cols if c not in have]
        if miss:
            raise SystemExit(f"REFUSING: {path} lacks {miss}")
        for bi in range(r.num_record_batches):
            b = pa.Table.from_batches([r.get_batch(bi)]).select(cols).to_pandas()
            b = b[b["case"] <= max_case]
            if len(b):
                parts.append(b)
    df = pd.concat(parts, ignore_index=True).drop_duplicates(["case", "input_index"])
    return df.reset_index(drop=True)


def region_mean(x, keep, case, rng, nboot=300):
    ok = keep & np.isfinite(x)
    if ok.sum() < 100:
        return np.nan, np.nan, int(ok.sum())
    cen = float(np.mean(x[ok]))
    uc = np.unique(case[ok])
    bs = []
    for _ in range(nboot):
        pick = rng.choice(uc, size=len(uc), replace=True)
        m = np.isin(case, pick) & ok
        if m.sum() > 100:
            bs.append(float(np.mean(x[m])))
    return cen, float(np.std(bs)), int(ok.sum())


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--crowd-gs", required=True, help="crowd g0.05 val_full (R_snc's sheared leg)")
    ap.add_argument("--ngmix-gs", required=True, help="ngmix g0.05 val (r_sim_self's sheared leg)")
    ap.add_argument("--g0-lookup", required=True, help="g0_lookup (R_snc's unsheared leg)")
    ap.add_argument("--ngmix-g0", required=True, help="ngmix g0.0 train leg (r_sim_self's unsheared)")
    ap.add_argument("--max-case", type=int, default=39)
    ap.add_argument("--nominal-g", type=float, default=0.05)
    ap.add_argument("--small-re", type=float, default=0.386)
    ap.add_argument("--primary-re-min", type=float, default=None,
                    help="apply the deliverable-domain lower size cut, so the numbers are "
                         "comparable to WORKLOG 2026-08-03u (which ran in-domain)")
    ap.add_argument("--primary-mag-max", type=float, default=None)
    args = ap.parse_args()

    gsc = read_cols(args.crowd_gs, ["case", "input_index", "Re_input_p", "r_input_p",
                                    "gamma1_input_p", "gamma2_input_p"] + NG, args.max_case)
    print(f"crowd gS: {len(gsc):,}")
    gsn = read_cols(args.ngmix_gs, ["case", "input_index"] + NG, args.max_case)
    print(f"ngmix gS: {len(gsn):,}")
    g00 = read_cols(args.ngmix_g0, ["case", "input_index"] + NG, args.max_case)
    print(f"ngmix g0: {len(g00):,}")
    lk = pf.read_table(args.g0_lookup,
                       columns=["case", "input_index", "ngmix0_g1", "ngmix0_g2"]).to_pandas()
    lk = lk[lk["case"] <= args.max_case].drop_duplicates(["case", "input_index"])
    print(f"g0 lookup: {len(lk):,}")

    df = (gsc.merge(gsn, on=["case", "input_index"], suffixes=("_c", "_n"))
              .merge(g00, on=["case", "input_index"])
              .merge(lk, on=["case", "input_index"]))
    print(f"\nmatched on all four sources: {len(df):,} rows")

    if args.primary_re_min is not None or args.primary_mag_max is not None:
        n0 = len(df)
        if args.primary_re_min is not None:
            df = df[df["Re_input_p"] > args.primary_re_min]
        if args.primary_mag_max is not None:
            df = df[df["r_input_p"] < args.primary_mag_max]
        df = df.reset_index(drop=True)
        print(f"domain cut (Re > {args.primary_re_min}, mag < {args.primary_mag_max}): "
              f"{n0:,} -> {len(df):,}")

    g1 = df["gamma1_input_p"].to_numpy(float)
    g2 = df["gamma2_input_p"].to_numpy(float)
    gp = np.hypot(g1, g2)
    keep = gp > 1e-6
    df, g1, g2, gp = df[keep].reset_index(drop=True), g1[keep], g2[keep], gp[keep]
    gh1, gh2 = g1 / gp, g2 / gp
    g = args.nominal_g

    def proj(a, b):
        return a.to_numpy(float) * gh1 + b.to_numpy(float) * gh2

    pS_c = proj(df["measured_ngmix_g1_c"], df["measured_ngmix_g2_c"])
    pS_n = proj(df["measured_ngmix_g1_n"], df["measured_ngmix_g2_n"])
    p0_t = proj(df["measured_ngmix_g1"], df["measured_ngmix_g2"])
    p0_l = proj(df["ngmix0_g1"], df["ngmix0_g2"])

    d_eS = (pS_c - pS_n) / g
    d_e0 = (p0_l - p0_t) / g
    gap = d_eS - d_e0                      # == R_snc - r_sim_self, in response units

    re = df["Re_input_p"].to_numpy(float)
    case = df["case"].to_numpy(np.int64)
    rng = np.random.default_rng(0)
    R_snc = (pS_c - p0_l) / g
    R_sim = (pS_n - p0_t) / g
    # ONE finite mask for every quantity. Letting each array drop its own non-finite rows makes the
    # four means describe four slightly different row sets, and the identity
    # R_snc - r_sim_self == d_eS - d_e0 then fails by a fraction of a percent for no physical
    # reason -- which is exactly what the integrity check caught on the first in-domain run.
    good = (np.isfinite(pS_c) & np.isfinite(pS_n) & np.isfinite(p0_t) & np.isfinite(p0_l))
    print(f"rows finite in ALL four sources: {int(good.sum()):,} / {len(good):,}")

    print("\n=== WHERE THE LABEL GAP COMES FROM (response units; bootstrapped over CASES) ===")
    print("    d_eS = sheared-leg disagreement, d_e0 = unsheared-leg disagreement,\n"
          "    and R_snc - r_sim_self must equal d_eS - d_e0.\n")
    for lab, kp0 in (("ALL", np.ones(len(df), bool)),
                     (f"small size Re <= {args.small_re}", re <= args.small_re)):
        kp = kp0 & good
        rs, _, n = region_mean(R_snc, kp, case, rng)
        rr, _, _ = region_mean(R_sim, kp, case, rng)
        a, sa, _ = region_mean(d_eS, kp, case, rng)
        b, sb, _ = region_mean(d_e0, kp, case, rng)
        tot, st, _ = region_mean(gap, kp, case, rng)
        print(f"  {lab}   N={n:,}")
        print(f"      <R_snc>={rs:+.5f}   <r_sim_self>={rr:+.5f}   gap={rs-rr:+.5f} "
              f"({100*(rs/rr-1):+.2f}%)")
        print(f"      d_eS (sheared sources disagree)   = {a:+.5f} +- {sa:.5f}")
        print(f"      d_e0 (unsheared sources disagree) = {b:+.5f} +- {sb:.5f}")
        print(f"      d_eS - d_e0                       = {tot:+.5f} +- {st:.5f}   "
              f"(must match gap {rs-rr:+.5f})")
        if abs(tot - (rs - rr)) > 1e-4:
            print("      *** DECOMPOSITION DOES NOT ADD UP -- do not use these numbers ***")
        dom = "UNSHEARED (g0 source)" if abs(b) > abs(a) else "SHEARED (gS source)"
        print(f"      -> dominated by the {dom}\n")

    # ---- are the two g=0 sources even the SAME measurement of the same galaxy? ----------------
    # SNC subtracts a galaxy's OWN unsheared shape. If the two sources were the same measurement they
    # would correlate ~1 and R_snc would equal r_sim_self exactly (d_eS is identically zero, so the
    # g0 source is the ONLY thing that can decorrelate them). 03q measured corr(R_snc, r_sim_self)
    # = 0.525, which this tests the origin of directly.
    print("=== ARE THE TWO g=0 SOURCES THE SAME MEASUREMENT? (per-object, projected on ghat) ===")
    for lab, kp0 in (("ALL", np.ones(len(df), bool)),
                     (f"small size Re <= {args.small_re}", re <= args.small_re)):
        ok = kp0 & good
        c = float(np.corrcoef(p0_l[ok], p0_t[ok])[0, 1])
        dd = p0_l[ok] - p0_t[ok]
        print(f"  {lab:<26} N={int(ok.sum()):>9,}  corr={c:.5f}  "
              f"mean(diff)={dd.mean():+.3e}  std(diff)={dd.std():.4f}  "
              f"std(lookup)={p0_l[ok].std():.4f}  std(train)={p0_t[ok].std():.4f}")
    print("  corr ~1 => the same measurement plus tiny numerical differences; corr well below 1 =>\n"
          "  they are NOT the same realisation, and at least one is not the SNC partner it must be.\n")

    print("READ IT AS: this locates WHERE the two estimators diverge, not which is correct. A gap\n"
          "carried by d_e0 means the two g=0 shape sources disagree on small galaxies; carried by\n"
          "d_eS it means two catalogues of the same sheared sim disagree.")
    print("\nSNC_DECOMP_DONE")


if __name__ == "__main__":
    main()
