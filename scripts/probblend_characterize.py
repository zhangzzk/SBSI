"""Probabilistic blending — step 1: how much of R_blend comes from UNDETECTED neighbours?

The current framework feeds the emulator the TRUE neighbour field (every input galaxy as a
secondary, detected or not). In real data we only have the DETECTED catalogue; the undetected
neighbours must be forward-modelled. This script quantifies the split.

For a few cases of the main set (fs2_25876, which has a detection catalogue):
  1. run the blend emulator on the full true field  ->  per-pair response (primary,secondary,distance)
  2. tag every primary and secondary with its detection truth (detected flag @ shear 0.05)
  3. restrict to DETECTED primaries (the observed objects) and split each primary's summed
     R_blend into the part from DETECTED secondaries (observable) vs UNDETECTED secondaries
     (the forward-model's job).

Outputs the mean split overall, vs primary magnitude, vs secondary magnitude, vs separation,
and the implied multiplicative-bias contribution if undetected neighbours are simply dropped
(delta_m ~ <R_blend_undet>/R_total, R_total ~ 0.462).
"""
import argparse, sys
import numpy as np, pandas as pd, pyarrow.feather as pf

BASE = "/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876"
DET = f"{BASE}/detection_catalogue_train.feather"
BLEND_MODELS = "/home/z/Zekang.Zhang/blendemu/models"
COND = dict(pixel_size=0.2, zero_point=30.0, psf_fwhm=0.73, moffat_beta=2.224, pixel_rms=0.312)
TILE = "tile180.0_-0.5"
R_TOTAL = 0.462  # R_flow + R_blend denominator (from constgold), to convert response deficit -> m


def field_path(case):
    return f"{BASE}/case{case}_0.0/real0/catalogues/input/gals_info_{TILE}.feather"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--tag", default="lsst_r_extnbr_ho")
    ap.add_argument("--output", default="/home/z/Zekang.Zhang/SBSI/results/probblend_char.feather")
    args = ap.parse_args()

    sys.path.insert(0, "/home/z/Zekang.Zhang/blendemu")
    from blendemu.inference import BlendingPredictor
    pred = BlendingPredictor.load(BLEND_MODELS, tag=args.tag, conditions=COND, device="cpu")

    # detection truth: detected flag per (case, input_index)
    det = pf.read_table(DET, columns=["case", "input_index", "detected"]).to_pandas()
    det = det[det.case.isin(args.cases)].drop_duplicates(["case", "input_index"])
    det_map = {c: dict(zip(g.input_index.to_numpy(), g.detected.to_numpy().astype(bool)))
               for c, g in det.groupby("case")}
    print(f"detection truth loaded: {len(det):,} (case,idx) rows; "
          f"det rate {det.detected.mean():.3f}", flush=True)

    allpairs = []
    for c in args.cases:
        t = pf.read_table(field_path(c)).to_pandas()
        t = t.rename(columns={col: col.replace("_input", "") for col in t.columns})
        reg = pred.predict_response(t, t)          # primary=secondary=full field
        pk = [col for col in reg.columns if col.startswith("index") and col.endswith("_p")][0]
        sk = [col for col in reg.columns if col.startswith("index") and col.endswith("_s")][0]
        df = reg[[pk, sk, "r_input_p", "r_input_s", "distance", "response"]].copy()
        df.columns = ["ip", "is_", "rp", "rs", "dist", "resp"]
        dm = det_map[c]
        df["det_p"] = df.ip.map(dm).fillna(False).astype(bool)
        df["det_s"] = df.is_.map(dm).fillna(False).astype(bool)
        df["case"] = c
        allpairs.append(df)
        # quick per-case line
        dp = df[df.det_p]
        rb_full = dp.groupby("ip").resp.sum()
        rb_det = dp[dp.det_s].groupby("ip").resp.sum().reindex(rb_full.index).fillna(0.0)
        frac_undet = 1 - rb_det.sum() / rb_full.sum()
        print(f"case{c}: {len(df):,} pairs, {df.det_p.mean():.2%} pairs w/ det primary; "
              f"<R_blend|det pri>={rb_full.mean():.4f}  undetected-nbr frac={frac_undet:.1%}",
              flush=True)
    P = pd.concat(allpairs, ignore_index=True)
    P.to_feather(args.output)
    print(f"\nwrote {args.output}: {len(P):,} pairs", flush=True)

    # ---- global split over DETECTED primaries ----
    dp = P[P.det_p].copy()
    g = dp.groupby(["case", "ip"])
    rb_full = g.resp.sum()
    rb_det = dp[dp.det_s].groupby(["case", "ip"]).resp.sum().reindex(rb_full.index).fillna(0.0)
    rb_undet = rb_full - rb_det
    prim_r = g.rp.first()
    print("\n================ R_blend decomposition over DETECTED primaries ================")
    print(f"  N detected primaries with >=1 nbr : {len(rb_full):,}")
    print(f"  <R_blend full>       = {rb_full.mean():.4f}")
    print(f"  <R_blend det-nbr>    = {rb_det.mean():.4f}   ({rb_det.mean()/rb_full.mean():.1%})")
    print(f"  <R_blend UNDET-nbr>  = {rb_undet.mean():.4f}   ({rb_undet.mean()/rb_full.mean():.1%})")
    print(f"  => dropping undetected nbrs biases R_total by {rb_undet.mean():.4f}"
          f"  => delta_m ~ {rb_undet.mean()/R_TOTAL:+.2%}")

    # vs primary magnitude
    print("\n  by PRIMARY magnitude:")
    edges = np.array([18, 23, 24, 25, 26, 27, 28.1])
    pm = pd.DataFrame({"rp": prim_r, "full": rb_full, "undet": rb_undet})
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = pm[(pm.rp >= lo) & (pm.rp < hi)]
        if len(s) < 200:
            continue
        print(f"    r_p {lo:.0f}-{hi:.0f}: N={len(s):>7,}  <full>={s.full.mean():.4f}  "
              f"<undet>={s.undet.mean():.4f}  undet-frac={s.undet.sum()/s.full.sum():+.1%}")

    # contribution vs SECONDARY magnitude (where does the undetected signal live?)
    print("\n  R_blend contribution by SECONDARY magnitude (detected primaries only):")
    sedges = np.array([18, 24, 25, 26, 26.5, 27, 27.5, 28, 28.5, 29.1])
    for lo, hi in zip(sedges[:-1], sedges[1:]):
        s = dp[(dp.rs >= lo) & (dp.rs < hi)]
        if len(s) < 500:
            continue
        tot = dp.resp.sum()
        print(f"    r_s {lo:.1f}-{hi:.1f}: sum_resp={s.resp.sum():.1f} ({s.resp.sum()/tot:+.1%} of total)  "
              f"det-frac-of-these={s.det_s.mean():.1%}")
    print("PROBBLEND_CHAR_DONE", flush=True)


if __name__ == "__main__":
    main()
