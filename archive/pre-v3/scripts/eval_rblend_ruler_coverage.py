"""How much of the fiducial emulator's summed `R_blend` does the per-pair RULER actually certify?

WHY THIS EXISTS
---------------
`scripts/eval_rblend_gap_measured.py` scores the emulator against half-shear TRUTH per pair, and
`scripts/eval_rblend_gap_summed.py` sums those pairs per primary. But the ruler can only see pairs the
CATALOGUE annotates: the `det_meas_ngmix_ap7_*` legs annotate neighbours out to **7.0"** (verified:
`max(distance) = 7.0000`), whereas the deployed `R_blend` that enters `m` is built by
`scripts/build_blend_lookup.py` -> `BlendingPredictor.predict_response`, whose stored regression
aperture is **k=20, r_max=10"** (read back from the checkpoint metadata, printed by this script).

So the ruler certifies the 0-7" part of `R_blend` and says NOTHING about 7-10". That matters because
WORKLOG 2026-07-30 RESULT 7 found the emulator's per-pair response does not fall off fast enough for
the neighbour sum to converge -- `<R_blend>` keeps growing with `r_max`. If most of the deployed
`R_blend` came from the annulus the ruler cannot see, an emulator verdict drawn from the ruler would
not transfer.

WHAT THIS MEASURES
------------------
The emulator's OWN summed prediction, on the same input fields, at several apertures -- all of them
INSIDE the trained `[0,10]` distance cut, so none of it is extrapolation:

    coverage(r) = <S_model(r_max = r)> / <S_model(r_max = 10)>       (deliverable domain)

`coverage(7)` is the fraction of the deployed `R_blend` the ruler's verdict actually covers. This is a
statement about the MODEL, not about truth -- that is exactly what is wanted, because the question is
"what fraction of the number the model contributes to `m` has been tested".

`k` is held at 60 throughout (NOT the certified 20) so the neighbour cap can never bind and the only
thing varying is the radius. WORKLOG RESULT 7 measured k=20 -> 60 at r_max=10 as +0.0002, so this
choice is worth ~0.1% and it removes a confound.

FIREWALL: reads HALF-SHEAR input fields (`lsst_sims_fs2_25876/case{c}_{sign}`) only -- the same sim
family the ruler is measured on. constgold is never opened. Nothing is trained, fitted or tuned.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
import pyarrow.feather as pf

HSBASE = "/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876"
BLEND_MODELS = "/home/z/Zekang.Zhang/blendemu/models"
COND = dict(pixel_size=0.2, zero_point=30.0, psf_fwhm=0.73, moffat_beta=2.224, pixel_rms=0.312)
TILE = "tile180.0_-0.5"


def summed_for_field(pred, t, base_cuts, k, rmax):
    """Per-primary summed emulator response at aperture (k, rmax), aligned to `t`'s row order."""
    pred._select = lambda task, _c=base_cuts, _k=k, _r=rmax: (_c, _r, _k)  # radius only
    reg = pred.predict_response(t, t)
    pk = [c for c in reg.columns if c.startswith("index")][0]
    rb = reg.groupby(pk)["response"].sum()
    key = pd.Index(t["index"].to_numpy())
    out = np.zeros(len(t), float)
    pos = key.get_indexer(rb.index.to_numpy())
    ok = pos >= 0
    out[pos[ok]] = rb.to_numpy(float)[ok]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--base", default=HSBASE)
    ap.add_argument("--sign", default="0.05", help="half-shear leg whose input field to use")
    ap.add_argument("--tags", nargs="+",
                    default=["lsst_r_extnbr_indom_tuned", "lsst_r_extnbr_ho"])
    ap.add_argument("--radii", type=float, nargs="+", default=[1.0, 3.0, 5.0, 7.0, 10.0])
    ap.add_argument("--k", type=int, default=60, help="neighbour cap, held non-binding")
    ap.add_argument("--true-mag-max", type=float, default=26.0)
    ap.add_argument("--true-re-min", type=float, default=0.3)
    ap.add_argument("--ruler-npz", default=None,
                    help="optional per-pair ruler dump; if given, coverage is ALSO split by the "
                         "primary's MEASURED mag (joined on case+input_index)")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    sys.path.insert(0, "/home/z/Zekang.Zhang/blendemu")
    from blendemu.inference import BlendingPredictor

    # optional measured-mag map, keyed by (case, input_index), from the ruler dump
    magmap = None
    if args.ruler_npz:
        try:
            d = np.load(args.ruler_npz, allow_pickle=True)
            kk = (d["case"].astype(np.int64) << 32) + d["input_index"].astype(np.int64)
            mm = d["mag0"].astype(float)
            sel = np.isin(d["case"].astype(np.int64), np.array(args.cases, np.int64))
            kk, mm = kk[sel], mm[sel]
            _, first = np.unique(kk, return_index=True)
            magmap = pd.Series(mm[first], index=kk[first])
            print(f"measured-mag map from {args.ruler_npz}: {len(magmap):,} primaries "
                  f"over cases {sorted(set(d['case'][sel].tolist()))[:8]}...", flush=True)
            del d
        except Exception as e:                                   # never let this kill the run
            print(f"WARNING: could not build the measured-mag map ({type(e).__name__}: {e}); "
                  f"continuing with TRUE mag only", flush=True)
            magmap = None

    rows = []
    for tag in args.tags:
        pred = BlendingPredictor.load(BLEND_MODELS, tag=tag, conditions=COND, device="cpu")
        base_cuts, base_rmax, base_k = pred._select("regression")
        print(f"\n=== {tag}: certified regression aperture k={base_k}, r_max={base_rmax}\"  "
              f"cuts={base_cuts}", flush=True)
        if base_rmax is not None and max(args.radii) > base_rmax:
            raise SystemExit(f"radii exceed the trained r_max={base_rmax}; that would be "
                             f"extrapolation and this script refuses to report it")
        for case in args.cases:
            fp = f"{args.base}/case{case}_{args.sign}/real0/catalogues/input/gals_info_{TILE}.feather"
            if not os.path.exists(fp):
                print(f"  case{case}: MISSING {fp}", flush=True)
                continue
            t = pf.read_table(fp).to_pandas()
            t = t.rename(columns={c: c.replace("_input", "") for c in t.columns})
            tmag = t["r"].to_numpy(float)
            dom = (tmag < args.true_mag_max) & (t["Re"].to_numpy(float) > args.true_re_min)
            mag_meas = None
            if magmap is not None:
                key = (np.int64(case) << 32) + t["index"].to_numpy(np.int64)
                mag_meas = magmap.reindex(key).to_numpy(float)
            for r in args.radii:
                S = summed_for_field(pred, t, base_cuts, args.k, r)
                rec = dict(tag=tag, case=case, r_max=r, n=len(S), n_dom=int(dom.sum()),
                           mean_all=float(S.mean()), mean_dom=float(S[dom].mean()))
                for lo, hi in ((18, 24), (24, 25), (25, 26)):
                    m = dom & (tmag >= lo) & (tmag < hi)
                    rec[f"dom_tmag{lo}"] = float(S[m].mean()) if m.sum() else np.nan
                if mag_meas is not None:
                    fin = dom & np.isfinite(mag_meas)
                    rec["dom_meas_bright"] = float(S[fin & (mag_meas < 26)].mean()) \
                        if (fin & (mag_meas < 26)).sum() else np.nan
                    m = fin & (mag_meas >= 26)
                    rec["dom_meas_shell"] = float(S[m].mean()) if m.sum() else np.nan
                    rec["n_meas_shell"] = int(m.sum())
                rows.append(rec)
                print(f"  case{case} r_max={r:>5.1f}\": N={len(S):,} in-domain={dom.sum():,}  "
                      f"<S>_all={S.mean():.4f}  <S>_dom={S[dom].mean():.4f}", flush=True)

    d = pd.DataFrame(rows)
    if d.empty:
        raise SystemExit("no fields processed")

    cols = [c for c in d.columns if c.startswith(("mean_", "dom_"))]
    for tag in d["tag"].unique():
        g = d[d["tag"] == tag].groupby("r_max")[cols].mean()
        ref = g.loc[max(args.radii)]
        print(f"\n{'='*100}\nCOVERAGE of the ruler's 7\" annotation -- emulator {tag}\n"
              f"averaged over {d[d['tag']==tag]['case'].nunique()} half-shear input fields; "
              f"k={args.k} (non-binding); denominator = the certified r_max=10\"\n{'='*100}")
        hdr = f"{'r_max':>7} {'<S>_all':>9} {'<S>_dom':>9} {'cover_dom':>10}"
        sub = [c for c in cols if c.startswith("dom_")]
        for c in sub:
            hdr += f" {c[-12:]:>13}"
        print(hdr)
        for r in sorted(g.index):
            line = (f"{r:>7.1f} {g.loc[r,'mean_all']:>9.4f} {g.loc[r,'mean_dom']:>9.4f} "
                    f"{g.loc[r,'mean_dom']/ref['mean_dom']:>10.3f}")
            for c in sub:
                v = g.loc[r, c] / ref[c] if np.isfinite(ref[c]) and ref[c] else np.nan
                line += f" {v:>13.3f}"
            print(line)
        c7 = float(g.loc[7.0, "mean_dom"] / ref["mean_dom"]) if 7.0 in g.index else np.nan
        print(f"\n  --> the ruler (7\" annotation) certifies {c7:.1%} of the deployed "
              f"<R_blend>; {1-c7:.1%} sits in the 7-10\" annulus and is UNTESTED by it.")

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        d.to_csv(args.output, index=False)
        print(f"\nsaved {args.output}")
    print("RULER_COVERAGE_DONE", flush=True)


if __name__ == "__main__":
    main()
