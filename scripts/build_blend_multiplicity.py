"""Per-(case, input_index) BlendEMU multiplicity/shape lookup for the constant-shear gold set.

Mirrors build_blend_lookup.py EXACTLY (same emulator call, same per-case input field) but instead of
only the summed R_blend it also carries the shape of the per-pair response distribution:
  R_blend  = sum of per-pair responses           (== build_blend_lookup, for cross-check)
  n_pairs  = number of neighbour pairs summed     (multiplicity)
  rb_max   = largest single-pair response         (one-dominant-neighbour vs many-small discriminator)
  rb_top2  = sum of the two largest pair responses

Purpose: the constgold residual sits entirely in the moderate-R_blend regime (q3, +9%). Two causes:
  (a) R_blend under-predicts because the per-pair SUM under-counts many-moderate-neighbour objects
      (multiplicity / super-additivity)  -> deficit grows with n_pairs at fixed R_blend.
  (b) R_flow (crowd-flux self-response) over-suppresses at moderate crowding -> deficit flat in n_pairs.
This lookup lets validate_constant bin the deficit by n_pairs (with proper per-subset R_flow) to decide.
"""
import argparse, os, sys
import numpy as np
import pandas as pd
import pyarrow.feather as pf
from sbs_shear.paths import CONST_SIM_DIR as CBASE

BLEND_MODELS = "/home/z/Zekang.Zhang/blendemu/models"
COND = dict(pixel_size=0.2, zero_point=30.0, psf_fwhm=0.73, moffat_beta=2.224, pixel_rms=0.312)
TILE = "tile180.0_-0.5"


def input_feather(case, sign="0.02", base=CBASE):
    return f"{base}/case{case}_{sign}/real0/catalogues/input/gals_info_{TILE}.feather"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, nargs="+", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--sign", default="0.02")
    ap.add_argument("--base", default=CBASE)
    ap.add_argument("--tag", default="lsst_r_extnbr_ho", help="emulator tag (matches blend_lookup_extnbrho)")
    args = ap.parse_args()

    sys.path.insert(0, "/home/z/Zekang.Zhang/blendemu")
    from blendemu.inference import BlendingPredictor
    pred = BlendingPredictor.load(BLEND_MODELS, tag=args.tag, conditions=COND, device="cpu")

    parts = []
    for c in args.cases:
        fp = input_feather(c, args.sign, args.base)
        if not os.path.exists(fp):
            print(f"case{c}: MISSING {fp}", flush=True); continue
        t = pf.read_table(fp).to_pandas()
        t = t.rename(columns={col: col.replace("_input", "") for col in t.columns})
        reg = pred.predict_response(t, t)
        pk = [col for col in reg.columns if col.startswith("index")][0]
        reg["absr"] = reg["response"].abs()
        # vectorized: sum + count via agg; rb_max/rb_top2 via a single |response|-descending sort
        agg = reg.groupby(pk)["response"].agg(R_blend="sum", n_pairs="count")
        rs = reg.sort_values([pk, "absr"], ascending=[True, False], kind="mergesort")
        rank = rs.groupby(pk).cumcount()
        rb_max = rs[rank == 0].set_index(pk)["response"]                       # largest |response| (signed)
        rb_top2 = rs[rank < 2].groupby(pk)["response"].sum()                   # sum of two largest-|.|
        agg["rb_max"] = rb_max
        agg["rb_top2"] = rb_top2
        parts.append(pd.DataFrame({"case": c,
                                   "input_index": agg.index.to_numpy().astype(np.int64),
                                   "R_blend": agg["R_blend"].to_numpy(float),
                                   "n_pairs": agg["n_pairs"].to_numpy(np.int64),
                                   "rb_max": agg["rb_max"].to_numpy(float),
                                   "rb_top2": agg["rb_top2"].to_numpy(float)}))
        print(f"case{c}: {len(agg):,} gals, R_blend mean={agg['R_blend'].mean():.4f} "
              f"n_pairs mean={agg['n_pairs'].mean():.2f} max={int(agg['n_pairs'].max())}", flush=True)
    out = pd.concat(parts, ignore_index=True)
    out.to_feather(args.output)
    print(f"wrote {args.output}: {len(out):,} rows over {out['case'].nunique()} cases", flush=True)


if __name__ == "__main__":
    main()
