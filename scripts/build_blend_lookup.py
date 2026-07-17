"""Per-(case, input_index) BlendEMU blending-response lookup for the constant-shear gold set.

Each constant case has its OWN input field (distinct galaxies/positions), so R_blend must be
computed per case. For each case we run the BlendEMU response emulator on that case's input
`gals_info` feather: for every galaxy (as PRIMARY = the measured object) sum the emulator's
per-pair blending response over its sheared neighbours (SECONDARY, drawn from the full field).

This is exactly the validated call from validate_constant_with_blend.r_blend_lookup, but keyed by
(case, input_index) instead of a single tile -- fixing the one-tile dilution (0.194 -> 0.053).

Output feather columns: case, input_index, R_blend   (only galaxies the emulator scores; the
validator treats any (case,input_index) absent here as R_blend=0, correct for r>26 / isolated).
"""
import argparse, os, sys
import numpy as np
import pandas as pd
import pyarrow.feather as pf

CBASE = "/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant"
BLEND_MODELS = "/home/z/Zekang.Zhang/blendemu/models"
COND = dict(pixel_size=0.2, zero_point=30.0, psf_fwhm=0.73, moffat_beta=2.224, pixel_rms=0.312)
TILE = "tile180.0_-0.5"


def input_feather(case, sign="0.02", base=CBASE):
    return f"{base}/case{case}_{sign}/real0/catalogues/input/gals_info_{TILE}.feather"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, nargs="+", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--sign", default="0.02", help="which shear-sign input field to use for positions")
    ap.add_argument("--base", default=CBASE, help="sim set root (constant vs half-shear base)")
    ap.add_argument("--tag", default="lsst_r", help="emulator tag (e.g. lsst_r_extdom for the extended-domain model)")
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
        reg = pred.predict_response(t, t)                       # primary=secondary=full field
        pk = [col for col in reg.columns if col.startswith("index")][0]   # index_input_p (primary)
        rb = reg.groupby(pk)["response"].sum()
        parts.append(pd.DataFrame({"case": c,
                                   "input_index": rb.index.to_numpy().astype(np.int64),
                                   "R_blend": rb.to_numpy(float)}))
        print(f"case{c}: {len(rb):,} gals scored, R_blend mean={rb.mean():.4f} "
              f"median={np.median(rb):.4f}", flush=True)
    out = pd.concat(parts, ignore_index=True)
    out.to_feather(args.output)
    print(f"wrote {args.output}: {len(out):,} rows over {out['case'].nunique()} cases", flush=True)


if __name__ == "__main__":
    main()
