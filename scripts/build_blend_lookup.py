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
import pathlib as _pathlib, sys as _sys  # noqa: E402  -- make `sbs_shear` importable
_sys.path.insert(0, str(next(p for p in _pathlib.Path(__file__).resolve().parents
                             if (p / 'sbs_shear').is_dir())))
from sbs_shear import paths  # noqa: E402
import pathlib as _pathlib, sys as _sys  # noqa: E402  -- make `sbs_shear` importable
_sys.path.insert(0, str(next(p for p in _pathlib.Path(__file__).resolve().parents
                             if (p / 'sbs_shear').is_dir())))
from sbs_shear.emulator import load_blending_predictor  # noqa: E402
import argparse, os, sys
import numpy as np
import pandas as pd
import pyarrow.feather as pf

CBASE = f"{paths.CONST_SIM_DIR}"
TILE = "tile180.0_-0.5"


def input_feather(case, sign="0.02", base=CBASE):
    return f"{base}/case{case}_{sign}/real0/catalogues/input/gals_info_{TILE}.feather"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, nargs="+", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--sign", default="0.02", help="which shear-sign input field to use for positions")
    ap.add_argument("--base", default=CBASE, help="sim set root (constant vs half-shear base)")
    ap.add_argument("--max-distance", type=float, default=None,
                    help="only sum emulator pairs with distance<=this (arcsec). Set to the sim's\n                         blend/render radius (constgold=3.0) so far un-rendered neighbours do NOT\n                         over-add; also zeros isolated objects (no neighbour within the radius).")
    ap.add_argument("--tag", default="lsst_r", help="emulator tag (e.g. lsst_r_extdom for the extended-domain model)")
    args = ap.parse_args()

    pred = load_blending_predictor(tag=args.tag)

    parts = []
    for c in args.cases:
        fp = input_feather(c, args.sign, args.base)
        if not os.path.exists(fp):
            print(f"case{c}: MISSING {fp}", flush=True); continue
        t = pf.read_table(fp).to_pandas()
        t = t.rename(columns={col: col.replace("_input", "") for col in t.columns})
        reg = pred.predict_response(t, t)                       # primary=secondary=full field
        if args.max_distance:                                   # gate to the sim's blend/render radius:
            n0 = len(reg)                                       # neighbours beyond it are NOT rendered ->
            reg = reg[reg["distance"] <= args.max_distance]     # they do NOT blend in the measured response
            print(f"  distance<= {args.max_distance}: kept {len(reg):,}/{n0:,} pairs", flush=True)
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
