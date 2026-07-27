"""Add per-leg MEASURED mag + size to the constgold (+/-0.02) catalogues.

WHY. The constgold per-leg catalogues (`constant_shear_catalogue_{+,-}0.02_train.feather`) were
built without SExtractor's measured photometry: they carry `measured_e1/e2` and `S/N` only. A
SELECTION test needs a cut that MOVES with shear, i.e. a cut on a per-leg MEASURED observable --
so a measured mag / measured size cut simply could not be applied on constgold.

The raw sim output does have them: every case's
`catalogues/Shapes/shape_catalogue_detect_position_all_<tile>.feather` carries `MAG_AUTO` and
`FLUX_RADIUS` alongside `NGMIX_G1/G2`. `blendemu.response.retrieve_constant_shear` even has an
`include_measured=` switch to copy them through -- it was just never enabled for this run.

WHAT. Rather than regenerate the (42M-row x 2-leg) catalogues, this builds a slim LOOKUP keyed by
(case, shear_case, input_index) holding only the two missing columns, using EXACTLY the same
detection<->input matching as the catalogue builder (`response._load_constant_match` +
`response._input_id_lookup`, which is what `retrieve_constant_shear` calls). Merging it back onto
the per-leg catalogue on (case, input_index) is therefore row-exact by construction; rows the
builder dropped (bright-neighbour guard / unmatched) simply do not merge.

  measured_mag_auto     SExtractor MAG_AUTO   [mag]
  measured_flux_radius  SExtractor FLUX_RADIUS [pixels; x pixel_size -> arcsec]

FIREWALL: pure catalogue plumbing -- nothing is trained or fitted here.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

BLENDEMU = "/home/z/Zekang.Zhang/blendemu"
if BLENDEMU not in sys.path:
    sys.path.insert(0, BLENDEMU)

from blendemu import response  # noqa: E402

CDIR = "/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant"
TILE = "tile180.0_-0.5"
MEAS = {"MAG_AUTO": "measured_mag_auto", "FLUX_RADIUS": "measured_flux_radius"}


def one(case, shear, data_path, tile_name, real, shape_suffix):
    """Slim (case, input_index, measured_*) frame for one case x one shear leg.

    Mirrors the first half of `retrieve_constant_shear`: resolve the input catalogue's stable IDs,
    match detections to them, then read the measured columns at the matched detection rows. The
    neighbour KD-tree that `retrieve_constant_shear` runs next is NOT needed for a lookup keyed on
    input_index, so it is skipped (that is the only difference, and it cannot change which rows or
    values come out).
    """
    icat_path = os.path.join(data_path, f"case{case}_{shear}", real,
                             "catalogues/input", f"gals_info_{tile_name}.feather")
    icat = pd.read_feather(icat_path)
    _, input_ids = response._input_id_lookup(icat)
    scat, comm, idx_det = response._load_constant_match(
        data_path, case, shear, real, tile_name, shape_suffix, input_ids)
    out = {"case": np.full(comm.shape[0], case, dtype=np.int32),
           "input_index": comm.astype(np.int64)}
    for src, dst in MEAS.items():
        out[dst] = scat[src].loc[idx_det].to_numpy(dtype=np.float32)
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-path", default=CDIR)
    ap.add_argument("--tile-name", default=TILE)
    ap.add_argument("--real", default="real0")
    ap.add_argument("--shape-suffix", default="_all")
    ap.add_argument("--shears", nargs="+", default=["0.02", "-0.02"])
    ap.add_argument("--min-case", type=int, default=0)
    ap.add_argument("--max-case", type=int, default=139, help="inclusive")
    ap.add_argument("--n-jobs", type=int, default=16)
    ap.add_argument("--output", required=True, help="feather; one row per (case, shear_case, input_index)")
    args = ap.parse_args()

    from joblib import Parallel, delayed

    cases = list(range(args.min_case, args.max_case + 1))
    t0 = time.time()
    print(f"constgold measured-column lookup: cases {cases[0]}..{cases[-1]} x shears {args.shears}  "
          f"n_jobs={args.n_jobs}", flush=True)

    frames = []
    for shear in args.shears:
        def job(c, shear=shear):
            try:
                d = one(c, shear, args.data_path, args.tile_name, args.real, args.shape_suffix)
            except (FileNotFoundError, KeyError, ValueError) as e:
                print(f"  case {c} shear {shear}: {type(e).__name__}: {e}", flush=True)
                return None
            d["shear_case"] = np.float32(float(shear))
            return d
        got = Parallel(n_jobs=args.n_jobs)(delayed(job)(c) for c in cases)
        got = [g for g in got if g is not None and len(g)]
        if not got:
            print(f"  WARNING shear {shear}: nothing built", flush=True)
            continue
        part = pd.concat(got, ignore_index=True)
        print(f"  shear {shear}: {len(part):,} rows, {part['case'].nunique()} cases "
              f"({time.time()-t0:.0f}s)", flush=True)
        frames.append(part)

    if not frames:
        raise SystemExit("no data built")
    whole = pd.concat(frames, ignore_index=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    whole.to_feather(args.output)

    fr = whole["measured_flux_radius"].to_numpy(float)
    mg = whole["measured_mag_auto"].to_numpy(float)
    print(f"\nwrote {args.output}  rows={len(whole):,}")
    print(f"  measured_mag_auto     finite={np.isfinite(mg).mean():.1%}  "
          f"pct[5,50,95]={np.nanpercentile(mg,[5,50,95]).round(2)}")
    print(f"  measured_flux_radius  finite={np.isfinite(fr).mean():.1%}  "
          f"px pct[5,50,95]={np.nanpercentile(fr,[5,50,95]).round(2)}  "
          f"arcsec pct[5,50,95]={(np.nanpercentile(fr,[5,50,95])*0.2).round(3)}")
    print("BUILD_CONSTGOLD_MEASURED_DONE", flush=True)


if __name__ == "__main__":
    main()
