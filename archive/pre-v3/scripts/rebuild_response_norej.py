"""Rebuild blendemu's response catalogue WITHOUT the bright-neighbour rejection.

WHY
---
`blendemu/response.py::retrieve_response` drops every primary that has a detected neighbour within
3" more than 5x brighter (`remove_detection_w_bright_neighbour(..., ratio_max=5, r_max=3/3600)`)
before computing the blending-response label. The half-shear ruler -- and the gold sample the m
pipeline actually sums over -- applies no such cut, so the emulator is asked to predict on pairs its
labels were never allowed to contain. See scripts/eval_emu_label_gap.py and
scripts/eval_contrast_cut.py for the measurements that motivate this.

This regenerates the labels with that rejection disabled. It is a CATALOGUE-BUILD step only: the
shape and cross-match catalogues already exist on disk for every case, so nothing is re-simulated
and no image is re-measured. Runtime is dominated by reading them.

WHERE THIS BELONGS
------------------
Permanently, the rejection should be a config knob in blendemu (`catalogues.response.reject_ratio_max`,
default 5 to preserve current behaviour). It is NOT edited in place here because
/home/z/Zekang.Zhang/blendemu currently has uncommitted work on `main`; touching it would collide
with that. So this driver monkeypatches blendemu's own function instead of forking its builder --
per AGENTS.md, catalogue building stays blendemu's job and no builder is reimplemented here.

USAGE
-----
    python scripts/rebuild_response_norej.py --ratio-max inf --n-jobs 16

`--ratio-max inf` removes the rejection entirely; a finite value (e.g. 100) keeps a loose guard
against catastrophic shape failures next to very bright objects, which is what the cut was for.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc
from joblib import Parallel, delayed

BE = "/home/z/Zekang.Zhang/blendemu"
sys.path.insert(0, BE)
sys.path.insert(0, os.path.join(BE, "scripts"))

DEFAULT_CONFIG = os.path.join(BE, "configs/fs2_lsst_r_extnbr_ho.yaml")


def patch_rejection(ratio_max):
    """Replace blendemu's bright-neighbour rejection with a looser (or absent) one.

    response.py holds a reference to the MODULE (`from . import utils`), so rebinding the attribute
    on the module is seen by the builder. Must be re-applied inside every worker process.
    """
    from blendemu import utils

    if not hasattr(utils, "_orig_remove_detection_w_bright_neighbour"):
        utils._orig_remove_detection_w_bright_neighbour = utils.remove_detection_w_bright_neighbour

    orig = utils._orig_remove_detection_w_bright_neighbour
    ratio = float(ratio_max)

    def patched(x, y, flux, ratio_max=None, r_min=0, r_max=5 / 3600):
        # ratio_max from the caller is IGNORED on purpose: the point is to override the hard-coded
        # value inside retrieve_response, which is where the cut is applied.
        if not np.isfinite(ratio):
            return np.array([], dtype=int)
        return orig(x, y, flux, ratio_max=ratio, r_min=r_min, r_max=r_max)

    utils.remove_detection_w_bright_neighbour = patched
    return ratio


def build_case(case, ratio_max, r_cfg, shear_cases, tile_name, out_path):
    patch_rejection(ratio_max)
    from blendemu import response

    try:
        df = response.retrieve_response(
            case=case, r_max=r_cfg["r_max"], r_min=r_cfg.get("r_min", 0), k=r_cfg["k"],
            real="real0", tile_name=tile_name, data_path=out_path, shear_cases=shear_cases,
        )
    except FileNotFoundError as e:
        print(f"  case {case}: MISSING {e}", flush=True)
        return None
    df = df[np.isfinite(df["delta_et1"].to_numpy(float))].reset_index(drop=True)
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--ratio-max", default="inf",
                    help="'inf' disables the rejection; a number keeps a looser guard")
    ap.add_argument("--n-cases", type=int, default=None, help="override simulation.response.n_cases")
    ap.add_argument("--n-jobs", type=int,
                    default=int(os.environ.get("SLURM_CPUS_PER_TASK", "16")))
    ap.add_argument("--suffix", default="norej")
    args = ap.parse_args()

    from blendemu.config import load_config
    import run_pipeline as RP

    cfg = load_config(args.config)
    out_path = cfg["simulation"]["output_path"]
    tile_name = cfg["shape_measurement"]["tile_name"]
    r_cfg = cfg["catalogues"]["response"]
    r_sim_cfg = RP._catalogue_sim_cfg(cfg["simulation"], r_cfg, "response")
    shear_cases, shear_scale = RP._paired_shear_settings(r_sim_cfg, label="catalogues.response")
    n_cases = args.n_cases if args.n_cases is not None else int(r_sim_cfg["n_cases"])

    ratio = float(args.ratio_max)
    print(f"config       : {args.config}")
    print(f"out_path     : {out_path}")
    print(f"shear_cases  : {shear_cases}  (finite-difference step Dg={shear_scale})")
    print(f"r_max={r_cfg['r_max']}\"  k={r_cfg['k']}  cases=0..{n_cases-1}  n_jobs={args.n_jobs}")
    print(f"rejection    : ratio_max={'DISABLED' if not np.isfinite(ratio) else ratio}"
          f"   (certified catalogue used ratio_max=5 within 3\")", flush=True)

    out_file = os.path.join(out_path, f"response_catalogue_{args.suffix}_train.feather")
    if os.path.exists(out_file):
        raise SystemExit(f"refusing to overwrite existing {out_file}")

    t0 = time.time()
    writer = None
    n_rows = 0
    schema = None
    # Chunk the case loop so at most n_jobs frames are alive at once; write straight through to
    # an IPC stream instead of concatenating 200 cases in RAM.
    chunk = max(args.n_jobs, 1)
    for start in range(0, n_cases, chunk):
        cases = list(range(start, min(start + chunk, n_cases)))
        frames = Parallel(n_jobs=args.n_jobs)(
            delayed(build_case)(c, ratio, r_cfg, shear_cases, tile_name, out_path) for c in cases)
        frames = [f for f in frames if f is not None and len(f)]
        if not frames:
            continue
        part = pd.concat(frames, ignore_index=True)
        del frames
        tbl = pa.Table.from_pandas(part, preserve_index=False)
        if writer is None:
            schema = tbl.schema
            writer = ipc.new_file(out_file, schema)
        else:
            tbl = tbl.cast(schema)
        for b in tbl.to_batches(max_chunksize=65536):
            writer.write_batch(b)
        n_rows += len(part)
        print(f"  cases {cases[0]}..{cases[-1]}: +{len(part):,} rows (total {n_rows:,})  "
              f"[{time.time()-t0:.0f}s]", flush=True)
        del part, tbl
    if writer is not None:
        writer.close()

    print(f"\nwrote {n_rows:,} rows -> {out_file}  ({time.time()-t0:.0f}s)")
    print(f"certified catalogue for comparison: response_catalogue_train.feather")
    print("REBUILD_NOREJ_DONE", flush=True)


if __name__ == "__main__":
    main()
