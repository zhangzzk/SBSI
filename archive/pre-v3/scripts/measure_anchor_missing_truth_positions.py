"""Measure previously unselected coherent anchors at fixed truth positions.

One SLURM task handles one case.  Only manifest anchors absent from the stored
both-leg detection/match response are measured, using the same deterministic
ngmix routine and seeds as ``measure_anchor_fixed_positions.py``.  Images are
not rerendered and no detection catalogue is consulted.
"""
from __future__ import annotations

import argparse
import os

from astropy.io import fits
from astropy import wcs
from mpi4py import MPI
import numpy as np
import pandas as pd

from blendemu import shape
from scripts.measure_anchor_fixed_positions import (
    deterministic_ngmix, image_paths, subpixel_centre,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--response", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--case-offset", type=int, default=200)
    parser.add_argument("--n-cases", type=int, default=100)
    parser.add_argument("--g", type=float, default=0.05)
    parser.add_argument("--stamp-size", type=int, default=48)
    parser.add_argument("--pixel-scale", type=float, default=0.2)
    args = parser.parse_args()

    rank = int(os.environ.get("SLURM_PROCID", MPI.COMM_WORLD.Get_rank()))
    if rank >= args.n_cases:
        return
    case = args.case_offset + rank
    os.makedirs(args.output_dir, exist_ok=True)
    output = os.path.join(args.output_dir, f"case{case}.feather")
    if os.path.exists(output):
        raise FileExistsError(f"refusing to overwrite {output}")

    manifest = pd.read_feather(os.path.join(args.base, f"anchors_case{case}.feather"))
    selected = pd.read_feather(args.response, columns=["case", "input_index"])
    selected_ids = selected.loc[selected.case == case, "input_index"].to_numpy(np.int64)
    missing = manifest.loc[~manifest["index"].isin(selected_ids)].copy().sort_values("index")
    ids = missing["index"].to_numpy(np.int64)
    result = pd.DataFrame({"case": np.full(len(ids), case), "input_index": ids})
    for leg_index, sign in enumerate((args.g, -args.g)):
        label = "plus" if sign > 0 else "minus"
        science_path, psf_path = image_paths(args.base, case, sign)
        image, header = fits.getdata(science_path, header=True)
        psf_image = fits.getdata(psf_path)
        world = wcs.WCS(header)
        x, y = world.wcs_world2pix(
            missing.RA.to_numpy(float), missing.DEC.to_numpy(float), 1,
        )
        measured = np.full((len(ids), 2), np.nan)
        for index, input_id in enumerate(ids):
            stamp = shape.cutout(image, x[index], y[index], stamp_size=args.stamp_size)
            centre = subpixel_centre(x[index], y[index], args.stamp_size, stamp.shape)
            seed = int((case * 1_000_003 + int(input_id) * 17 + leg_index * 97) % (2**32 - 1))
            try:
                measured[index] = deterministic_ngmix(
                    stamp, psf_image, args.pixel_scale, centre, seed,
                )
            except Exception:
                measured[index] = np.nan
        result[f"g1_truthpos_{label}"] = measured[:, 0]
        result[f"g2_truthpos_{label}"] = measured[:, 1]
    result.to_feather(output)
    finite = np.isfinite(result.iloc[:, 2:].to_numpy(float)).all(axis=1)
    print(
        f"case {case}: manifest={len(manifest)} selected={len(selected_ids)} "
        f"missing={len(result)} finite={finite.sum()} output={output}", flush=True,
    )
    print("ANCHOR_MISSING_TRUTHPOS_CASE_DONE", flush=True)


if __name__ == "__main__":
    main()
