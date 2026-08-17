"""Compute six true radial shell fluxes and Eq. 17 purity for scene anchors.

The six disjoint annuli split each previously tested 0--1, 1--3 and 3--10
arcsec range in two: [0,0.5), [0.5,1), [1,2), [2,3), [3,5), [5,10).
Fluxes are absolute intrinsic simulation-count flux, encoded as log10(1+F).
Purity uses the same noiseless PSF-convolved operational definition validated
by ``compute_v22_q3_purity.py``.
"""
from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import galsim
import numpy as np
import pandas as pd
import pyarrow.feather as pf
from scipy.spatial import cKDTree

from scripts.compute_v22_q3_purity import overlap_purity


TILE = "tile180.0_-0.5"
ZERO_POINT = 30.0
SHELL_EDGES = np.asarray([0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0])
SHELL_COLUMNS = [
    "logflux_abs_shell_0_0p5", "logflux_abs_shell_0p5_1",
    "logflux_abs_shell_1_2", "logflux_abs_shell_2_3",
    "logflux_abs_shell_3_5", "logflux_abs_shell_5_10",
]


def input_path(base: str, case: int, sign: str) -> str:
    return os.path.join(
        base, f"case{case}_{sign}", "real0", "catalogues", "input",
        f"gals_info_{TILE}.feather",
    )


def read_field(path: str) -> pd.DataFrame:
    source = pf.read_table(path, columns=[
        "index_input", "RA_input", "DEC_input", "position_angle_input",
        "Re_input", "axis_ratio_input", "sersic_n_input", "r_input",
    ]).to_pandas()
    return source.rename(columns={
        "index_input": "index", "RA_input": "RA", "DEC_input": "DEC",
        "position_angle_input": "position_angle", "Re_input": "Re",
        "axis_ratio_input": "axis_ratio", "sersic_n_input": "sersic_n",
        "r_input": "r",
    })


def process_case(task: tuple) -> tuple[pd.DataFrame, dict]:
    (case, base, sign, anchor_ids, stamp_size, max_stamp_size, pixel_scale,
     neighbour_radius, sky_rms, threshold_fraction, psf_fwhm, psf_beta) = task
    field = read_field(input_path(base, case, sign))
    if field["index"].duplicated().any():
        raise RuntimeError(f"case {case}: duplicate source index")
    by_id = pd.Series(np.arange(len(field), dtype=np.int64), index=field["index"])
    missing = np.setdiff1d(anchor_ids, by_id.index.to_numpy(np.int64))
    if len(missing):
        raise RuntimeError(f"case {case}: {len(missing)} anchors absent from input field")
    anchor_positions = by_id.loc[anchor_ids].to_numpy(np.int64)
    dec0 = float(np.median(field["DEC"]))
    xy = np.column_stack([
        field["RA"].to_numpy(float) * np.cos(np.deg2rad(dec0)) * 3600.0,
        field["DEC"].to_numpy(float) * 3600.0,
    ])
    tree = cKDTree(xy)
    flux = 10.0 ** (-0.4 * (field["r"].to_numpy(float) - ZERO_POINT))
    psf = galsim.Moffat(beta=psf_beta, fwhm=psf_fwhm, trunc=4.5 * psf_fwhm)
    threshold = threshold_fraction * sky_rms
    rows = []
    for anchor_id, anchor_position in zip(anchor_ids, anchor_positions):
        local = np.asarray(tree.query_ball_point(
            xy[anchor_position], r=neighbour_radius), dtype=np.int64)
        local = local[local != anchor_position]
        offset = xy[local] - xy[anchor_position]
        distance = np.hypot(offset[:, 0], offset[:, 1])
        shell = np.searchsorted(SHELL_EDGES, distance, side="right") - 1
        shell_flux = np.zeros(6, dtype=np.float64)
        inside = (shell >= 0) & (shell < 6)
        np.add.at(shell_flux, shell[inside], flux[local[inside]])

        current_stamp = stamp_size
        while True:
            stats = overlap_purity(
                field.iloc[anchor_position], field.iloc[local],
                stamp_size=current_stamp, pixel_scale=pixel_scale, psf=psf,
                threshold=threshold,
            )
            if not stats["mask_touches_stamp_edge"]:
                break
            if current_stamp >= max_stamp_size:
                raise RuntimeError(
                    f"case {case} anchor {anchor_id}: mask reaches max stamp edge")
            current_stamp = min(2 * current_stamp, max_stamp_size)
        row = {"case": case, "input_index": int(anchor_id)}
        row.update({name: float(np.log10(1.0 + value))
                    for name, value in zip(SHELL_COLUMNS, shell_flux)})
        row.update({
            "purity_eq17": stats["purity_eq17"],
            "true_blendedness_eq17": stats["true_blendedness_eq17"],
            "n_mask_pixels": stats["n_mask_pixels"],
            "n_local_neighbours": stats["n_local_neighbours"],
            "stamp_size_used": current_stamp,
            "mask_touches_stamp_edge": stats["mask_touches_stamp_edge"],
        })
        rows.append(row)
    out = pd.DataFrame(rows)
    return out, {
        "case": case, "n_field_sources": len(field), "n_anchors": len(out),
        "mean_purity": float(out["purity_eq17"].mean()),
        "max_stamp": int(out["stamp_size_used"].max()),
        "edge_masks": int(out["mask_touches_stamp_edge"].sum()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--sign", required=True)
    ap.add_argument("--cases", type=int, nargs="+", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--n-jobs", type=int, default=1)
    ap.add_argument("--stamp-size", type=int, default=96)
    ap.add_argument("--max-stamp-size", type=int, default=384)
    ap.add_argument("--pixel-scale", type=float, default=0.2)
    ap.add_argument("--neighbour-radius", type=float, default=15.0)
    ap.add_argument("--sky-rms", type=float, default=0.311519)
    ap.add_argument("--threshold-fraction", type=float, default=0.05)
    ap.add_argument("--psf-fwhm", type=float, default=0.73)
    ap.add_argument("--psf-beta", type=float, default=2.224068)
    args = ap.parse_args()
    if os.path.exists(args.output):
        raise FileExistsError(f"refusing to overwrite {args.output}")
    manifest = pf.read_table(args.manifest, columns=["case", "input_index"]).to_pandas()
    manifest = manifest[manifest["case"].isin(args.cases)].copy()
    if manifest["case"].nunique() != len(set(args.cases)):
        raise RuntimeError("manifest does not cover every requested case")
    tasks = []
    for case in args.cases:
        ids = manifest.loc[manifest["case"] == case, "input_index"].to_numpy(np.int64)
        tasks.append((
            case, args.base, args.sign, ids, args.stamp_size, args.max_stamp_size,
            args.pixel_scale, args.neighbour_radius, args.sky_rms,
            args.threshold_fraction, args.psf_fwhm, args.psf_beta,
        ))
    parts, summaries = [], []
    with ProcessPoolExecutor(max_workers=args.n_jobs) as executor:
        futures = {executor.submit(process_case, task): task[0] for task in tasks}
        for future in as_completed(futures):
            part, summary = future.result()
            parts.append(part); summaries.append(summary)
            print(f"case {summary['case']}: anchors={summary['n_anchors']:,}, "
                  f"mean rho={summary['mean_purity']:.5f}, "
                  f"max stamp={summary['max_stamp']}", flush=True)
    out = pd.concat(parts, ignore_index=True).sort_values(
        ["case", "input_index"], kind="mergesort").reset_index(drop=True)
    if out.duplicated(["case", "input_index"]).any():
        raise RuntimeError("duplicate output key")
    if out["mask_touches_stamp_edge"].any():
        raise RuntimeError("final purity mask touches a stamp edge")
    expected = len(manifest)
    if len(out) != expected:
        raise RuntimeError(f"expected {expected:,} rows, got {len(out):,}")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_feather(args.output)
    sidecar = os.path.splitext(args.output)[0] + ".json"
    with open(sidecar, "x", encoding="utf-8") as handle:
        json.dump({
            "manifest": os.path.abspath(args.manifest), "base": os.path.abspath(args.base),
            "sign": args.sign, "cases": sorted(args.cases), "n_rows": len(out),
            "shell_edges_arcsec": SHELL_EDGES.tolist(), "shell_columns": SHELL_COLUMNS,
            "shell_definition": "log10(1 + absolute intrinsic neighbour flux)",
            "purity_definition": "Nourbakhsh et al. 2022 Eq. 17; unsheared PSF-convolved profiles",
            "summaries": sorted(summaries, key=lambda item: item["case"]),
        }, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"wrote {args.output} and {sidecar}", flush=True)


if __name__ == "__main__":
    main()
