"""Compute V2.7 absolute intrinsic neighbour flux in six radial shells.

This is the catalogue-only arm of V2.7.  It intentionally does not import or
render GalSim profiles: each feature is the sum of intrinsic r-band flux from
all other input-field sources whose centre falls in the corresponding annulus,
encoded as ``log10(1 + F_shell)``.
"""
from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.feather as pf
from scipy.spatial import cKDTree


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
        "index_input", "RA_input", "DEC_input", "r_input",
    ]).to_pandas()
    return source.rename(columns={
        "index_input": "index", "RA_input": "RA", "DEC_input": "DEC",
        "r_input": "r",
    })


def accumulate_shell_flux(distance: np.ndarray, flux: np.ndarray) -> np.ndarray:
    """Sum neighbour flux into the six fixed annuli."""
    shell = np.searchsorted(SHELL_EDGES, distance, side="right") - 1
    inside = (shell >= 0) & (shell < len(SHELL_COLUMNS))
    out = np.zeros(len(SHELL_COLUMNS), dtype=np.float64)
    np.add.at(out, shell[inside], flux[inside])
    return out


def process_case(task: tuple) -> tuple[pd.DataFrame, dict]:
    case, base, sign, anchor_ids = task
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
    flux = 10.0 ** (-0.4 * (field["r"].to_numpy(float) - ZERO_POINT))
    if not np.isfinite(xy).all() or not np.isfinite(flux).all():
        raise RuntimeError(f"case {case}: non-finite truth coordinate or flux")
    tree = cKDTree(xy)
    rows = []
    neighbour_counts = []
    for anchor_id, anchor_position in zip(anchor_ids, anchor_positions):
        local = np.asarray(
            tree.query_ball_point(xy[anchor_position], r=float(SHELL_EDGES[-1])),
            dtype=np.int64,
        )
        local = local[local != anchor_position]
        distance = np.hypot(
            xy[local, 0] - xy[anchor_position, 0],
            xy[local, 1] - xy[anchor_position, 1],
        )
        shell_flux = accumulate_shell_flux(distance, flux[local])
        row = {"case": case, "input_index": int(anchor_id)}
        row.update({name: float(np.log10(1.0 + value))
                    for name, value in zip(SHELL_COLUMNS, shell_flux)})
        rows.append(row)
        neighbour_counts.append(len(local))
    out = pd.DataFrame(rows)
    return out, {
        "case": case, "n_field_sources": len(field), "n_anchors": len(out),
        "mean_neighbours_within_10arcsec": float(np.mean(neighbour_counts)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--sign", required=True)
    ap.add_argument("--cases", type=int, nargs="+", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--n-jobs", type=int, default=1)
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
        tasks.append((case, args.base, args.sign, ids))
    parts, summaries = [], []
    with ProcessPoolExecutor(max_workers=args.n_jobs) as executor:
        futures = {executor.submit(process_case, task): task[0] for task in tasks}
        for future in as_completed(futures):
            part, summary = future.result()
            parts.append(part)
            summaries.append(summary)
            print(f"case {summary['case']}: anchors={summary['n_anchors']:,}, "
                  f"mean neighbours={summary['mean_neighbours_within_10arcsec']:.2f}",
                  flush=True)
    out = pd.concat(parts, ignore_index=True).sort_values(
        ["case", "input_index"], kind="mergesort").reset_index(drop=True)
    if out.duplicated(["case", "input_index"]).any():
        raise RuntimeError("duplicate output key")
    expected = sum(len(task[3]) for task in tasks)
    if len(out) != expected:
        raise RuntimeError(f"expected {expected:,} output rows, got {len(out):,}")
    if not np.isfinite(out[SHELL_COLUMNS].to_numpy(float)).all():
        raise RuntimeError("non-finite shell feature")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_feather(args.output)
    with open(os.path.splitext(args.output)[0] + ".json", "x", encoding="utf-8") as handle:
        json.dump({
            "output": os.path.abspath(args.output),
            "manifest": os.path.abspath(args.manifest),
            "base": os.path.abspath(args.base), "sign": args.sign,
            "cases": sorted(args.cases), "n_rows": len(out),
            "shell_edges_arcsec": SHELL_EDGES.tolist(),
            "shell_columns": SHELL_COLUMNS,
            "definition": "log10(1 + absolute intrinsic r-band flux in annulus)",
            "rendering": "none; catalogue positions and magnitudes only",
            "case_summaries": sorted(summaries, key=lambda item: item["case"]),
        }, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"wrote {args.output}: {len(out):,} rows", flush=True)


if __name__ == "__main__":
    main()
