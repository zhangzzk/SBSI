"""Catalogue-only outer-scene features for the controlled anchor experiment.

No image or response is rendered here.  For each retained anchor, sum true
r-band flux in three annuli beyond the deployed 10-arcsec BlendEMU aperture.
Two deterministic profile-extent summaries are also recorded.  These features
are used only to test whether the development/validation change in the matched
outside response follows the outer input scene.
"""
from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


EDGES = np.asarray([10.0, 15.0, 20.0, 30.0])
FLUX_COLUMNS = ["logflux_outer_10_15", "logflux_outer_15_20", "logflux_outer_20_30"]


def outer_summaries(distance: np.ndarray, flux: np.ndarray, re_value: np.ndarray) -> dict:
    shell = np.searchsorted(EDGES, distance, side="right") - 1
    inside = (shell >= 0) & (shell < len(FLUX_COLUMNS))
    sums = np.zeros(len(FLUX_COLUMNS), dtype=float)
    np.add.at(sums, shell[inside], flux[inside])
    safe_distance = np.maximum(distance[inside], 1.0e-6)
    extent = re_value[inside] / safe_distance
    weighted = np.sum(flux[inside] * extent * extent, dtype=np.float64)
    return {
        **{name: float(np.log10(1.0 + value)) for name, value in zip(FLUX_COLUMNS, sums)},
        "logflux_re2_over_d2_outer_10_30": float(np.log10(1.0 + weighted)),
        "max_re_over_d_outer_10_30": float(np.max(extent)) if len(extent) else 0.0,
        "n_outer_10_30": int(inside.sum()),
    }


def process_case(task: tuple[int, str, np.ndarray]) -> pd.DataFrame:
    case, base, anchor_ids = task
    path = os.path.join(base, f"gals{case}_0.05.feather")
    field = pd.read_feather(path, columns=["index", "RA", "DEC", "r", "Re"])
    if field["index"].duplicated().any():
        raise RuntimeError(f"case {case}: duplicate source index")
    row_by_id = pd.Series(np.arange(len(field), dtype=np.int64), index=field["index"])
    missing = np.setdiff1d(anchor_ids, row_by_id.index.to_numpy(np.int64))
    if len(missing):
        raise RuntimeError(f"case {case}: {len(missing)} anchors absent from field")
    anchor_rows = row_by_id.loc[anchor_ids].to_numpy(np.int64)
    dec0 = float(np.median(field["DEC"]))
    xy = np.column_stack([
        field["RA"].to_numpy(float) * np.cos(np.deg2rad(dec0)) * 3600.0,
        field["DEC"].to_numpy(float) * 3600.0,
    ])
    flux = np.power(10.0, -0.4 * (field["r"].to_numpy(float) - 30.0))
    re_value = field["Re"].to_numpy(float)
    if not np.isfinite(xy).all() or not np.isfinite(flux).all() or not np.isfinite(re_value).all():
        raise RuntimeError(f"case {case}: non-finite scene truth")
    tree = cKDTree(xy)
    rows = []
    for anchor_id, anchor_row in zip(anchor_ids, anchor_rows):
        local = np.asarray(tree.query_ball_point(xy[anchor_row], r=EDGES[-1]), dtype=np.int64)
        local = local[local != anchor_row]
        distance = np.hypot(
            xy[local, 0] - xy[anchor_row, 0],
            xy[local, 1] - xy[anchor_row, 1],
        )
        rows.append({
            "case": int(case), "input_index": int(anchor_id),
            **outer_summaries(distance, flux[local], re_value[local]),
        })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--anchors", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--case-min", type=int, default=200)
    ap.add_argument("--case-max", type=int, default=299)
    ap.add_argument("--n-jobs", type=int, default=8)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    if os.path.exists(args.output):
        raise FileExistsError(f"refusing to overwrite {args.output}")
    anchors = pd.read_feather(args.anchors, columns=["case", "input_index"])
    anchors = anchors[(anchors.case >= args.case_min) & (anchors.case <= args.case_max)]
    anchors = anchors.drop_duplicates(["case", "input_index"])
    cases = list(range(args.case_min, args.case_max + 1))
    if sorted(anchors.case.unique().tolist()) != cases:
        raise RuntimeError("anchor table does not cover every requested case")
    tasks = [(case, args.base, anchors.loc[anchors.case == case, "input_index"].to_numpy(np.int64))
             for case in cases]
    parts = []
    with ProcessPoolExecutor(max_workers=args.n_jobs) as executor:
        futures = {executor.submit(process_case, task): task[0] for task in tasks}
        for future in as_completed(futures):
            part = future.result()
            parts.append(part)
            print(f"case {futures[future]}: anchors={len(part):,}", flush=True)
    out = pd.concat(parts, ignore_index=True).sort_values(
        ["case", "input_index"], kind="mergesort",
    ).reset_index(drop=True)
    if out.duplicated(["case", "input_index"]).any() or len(out) != len(anchors):
        raise RuntimeError("outer-feature key coverage failure")
    feature_columns = [*FLUX_COLUMNS, "logflux_re2_over_d2_outer_10_30",
                       "max_re_over_d_outer_10_30", "n_outer_10_30"]
    if not np.isfinite(out[feature_columns].to_numpy(float)).all():
        raise RuntimeError("non-finite outer feature")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_feather(args.output)
    with open(os.path.splitext(args.output)[0] + ".json", "x", encoding="utf-8") as handle:
        json.dump({
            "definition": "catalogue-only intrinsic outer-scene summaries",
            "shell_edges_arcsec": EDGES.tolist(), "feature_columns": feature_columns,
            "n_rows": len(out), "n_cases": len(cases), "rendering": "none",
        }, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"wrote {args.output}: {len(out):,} rows", flush=True)
    print("ANCHORBLEND_OUTER_FEATURES_DONE", flush=True)


if __name__ == "__main__":
    main()
