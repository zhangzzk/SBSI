"""Pre-render audit of matched random-direction guarded anchor catalogues."""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


def label(g: float) -> str:
    return str(float(g))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root10", required=True)
    ap.add_argument("--root15", required=True)
    ap.add_argument("--cases", type=int, nargs="+", required=True)
    ap.add_argument("--g", type=float, default=0.05)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    if os.path.exists(args.output):
        raise FileExistsError(f"refusing to overwrite {args.output}")
    summaries = []
    for case in args.cases:
        manifest10 = pd.read_feather(
            os.path.join(args.root10, f"anchors_case{case}.feather")
        )
        manifest15 = pd.read_feather(
            os.path.join(args.root15, f"anchors_case{case}.feather")
        )
        if not manifest10[["index", "u1", "u2"]].equals(
            manifest15[["index", "u1", "u2"]]
        ):
            raise RuntimeError(f"case {case}: anchor manifests differ")
        per_radius = {}
        plus_by_radius = {}
        for radius, root in ((10.0, args.root10), (15.0, args.root15)):
            columns = ["index", "RA", "DEC", "g1", "g2"]
            plus = pd.read_feather(
                os.path.join(root, f"gals{case}_{label(args.g)}.feather"), columns=columns,
            )
            minus = pd.read_feather(
                os.path.join(root, f"gals{case}_{label(-args.g)}.feather"), columns=columns,
            )
            for column in ("index", "RA", "DEC"):
                if not np.array_equal(plus[column], minus[column]):
                    raise RuntimeError(f"case {case} radius {radius}: latent column {column} differs")
            if not np.array_equal(plus[["g1", "g2"]].to_numpy(float),
                                  -minus[["g1", "g2"]].to_numpy(float)):
                raise RuntimeError(f"case {case} radius {radius}: shear legs are not antithetic")
            by_id = pd.Series(np.arange(len(plus), dtype=np.int64), index=plus["index"])
            anchor_rows = by_id.loc[manifest10["index"]].to_numpy(np.int64)
            anchor = np.zeros(len(plus), dtype=bool)
            anchor[anchor_rows] = True
            if np.any(plus.loc[anchor, ["g1", "g2"]].to_numpy(float)):
                raise RuntimeError(f"case {case} radius {radius}: anchor is sheared")
            dec0 = float(np.median(plus["DEC"]))
            xy = np.column_stack([
                plus["RA"].to_numpy(float) * np.cos(np.deg2rad(dec0)) * 3600.0,
                plus["DEC"].to_numpy(float) * 3600.0,
            ])
            distance, owner_relative = cKDTree(xy[anchor]).query(xy, k=1)
            owner_rows = anchor_rows[owner_relative]
            expected = (distance <= radius) & (~anchor)
            norm = np.hypot(plus["g1"].to_numpy(float), plus["g2"].to_numpy(float))
            actual = norm > 0
            if not np.array_equal(expected, actual):
                raise RuntimeError(
                    f"case {case} radius {radius}: shear mask differs on {(expected != actual).sum()} rows"
                )
            if len(norm[actual]) and not np.allclose(norm[actual], args.g, rtol=0, atol=1e-12):
                raise RuntimeError(f"case {case} radius {radius}: non-unit shear magnitude")
            directions = manifest10.set_index("index")[["u1", "u2"]]
            row_u1 = np.zeros(len(plus), dtype=float)
            row_u2 = np.zeros(len(plus), dtype=float)
            row_u1[anchor_rows] = directions.loc[plus.loc[anchor, "index"], "u1"]
            row_u2[anchor_rows] = directions.loc[plus.loc[anchor, "index"], "u2"]
            expected_g = args.g * np.column_stack([
                row_u1[owner_rows[actual]], row_u2[owner_rows[actual]],
            ])
            error = np.max(np.abs(
                plus.loc[actual, ["g1", "g2"]].to_numpy(float) - expected_g
            )) if actual.any() else 0.0
            if error > 1e-12:
                raise RuntimeError(f"case {case} radius {radius}: owner direction error {error}")
            per_radius[str(int(radius))] = {
                "n_sheared_sources": int(actual.sum()),
                "max_sheared_distance": float(distance[actual].max()) if actual.any() else 0.0,
                "min_unsheared_nonanchor_distance": float(distance[(~actual) & (~anchor)].min()),
                "max_owner_direction_error": float(error),
            }
            plus_by_radius[radius] = plus
        for column in ("index", "RA", "DEC"):
            if not np.array_equal(plus_by_radius[10.0][column], plus_by_radius[15.0][column]):
                raise RuntimeError(f"case {case}: radius arms differ in latent {column}")
        sheared10 = np.hypot(plus_by_radius[10.0].g1, plus_by_radius[10.0].g2) > 0
        sheared15 = np.hypot(plus_by_radius[15.0].g1, plus_by_radius[15.0].g2) > 0
        if np.any(sheared10 & ~sheared15):
            raise RuntimeError(f"case {case}: radius-10 mask is not a subset of radius-15")
        summaries.append({
            "case": case, "n_sources": len(plus_by_radius[10.0]),
            "n_anchors": len(manifest10), "radii": per_radius,
        })
        print(
            f"case {case}: anchors={len(manifest10):,} "
            f"sheared10={per_radius['10']['n_sheared_sources']:,} "
            f"sheared15={per_radius['15']['n_sheared_sources']:,}", flush=True,
        )
    with open(args.output, "x", encoding="utf-8") as handle:
        json.dump({
            "design": "independent stable spin-2 direction per anchor neighbourhood",
            "cases": args.cases, "g": args.g, "case_summaries": summaries,
        }, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print("ANCHORBLEND_RANDOM_CATALOGUE_AUDIT_DONE", flush=True)


if __name__ == "__main__":
    main()
