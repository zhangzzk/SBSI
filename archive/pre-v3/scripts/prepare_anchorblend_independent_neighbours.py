"""Prepare anchor scenes with independent shear directions per local source."""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from scripts.prepare_anchorblend_guarded_catalogues import (
    guarded_assignment, materialize_simulation_support, stable_anchor_directions,
)


def label(value: float) -> str:
    return str(float(value))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cases", type=int, nargs="+", required=True)
    parser.add_argument("--g", type=float, default=0.05)
    parser.add_argument("--radius", type=float, default=10.0)
    parser.add_argument("--direction-seed", type=int, default=20260812)
    parser.add_argument(
        "--active-mode", choices=["all", "upper_index_half"], default="all",
        help="which local non-anchor sources receive independent shear",
    )
    args = parser.parse_args()
    if os.path.abspath(args.source) == os.path.abspath(args.output):
        raise ValueError("source and output roots must differ")
    os.makedirs(args.output, exist_ok=True)
    shears = (float(args.g), -float(args.g))

    for case in args.cases:
        source_manifest = os.path.join(args.source, f"anchors_case{case}.feather")
        output_manifest = os.path.join(args.output, f"anchors_case{case}.feather")
        outputs = [
            os.path.join(args.output, f"gals{case}_{label(sign)}.feather")
            for sign in shears
        ]
        if os.path.exists(output_manifest) or any(os.path.exists(path) for path in outputs):
            raise FileExistsError(f"refusing existing independent-neighbour case {case}")
        manifest = pd.read_feather(source_manifest)
        legs = [
            pd.read_feather(os.path.join(args.source, f"gals{case}_{label(sign)}.feather"))
            for sign in shears
        ]
        if list(legs[0].columns) != list(legs[1].columns):
            raise RuntimeError(f"case {case}: source leg schemas differ")
        for column in legs[0].columns.difference(["g1", "g2"], sort=False):
            if not legs[0][column].equals(legs[1][column]):
                raise RuntimeError(f"case {case}: latent column {column} differs")
        anchor = legs[0]["index"].isin(manifest["index"]).to_numpy()
        if int(anchor.sum()) != len(manifest):
            raise RuntimeError(f"case {case}: manifest IDs do not match source")
        local, distance, _ = guarded_assignment(legs[0], anchor, args.radius)
        if args.active_mode == "all":
            active = local.copy()
        else:
            split = len(legs[0]) // 2
            active = local & (legs[0]["index"].to_numpy(np.int64) >= split)
        source_ids = legs[0].loc[active, "index"].to_numpy(np.int64)
        u1, u2 = stable_anchor_directions(case, source_ids, args.direction_seed)
        for frame, sign, output in zip(legs, (+1.0, -1.0), outputs):
            frame.loc[:, ["g1", "g2"]] = 0.0
            frame.loc[active, "g1"] = sign * args.g * u1
            frame.loc[active, "g2"] = sign * args.g * u2
            if np.any(frame.loc[anchor, ["g1", "g2"]].to_numpy(float)):
                raise RuntimeError(f"case {case}: anchor retained shear")
            norm = np.hypot(frame.loc[active, "g1"], frame.loc[active, "g2"])
            if len(norm) and not np.allclose(norm, args.g, rtol=0.0, atol=1e-12):
                raise RuntimeError(f"case {case}: local shear magnitude drift")
            frame.to_feather(output)
        saved = manifest.copy()
        saved["direction_seed"] = int(args.direction_seed)
        saved["guard_radius_arcsec"] = float(args.radius)
        saved["local_sheared_sources"] = int(local.sum())
        saved["active_mode"] = args.active_mode
        saved["active_sheared_sources"] = int(active.sum())
        saved.to_feather(output_manifest)
        print(
            f"case {case}: anchors={anchor.sum():,} local={local.sum():,} "
            f"active={active.sum():,} mode={args.active_mode} "
            f"outside={(~local & ~anchor).sum():,} maxdist={distance[local].max():.6f}",
            flush=True,
        )
    materialize_simulation_support(args.source, args.output, args.cases, shears)
    print("ANCHORBLEND_INDEPENDENT_NEIGHBOUR_PREP_DONE", flush=True)


if __name__ == "__main__":
    main()
