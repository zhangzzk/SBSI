"""Clone anchorblend inputs with shear confined to a radius around anchors.

The original anchor experiment shears every non-anchor in the field, so its
direct truth includes any response from sources beyond BlendEMU's 10 arcsec
aperture.  This controlled arm keeps the exact galaxies, morphologies,
positions, anchors, and case IDs, but sets shear to zero outside ``radius``.
It therefore measures the aperture-matched coherent-neighbour response without
using constgold or changing the anchor population.
"""
from __future__ import annotations

import argparse
import os
import shutil

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from sbs_shear.anchorblend import assert_anchor_spacing, sparse_anchor_mask


def label(g: float) -> str:
    return str(float(g))


def guarded_mask(frame: pd.DataFrame, anchor: np.ndarray, radius_arcsec: float) -> tuple[np.ndarray, np.ndarray]:
    local, distance, _ = guarded_assignment(frame, anchor, radius_arcsec)
    return local, distance


def guarded_assignment(
    frame: pd.DataFrame, anchor: np.ndarray, radius_arcsec: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return local-source mask, nearest-anchor distance and anchor owner row."""
    dec0 = float(np.median(frame["DEC"]))
    cosdec = np.cos(np.deg2rad(dec0))
    xy = np.column_stack([
        frame["RA"].to_numpy(float) * cosdec * 3600.0,
        frame["DEC"].to_numpy(float) * 3600.0,
    ])
    anchor_rows = np.flatnonzero(anchor)
    if not len(anchor_rows):
        raise RuntimeError("guarded assignment has no anchors")
    distance, owner_relative = cKDTree(xy[anchor]).query(xy, k=1)
    owner = anchor_rows[owner_relative]
    local = (distance <= float(radius_arcsec)) & (~anchor)
    return local, distance, owner


def stable_anchor_directions(
    case: int, anchor_ids: np.ndarray, seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic independent spin-2 unit vectors keyed by case/anchor ID."""
    ids = np.asarray(anchor_ids, np.uint64)
    mask64 = (1 << 64) - 1
    z = ids + np.uint64(seed) + np.uint64(
        ((int(case) + 1) * 0x9E3779B185EBCA87) & mask64
    )
    z ^= z >> np.uint64(30)
    z *= np.uint64(0xBF58476D1CE4E5B9)
    z ^= z >> np.uint64(27)
    z *= np.uint64(0x94D049BB133111EB)
    z ^= z >> np.uint64(31)
    unit = (z >> np.uint64(11)).astype(np.float64) / float(1 << 53)
    angle = 2.0 * np.pi * unit
    return np.cos(angle), np.sin(angle)


def sparse_anchor_layer(frame: pd.DataFrame, candidate: np.ndarray,
                        min_separation: float, layer: int) -> np.ndarray:
    """Return one deterministic disjoint greedy layer of candidate anchors."""
    if int(layer) != layer or layer < 0:
        raise ValueError("anchor layer must be a non-negative integer")
    remaining = np.asarray(candidate, bool).copy()
    for index in range(int(layer) + 1):
        chosen = sparse_anchor_mask(
            frame["RA"], frame["DEC"], remaining,
            min_separation_arcsec=min_separation,
        )
        if index == int(layer):
            return chosen
        remaining &= ~chosen
        if not remaining.any():
            return np.zeros(len(frame), dtype=bool)
    raise AssertionError("unreachable")


def materialize_simulation_support(
    source: str,
    output: str,
    cases: list[int],
    shears: tuple[float, ...],
) -> None:
    """Copy noise/config inputs while redirecting every source-root path.

    The guarded catalogues deliberately reuse the original render setup.  The
    MultiBand_ImSim wrapper expects one INI per case/shear in the output root;
    copying only the feather catalogues makes its subprocess fail before any
    images are produced.
    """
    source_root = os.path.abspath(source).rstrip(os.sep)
    output_root = os.path.abspath(output).rstrip(os.sep)
    noise_source = os.path.join(source_root, "noise.csv")
    noise_output = os.path.join(output_root, "noise.csv")
    if not os.path.exists(noise_source):
        raise FileNotFoundError(noise_source)
    if not os.path.exists(noise_output):
        shutil.copy2(noise_source, noise_output)
    else:
        with open(noise_source, "rb") as source_handle, open(noise_output, "rb") as output_handle:
            if source_handle.read() != output_handle.read():
                raise RuntimeError("existing output noise.csv differs from source")

    for case in cases:
        for shear in shears:
            name = f"sim_config_case{case}_{label(shear)}.ini"
            source_path = os.path.join(source_root, name)
            output_path = os.path.join(output_root, name)
            if not os.path.exists(source_path):
                raise FileNotFoundError(source_path)
            with open(source_path, encoding="utf-8") as handle:
                source_text = handle.read()
            if source_root not in source_text:
                raise RuntimeError(f"{source_path}: source root is absent")
            output_text = source_text.replace(source_root, output_root)
            expected_catalogue = os.path.join(
                output_root, f"gals{case}_{label(shear)}.feather"
            )
            if expected_catalogue not in output_text or "[Paths]" not in output_text:
                raise RuntimeError(f"{source_path}: redirected config failed validation")
            if os.path.exists(output_path):
                with open(output_path, encoding="utf-8") as handle:
                    if handle.read() != output_text:
                        raise RuntimeError(f"REFUSING differing existing config {output_path}")
            else:
                with open(output_path, "x", encoding="utf-8") as handle:
                    handle.write(output_text)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--cases", type=int, nargs="+", required=True)
    ap.add_argument("--g", type=float, default=0.05)
    ap.add_argument("--radius", type=float, default=10.0)
    ap.add_argument("--min-separation", type=float, default=20.0)
    ap.add_argument(
        "--random-anchor-directions", action="store_true",
        help="assign one stable independent spin-2 direction per anchor neighbourhood",
    )
    ap.add_argument(
        "--anchor-layer", type=int, default=0,
        help="zero-based disjoint greedy layer of the source anchor manifest",
    )
    ap.add_argument("--direction-seed", type=int, default=25876)
    ap.add_argument(
        "--support-only", action="store_true",
        help="only materialize redirected simulation INIs and noise.csv",
    )
    args = ap.parse_args()
    if os.path.abspath(args.source) == os.path.abspath(args.output):
        raise ValueError("source and output roots must differ")
    os.makedirs(args.output, exist_ok=True)

    shears = (float(args.g), -float(args.g))
    if args.support_only:
        materialize_simulation_support(args.source, args.output, args.cases, shears)
        print("ANCHORBLEND_GUARDED_SUPPORT_DONE", flush=True)
        return

    for case in args.cases:
        source_manifest = os.path.join(args.source, f"anchors_case{case}.feather")
        output_manifest = os.path.join(args.output, f"anchors_case{case}.feather")
        outputs = [
            os.path.join(args.output, f"gals{case}_{label(sign)}.feather")
            for sign in (args.g, -args.g)
        ]
        if os.path.exists(output_manifest) or any(os.path.exists(path) for path in outputs):
            raise SystemExit(f"REFUSING existing guarded output for case {case}")
        for sign in (args.g, -args.g):
            if os.path.exists(os.path.join(args.output, f"case{case}_{label(sign)}")):
                raise SystemExit(f"REFUSING existing render tree for case {case}, sign {sign}")

        manifest = pd.read_feather(source_manifest)
        legs = [
            pd.read_feather(os.path.join(args.source, f"gals{case}_{label(sign)}.feather"))
            for sign in (args.g, -args.g)
        ]
        if list(legs[0].columns) != list(legs[1].columns):
            raise RuntimeError(f"case {case}: source leg schemas differ")
        for column in legs[0].columns.difference(["g1", "g2"], sort=False):
            if not legs[0][column].equals(legs[1][column]):
                raise RuntimeError(f"case {case}: source latent column {column} differs")
        candidate_anchor = legs[0]["index"].isin(manifest["index"]).to_numpy()
        if candidate_anchor.sum() != len(manifest):
            raise RuntimeError(f"case {case}: manifest does not match source rows")
        if args.random_anchor_directions:
            if args.min_separation <= 2.0 * args.radius:
                raise RuntimeError(
                    "random anchor neighbourhoods require min-separation > 2*radius"
                )
            anchor = sparse_anchor_layer(
                legs[0], candidate_anchor, args.min_separation, args.anchor_layer,
            )
            if not anchor.any():
                raise RuntimeError(f"case {case}: anchor layer {args.anchor_layer} is empty")
        else:
            anchor = candidate_anchor
        minimum = assert_anchor_spacing(
            legs[0]["RA"], legs[0]["DEC"], anchor,
            min_separation_arcsec=args.min_separation,
        )
        local, distance, owner = guarded_assignment(legs[0], anchor, args.radius)
        if np.any(local & anchor):
            raise RuntimeError("anchor included in local-neighbour mask")
        # Strict separation prevents one source from belonging to overlapping
        # anchor apertures.  Count is logged; the mask itself needs no owner ID.
        anchor_xy_count = int(anchor.sum())
        anchor_rows = np.flatnonzero(anchor)
        u1 = u2 = None
        if args.random_anchor_directions:
            u1, u2 = stable_anchor_directions(
                case, legs[0].loc[anchor, "index"].to_numpy(np.int64), args.direction_seed,
            )
            by_row_u1 = np.zeros(len(legs[0]), dtype=float)
            by_row_u2 = np.zeros(len(legs[0]), dtype=float)
            by_row_u1[anchor_rows] = u1
            by_row_u2[anchor_rows] = u2
        for frame, sign, path in zip(legs, (+1.0, -1.0), outputs):
            frame.loc[:, ["g1", "g2"]] = 0.0
            if args.random_anchor_directions:
                frame.loc[local, "g1"] = (
                    sign * float(args.g) * by_row_u1[owner[local]]
                )
                frame.loc[local, "g2"] = (
                    sign * float(args.g) * by_row_u2[owner[local]]
                )
            else:
                frame.loc[local, "g1"] = sign * float(args.g)
            if np.any(frame.loc[anchor, ["g1", "g2"]].to_numpy(float)):
                raise RuntimeError(f"case {case}: anchor retained shear")
            local_norm = np.hypot(
                frame.loc[local, "g1"].to_numpy(float),
                frame.loc[local, "g2"].to_numpy(float),
            )
            if len(local_norm) and np.max(np.abs(local_norm - args.g)) > 1e-12:
                raise RuntimeError(f"case {case}: guarded-neighbour shear magnitude mismatch")
            frame.to_feather(path)
        output_manifest_frame = legs[0].loc[anchor, ["index", "r", "Re"]].copy()
        output_manifest_frame["anchor_layer"] = int(args.anchor_layer)
        if args.random_anchor_directions:
            output_manifest_frame["u1"] = u1
            output_manifest_frame["u2"] = u2
            output_manifest_frame["direction_seed"] = int(args.direction_seed)
        output_manifest_frame.assign(
            guard_radius_arcsec=float(args.radius),
            local_sheared_sources=int(local.sum()),
        ).to_feather(output_manifest)
        print(
            f"case {case}: anchors={anchor_xy_count:,} local-sheared={local.sum():,} "
            f"outside-unsheared={(~local & ~anchor).sum():,} min-anchor-sep={minimum:.3f}" 
            f" max-local-distance={distance[local].max():.6f}", flush=True,
        )
    materialize_simulation_support(args.source, args.output, args.cases, shears)
    print("ANCHORBLEND_GUARDED_PREP_DONE", flush=True)


if __name__ == "__main__":
    main()
