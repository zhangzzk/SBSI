"""Prepare a sparse constgold q3 response-decomposition pilot.

This is an evaluation-only diagnostic on the frozen V2.2 constgold domain.  It
selects unique primaries in the pre-existing high-positive-residual blendness
bin, keeps every source in a local scene around each selected primary, and
writes four matched antithetic treatments:

``total``
    Shear every retained source.
``self``
    Shear only the selected primary.
``deployed``
    Shear only neighbours selected by the deployed V2.2 emulator.
``other``
    Shear retained local sources omitted by the deployed neighbour list.

The local scenes are disjoint by construction.  Hence every retained source has
one unambiguous owner and the four later finite differences can be added without
double-counting a source.  The input galaxy population, positions, PSF/noise
configuration, and stable IDs are inherited from the original constgold cases.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.feather as pf
from scipy.spatial import cKDTree

SBSI_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SBSI_ROOT not in sys.path:
    sys.path.insert(0, SBSI_ROOT)
BLENDEMU_ROOT = "/home/z/Zekang.Zhang/blendemu"
if BLENDEMU_ROOT not in sys.path:
    sys.path.insert(0, BLENDEMU_ROOT)

from blendemu.catalog import write_config_file, write_noise_file_from_csv  # noqa: E402
from blendemu.inference import BlendingPredictor  # noqa: E402
from sbs_shear.anchorblend import assert_anchor_spacing, sparse_anchor_mask  # noqa: E402
from scripts.prepare_v22_matched_decomposition import (  # noqa: E402
    COND, MODEL_DIR, assert_same_latents, generated_path, input_path, pair_id_columns,
)


Q3_LOW = 0.1025923490524292
Q3_HIGH = 0.31271257996559143
MODES = ("total", "self", "deployed", "other")
IDENTITY_COLUMNS = [
    "index", "cata_idx", "RA", "DEC", "position_angle", "redshift", "Re",
    "axis_ratio", "sersic_n", "r", "shear_component_convention",
]


def parse_bases(items: list[str]) -> dict[str, str]:
    bases: dict[str, str] = {}
    for item in items:
        mode, sep, path = item.partition("=")
        if not sep or not mode or not path:
            raise ValueError(f"--base must be MODE=PATH, got {item!r}")
        if mode in bases:
            raise ValueError(f"duplicate mode {mode!r}")
        bases[mode] = path
    if set(bases) != set(MODES):
        raise ValueError(f"expected exactly {list(MODES)}, got {sorted(bases)}")
    return bases


def original_generated_path(base: str, case: int, sign: float) -> str:
    return os.path.join(base, f"gals{case}_{str(float(sign))}.feather")


def candidate_mask(reference: pd.DataFrame, lookup_case: pd.DataFrame) -> np.ndarray:
    if lookup_case.duplicated("input_index").any():
        raise RuntimeError("lookup has duplicate (case,input_index) rows")
    rb = pd.Series(
        lookup_case["R_blend"].to_numpy(float),
        index=lookup_case["input_index"].to_numpy(np.int64),
    ).reindex(reference["index"].to_numpy(np.int64)).to_numpy(float)
    mag = reference["r"].to_numpy(float)
    size = reference["Re"].to_numpy(float)
    return (
        np.isfinite(rb) & (rb >= Q3_LOW) & (rb < Q3_HIGH)
        & np.isfinite(mag) & (mag > 18.0) & (mag < 25.8)
        & np.isfinite(size) & (size > 0.5) & (size < 1.5)
    )


def scene_ownership(reference: pd.DataFrame, anchors: np.ndarray,
                    radius_arcsec: float) -> pd.DataFrame:
    by_id = reference.set_index("index", verify_integrity=True)
    anchor = by_id.loc[anchors]
    cosdec = np.cos(np.deg2rad(float(np.median(reference["DEC"]))))
    xy = np.column_stack([
        reference["RA"].to_numpy(float) * cosdec * 3600.0,
        reference["DEC"].to_numpy(float) * 3600.0,
    ])
    axy = np.column_stack([
        anchor["RA"].to_numpy(float) * cosdec * 3600.0,
        anchor["DEC"].to_numpy(float) * 3600.0,
    ])
    tree = cKDTree(xy)
    around = tree.query_ball_point(axy, r=radius_arcsec, workers=-1)
    source_rows: list[np.ndarray] = []
    owners: list[np.ndarray] = []
    for anchor_id, rows in zip(anchors, around):
        rows = np.asarray(rows, dtype=np.int64)
        source_rows.append(rows)
        owners.append(np.full(len(rows), anchor_id, dtype=np.int64))
    rows = np.concatenate(source_rows)
    owner = np.concatenate(owners)
    ids = reference["index"].to_numpy(np.int64)[rows]
    if pd.Series(ids).duplicated().any():
        raise RuntimeError("local scenes overlap; increase minimum anchor separation")
    if not np.isin(anchors, ids).all():
        raise RuntimeError("an anchor is absent from its own local scene")
    return pd.DataFrame({
        "source_index": ids, "anchor_index": owner,
        "source_row": rows.astype(np.int64),
    })


def assign_treatment(local: pd.DataFrame, mode: str, sign: float, g: float) -> pd.DataFrame:
    out = local[IDENTITY_COLUMNS].copy()
    out["g1"] = 0.0
    out["g2"] = 0.0
    selected = out["index"].isin(
        local.loc[local["source_class"] == mode, "index"]
        if mode != "total" else out["index"]
    )
    out.loc[selected, "g1"] = (1.0 if sign > 0 else -1.0) * g
    # Keep the generated-catalogue schema/order used by MultiBand_ImSim.
    columns = [c for c in local.attrs["reference_columns"] if c in out]
    return out[columns]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", action="append", required=True, help="MODE=PATH")
    ap.add_argument("--original-base", required=True)
    ap.add_argument("--lookup", required=True)
    ap.add_argument("--manifest-dir", required=True)
    ap.add_argument("--cases", type=int, nargs="+", required=True)
    ap.add_argument("--g", type=float, default=0.02)
    ap.add_argument("--scene-radius", type=float, default=15.0)
    ap.add_argument("--min-separation", type=float, default=30.01)
    ap.add_argument("--tag", default="lsst_r_extnbr_v22")
    ap.add_argument("--base-config", required=True)
    ap.add_argument("--noise-csv", required=True)
    ap.add_argument("--noise-band", default="LSST_r")
    ap.add_argument("--mag-cut", type=float, default=29.0)
    ap.add_argument("--pixel-scale", type=float, default=0.2)
    ap.add_argument("--lookup-replay-atol", type=float, default=2.0e-7,
                    help="maximum allowed per-anchor stored/replayed R_blend difference")
    ap.add_argument("--lookup-replay-mean-atol", type=float, default=1.0e-10,
                    help="maximum allowed mean absolute stored/replayed R_blend difference")
    args = ap.parse_args()
    if args.scene_radius <= 10.0:
        raise ValueError("scene radius must exceed the deployed 10 arcsec aperture")
    if args.min_separation <= 2.0 * args.scene_radius:
        raise ValueError("minimum separation must be strictly greater than 2*scene radius")

    bases = parse_bases(args.base)
    manifest = Path(args.manifest_dir)
    if manifest.exists():
        raise FileExistsError(f"refusing existing manifest {manifest}")
    for base in bases.values():
        if os.path.exists(base):
            raise FileExistsError(f"refusing existing simulation base {base}")
    manifest.mkdir(parents=True)
    for base in bases.values():
        Path(base).mkdir(parents=True)
        write_noise_file_from_csv(args.noise_csv, args.noise_band, base)

    lookup = pf.read_table(
        args.lookup, columns=["case", "input_index", "R_blend"],
    ).to_pandas()
    lookup = lookup[lookup["case"].isin(args.cases)].copy()
    if lookup.duplicated(["case", "input_index"]).any():
        raise RuntimeError("lookup has duplicate keys")

    predictor = BlendingPredictor.load(
        MODEL_DIR, tag=args.tag, conditions=COND, device="cpu",
    )
    cuts, rmax, k = predictor._select("regression")
    if not np.isclose(float(rmax), 10.0) or int(k) != 20:
        raise RuntimeError(f"expected deployed r_max=10/k=20, got {rmax}/{k}")

    summary = {
        "design": "constgold q3 disjoint local scenes; coherent subset shears",
        "evaluation_only": True,
        "blendness_bin": {"name": "q3", "low_inclusive": Q3_LOW,
                          "high_exclusive": Q3_HIGH},
        "cases": args.cases, "g": args.g,
        "scene_radius_arcsec": args.scene_radius,
        "min_anchor_separation_arcsec": args.min_separation,
        "tag": args.tag, "deployed_r_max_arcsec": float(rmax),
        "deployed_k": int(k), "regression_cuts": cuts,
        "lookup_replay_atol": args.lookup_replay_atol,
        "lookup_replay_mean_atol": args.lookup_replay_mean_atol,
        "modes": list(MODES), "per_case": [],
    }
    for case in args.cases:
        plus = pd.read_feather(original_generated_path(args.original_base, case, +args.g))
        minus = pd.read_feather(original_generated_path(args.original_base, case, -args.g))
        assert_same_latents(plus, minus, f"original constgold case {case} +/-")
        reference = plus.copy()
        candidates = candidate_mask(reference, lookup.loc[lookup["case"] == case])
        anchor_mask = sparse_anchor_mask(
            reference["RA"], reference["DEC"], candidates,
            min_separation_arcsec=args.min_separation,
        )
        minimum = assert_anchor_spacing(
            reference["RA"], reference["DEC"], anchor_mask,
            min_separation_arcsec=args.min_separation,
        )
        anchors = reference.loc[anchor_mask, "index"].to_numpy(np.int64)
        if len(anchors) < 50:
            raise RuntimeError(f"case {case}: only {len(anchors)} q3 sparse anchors")

        # The frozen lookup was built from this post-render input catalogue.  Use
        # that exact field for deployed pair registration: replaying on the
        # pre-render generated feather can differ at a 10-arcsec neighbour
        # boundary because of coordinate round-off.
        field = pd.read_feather(input_path(args.original_base, case, +args.g))
        field = field.rename(columns={c: c.replace("_input", "") for c in field})
        if field["index"].duplicated().any():
            raise RuntimeError(f"case {case}: duplicate stable ID in original input field")
        if not np.isin(anchors, field["index"].to_numpy(np.int64)).all():
            raise RuntimeError(f"case {case}: selected anchor missing from original input field")
        anchor_frame = field[field["index"].isin(anchors)].copy()
        pairs = predictor.predict_response(anchor_frame, field)
        pcol, scol = pair_id_columns(pairs)
        pairs = pairs.rename(columns={pcol: "anchor_index", scol: "secondary_index"})
        pairs[["anchor_index", "secondary_index"]] = pairs[
            ["anchor_index", "secondary_index"]
        ].astype(np.int64)
        if pairs.duplicated(["anchor_index", "secondary_index"]).any():
            raise RuntimeError(f"case {case}: duplicate deployed pair")
        if not pairs["anchor_index"].isin(anchors).all():
            raise RuntimeError(f"case {case}: predictor returned non-anchor primary")
        owner_count = pairs.groupby("secondary_index")["anchor_index"].nunique()
        if len(owner_count) and int(owner_count.max()) > 1:
            raise RuntimeError(f"case {case}: deployed neighbour belongs to multiple anchors")

        rb_pred = pairs.groupby("anchor_index")["response"].sum().reindex(
            anchors, fill_value=0.0,
        )
        rb_lookup = lookup.loc[lookup["case"] == case].set_index(
            "input_index", verify_integrity=True,
        )["R_blend"].reindex(anchors)
        if rb_lookup.isna().any():
            raise RuntimeError(f"case {case}: selected anchor missing from lookup")
        rb_diff = rb_pred.to_numpy(float) - rb_lookup.to_numpy(float)
        max_abs_rb_diff = float(np.max(np.abs(rb_diff)))
        mean_abs_rb_diff = float(np.mean(np.abs(rb_diff)))
        if (max_abs_rb_diff > args.lookup_replay_atol
                or mean_abs_rb_diff > args.lookup_replay_mean_atol):
            raise RuntimeError(
                f"case {case}: replayed/lookup R_blend mismatch "
                f"max={max_abs_rb_diff:.3e}, meanabs={mean_abs_rb_diff:.3e}"
            )

        ownership = scene_ownership(reference, anchors, args.scene_radius)
        deployed_ids = pairs["secondary_index"].drop_duplicates().to_numpy(np.int64)
        if not np.isin(deployed_ids, ownership["source_index"]).all():
            missing = int((~np.isin(deployed_ids, ownership["source_index"])).sum())
            raise RuntimeError(f"case {case}: {missing} deployed neighbours outside local scenes")
        ownership["source_class"] = "other"
        ownership.loc[ownership["source_index"].isin(deployed_ids), "source_class"] = "deployed"
        ownership.loc[ownership["source_index"].isin(anchors), "source_class"] = "self"
        if (ownership["source_class"] == "self").sum() != len(anchors):
            raise RuntimeError(f"case {case}: anchor ownership is not one-to-one")
        pair_owner = pairs.set_index("secondary_index")["anchor_index"]
        deployed_owner = ownership.loc[
            ownership["source_class"] == "deployed", ["source_index", "anchor_index"]
        ].set_index("source_index")["anchor_index"]
        if not deployed_owner.sort_index().equals(pair_owner.sort_index()):
            raise RuntimeError(f"case {case}: deployed source assigned to wrong scene owner")

        local = reference.iloc[ownership["source_row"].to_numpy(np.int64)].copy()
        local = local.merge(
            ownership[["source_index", "anchor_index", "source_class"]],
            left_on="index", right_on="source_index", how="left", validate="one_to_one",
        ).drop(columns="source_index")
        local.attrs["reference_columns"] = reference.columns.tolist()
        if not local[IDENTITY_COLUMNS].equals(
                reference.iloc[ownership["source_row"].to_numpy(np.int64)][IDENTITY_COLUMNS]
                .reset_index(drop=True)):
            raise RuntimeError(f"case {case}: local-scene latent identity changed")

        anchor_out = reference.loc[anchor_mask, ["index", "r", "Re"]].copy()
        anchor_out.insert(0, "case", case)
        anchor_out["R_blend_lookup"] = rb_lookup.to_numpy(float)
        anchor_out["R_blend_replayed"] = rb_pred.to_numpy(float)
        anchor_out.to_feather(manifest / f"anchors_case{case}.feather")
        pair_out = pairs[["anchor_index", "secondary_index", "distance", "response"]].copy()
        pair_out.insert(0, "case", case)
        pair_out.to_feather(manifest / f"pairs_case{case}.feather")
        ownership.insert(0, "case", case)
        ownership.drop(columns="source_row").to_feather(manifest / f"sources_case{case}.feather")

        for mode, base in bases.items():
            for sign in (+args.g, -args.g):
                out = assign_treatment(local, mode, sign, args.g)
                path = generated_path(base, case, sign)
                out.to_feather(path)
                write_config_file(
                    case, sign, path=base, base_config_path=args.base_config,
                    file_name_cat=path, mag_cut=args.mag_cut, pixel_scale=args.pixel_scale,
                )

        counts = ownership["source_class"].value_counts()
        pair_count = pairs.groupby("anchor_index").size()
        summary["per_case"].append({
            "case": case, "n_original_sources": len(reference),
            "n_q3_candidates": int(candidates.sum()), "n_anchors": len(anchors),
            "minimum_anchor_spacing_arcsec": minimum,
            "n_local_sources": len(ownership),
            "n_deployed_sources": int(counts.get("deployed", 0)),
            "n_other_sources": int(counts.get("other", 0)),
            "n_pairs": len(pairs),
            "mean_pairs_per_anchor": float(len(pairs) / len(anchors)),
            "max_pairs_per_anchor": int(pair_count.max()) if len(pair_count) else 0,
            "max_abs_lookup_replay_difference": max_abs_rb_diff,
            "mean_abs_lookup_replay_difference": mean_abs_rb_diff,
        })
        print(
            f"case {case}: q3={candidates.sum():,} anchors={len(anchors):,} "
            f"local={len(ownership):,} deployed={counts.get('deployed', 0):,} "
            f"other={counts.get('other', 0):,} pairs={len(pairs):,} "
            f"lookup-diff(max/meanabs)={max_abs_rb_diff:.2e}/{mean_abs_rb_diff:.2e}",
            flush=True,
        )

    with open(manifest / "design.json", "x", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"wrote constgold q3 design -> {manifest}")
    print("V22_CONSTGOLD_Q3_DECOMP_PREP_DONE", flush=True)


if __name__ == "__main__":
    main()
