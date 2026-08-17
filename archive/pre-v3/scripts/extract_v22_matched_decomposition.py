"""Extract same-anchor truth components from the matched V2.2 simulations.

This script consumes the pre-registered shear modes prepared by
``prepare_v22_matched_decomposition.py``.  It intersects detections across every
mode/sign before forming any response, so all terms use exactly the same anchors.
It writes per-anchor truth responses, the exact deployed V2.2 emulator sum, the
flow-conditioning frame, and absolute full-scene neighbour flux in 0--1, 1--3 and
3--10 arcsec shells.  Pair truth can be supplied either by stochastic ``pairN``
Hutchinson probes or by variance-reduced ``rankN`` modes that shear at most one
deployed neighbour per anchor.  The V2.2 flow itself is scored separately on a GPU.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

SBSI_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SBSI_ROOT not in sys.path:
    sys.path.insert(0, SBSI_ROOT)
BLENDEMU_ROOT = "/home/z/Zekang.Zhang/blendemu"
if BLENDEMU_ROOT not in sys.path:
    sys.path.insert(0, BLENDEMU_ROOT)

from blendemu.response import retrieve_constant_shear  # noqa: E402
from sbs_shear.coordinates import ellipticity_from_axis_ratio_angle  # noqa: E402


TILE = "tile180.0_-0.5"
ZERO_POINT = 30.0
PIXEL_RMS, PIXEL_SIZE, PSF_FWHM, MOFFAT_BETA = 0.312, 0.2, 0.73, 2.224


def aperture_rms() -> float:
    factor = np.sqrt(
        (2 ** (1 / (MOFFAT_BETA - 1)) - 1)
        / (2 ** (1 / MOFFAT_BETA) - 1)
    ) / 2
    psf_size = PSF_FWHM * factor
    return float(PIXEL_RMS * (psf_size / PIXEL_SIZE) ** 2 * np.pi)


def label(g: float) -> str:
    return str(float(g))


def read_anchor_leg(base: str, case: int, sign: float, anchors: np.ndarray) -> pd.DataFrame:
    d = retrieve_constant_shear(
        case, label(sign), r_max=10.0, r_min=0.0, k=20,
        data_path=base, shape_suffix="_all", include_isolated=True,
        include_measured=True, measured_columns=["MAG_AUTO", "FLUX_RADIUS"],
    )
    d = d[d["input_index"].isin(anchors)].copy()
    # A successful catalogue match is not sufficient: ngmix can still return a
    # non-finite shape for a tiny number of detections.  Since every response
    # term must use the identical anchor set, remove such rows before the
    # all-mode/sign inner intersection rather than letting NaNs leak into only
    # selected terms downstream.
    d = d[np.isfinite(d["measured_e1"]) & np.isfinite(d["measured_e2"])].copy()
    if d["input_index"].duplicated().any():
        raise RuntimeError(f"case {case} sign {sign:+g}: duplicate anchor detection")
    keep = [
        "input_index", "measured_e1", "measured_e2",
        "measured_mag_auto", "measured_flux_radius",
        "r_input_p", "Re_input_p", "sersic_n_input_p",
        "axis_ratio_input_p", "position_angle_input_p",
    ]
    return d[keep]


def full_scene_features(base: str, case: int, sign: float,
                        anchors: np.ndarray) -> pd.DataFrame:
    path = os.path.join(base, f"gals{case}_{label(sign)}.feather")
    field = pd.read_feather(path)
    by_id = field.set_index("index", verify_integrity=True)
    missing = np.setdiff1d(anchors, by_id.index.to_numpy(np.int64))
    if len(missing):
        raise RuntimeError(f"case {case}: manifest anchors missing from generated scene")
    anchor = by_id.loc[anchors]
    dec0 = float(np.median(field["DEC"]))
    cosdec = np.cos(np.deg2rad(dec0))
    xy = np.column_stack([
        field["RA"].to_numpy(float) * cosdec * 3600.0,
        field["DEC"].to_numpy(float) * 3600.0,
    ])
    axy = np.column_stack([
        anchor["RA"].to_numpy(float) * cosdec * 3600.0,
        anchor["DEC"].to_numpy(float) * 3600.0,
    ])
    flux = np.power(10.0, -0.4 * (field["r"].to_numpy(float) - ZERO_POINT))
    id_array = field["index"].to_numpy(np.int64)
    tree = cKDTree(xy)
    around = tree.query_ball_point(axy, r=10.0, workers=-1)
    shell = np.zeros((len(anchors), 3), dtype=np.float64)
    crowd_near = np.zeros(len(anchors), dtype=np.float64)
    crowd_far = np.zeros(len(anchors), dtype=np.float64)
    crowd_max = np.zeros(len(anchors), dtype=np.float64)
    for i, rows in enumerate(around):
        rows = np.asarray(rows, dtype=np.int64)
        rows = rows[id_array[rows] != anchors[i]]
        if not len(rows):
            continue
        distance = np.hypot(xy[rows, 0] - axy[i, 0], xy[rows, 1] - axy[i, 1])
        f = flux[rows]
        shell[i, 0] = f[(distance >= 0.0) & (distance < 1.0)].sum()
        shell[i, 1] = f[(distance >= 1.0) & (distance < 3.0)].sum()
        shell[i, 2] = f[(distance >= 3.0) & (distance <= 10.0)].sum()
        within7 = distance < 7.0
        crowd_near[i] = f[distance < 3.0].sum()
        crowd_far[i] = f[(distance >= 3.0) & within7].sum()
        crowd_max[i] = f[within7].max(initial=0.0)

    ar = aperture_rms()
    e1, e2 = ellipticity_from_axis_ratio_angle(
        anchor["axis_ratio"].to_numpy(float),
        anchor["position_angle"].to_numpy(float),
    )
    return pd.DataFrame({
        "input_index": anchors,
        "r_input_p": anchor["r"].to_numpy(float),
        "Re_input_p": anchor["Re"].to_numpy(float),
        "sersic_n_input_p": anchor["sersic_n"].to_numpy(float),
        "e1_input_rot0_p": e1,
        "e2_input_rot0_p": e2,
        "flux_abs_near_0_1": shell[:, 0],
        "flux_abs_mid_1_3": shell[:, 1],
        "flux_abs_far_3_10": shell[:, 2],
        "logflux_abs_near_0_1": np.log10(1.0 + shell[:, 0]),
        "logflux_abs_mid_1_3": np.log10(1.0 + shell[:, 1]),
        "logflux_abs_far_3_10": np.log10(1.0 + shell[:, 2]),
        "nbr_flux_near": np.log10(1.0 + crowd_near / ar),
        "nbr_flux_far": np.log10(1.0 + crowd_far / ar),
        "nbr_flux_max": np.log10(1.0 + crowd_max / ar),
    })


def response_columns(joined: pd.DataFrame, mode: str, g: float) -> tuple[np.ndarray, np.ndarray]:
    de1 = joined[f"measured_e1_{mode}_plus"].to_numpy(float) \
          - joined[f"measured_e1_{mode}_minus"].to_numpy(float)
    de2 = joined[f"measured_e2_{mode}_plus"].to_numpy(float) \
          - joined[f"measured_e2_{mode}_minus"].to_numpy(float)
    return de1 / (2.0 * g), de2 / (2.0 * g)


def numbered_modes(bases: dict[str, str], prefix: str) -> list[str]:
    modes = [mode for mode in bases if mode.startswith(prefix)]
    try:
        modes.sort(key=lambda mode: int(mode[len(prefix):]))
    except ValueError as exc:
        raise ValueError(f"{prefix} modes must end in integers: {modes}") from exc
    for number, mode in enumerate(modes):
        if mode != f"{prefix}{number}":
            raise ValueError(
                f"{prefix} modes must be contiguous {prefix}0..{prefix}N, got {modes}"
            )
    return modes


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", action="append", required=True,
                    help="MODE=PATH, identical to the preparation command")
    ap.add_argument("--manifest-dir", required=True)
    ap.add_argument("--cases", type=int, nargs="+", required=True)
    ap.add_argument("--g", type=float, default=0.05)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    bases: dict[str, str] = {}
    for item in args.base:
        mode, sep, path = item.partition("=")
        if not sep:
            raise ValueError(f"--base must be MODE=PATH, got {item!r}")
        bases[mode] = path
    pair_modes = numbered_modes(bases, "pair")
    rank_modes = numbered_modes(bases, "rank")
    if pair_modes and rank_modes:
        raise ValueError("use either pairN probes or rankN exact modes, not both")
    component_modes = pair_modes or rank_modes
    if not {"total", "self", "neighbour"}.issubset(bases) or not component_modes:
        raise ValueError(f"incomplete modes: {sorted(bases)}")
    if os.path.exists(args.output):
        raise FileExistsError(f"refusing to overwrite {args.output}")

    manifest_dir = Path(args.manifest_dir)
    parts = []
    case_summary = []
    for case in args.cases:
        anchors = pd.read_feather(manifest_dir / f"anchors_case{case}.feather")
        anchor_ids = anchors["index"].to_numpy(np.int64)
        pairs = pd.read_feather(manifest_dir / f"pairs_case{case}.feather")
        joined = pd.DataFrame({"input_index": anchor_ids})
        for mode, base in bases.items():
            for sign_name, sign in (("plus", +args.g), ("minus", -args.g)):
                leg = read_anchor_leg(base, case, sign, anchor_ids)
                rename = {
                    c: f"{c}_{mode}_{sign_name}" for c in leg.columns
                    if c != "input_index"
                }
                joined = joined.merge(
                    leg.rename(columns=rename), on="input_index", how="inner",
                    validate="one_to_one",
                )
        coverage = len(joined) / len(anchor_ids)
        if coverage < 0.70:
            raise RuntimeError(f"case {case}: all-leg anchor coverage {coverage:.2%} < 70%")

        scene = full_scene_features(bases["total"], case, +args.g, anchor_ids)
        joined = joined.merge(scene, on="input_index", how="left", validate="one_to_one")
        if joined[["logflux_abs_near_0_1", "logflux_abs_mid_1_3",
                   "logflux_abs_far_3_10"]].isna().any().any():
            raise RuntimeError(f"case {case}: scene-feature join failed")

        # Use the shear-even average of the self +/- legs for the V2.2 measured-primary
        # conditioners, matching the constgold evaluation convention.
        joined["measured_mag_auto"] = 0.5 * (
            joined["measured_mag_auto_self_plus"]
            + joined["measured_mag_auto_self_minus"]
        )
        joined["measured_flux_radius"] = 0.5 * (
            joined["measured_flux_radius_self_plus"]
            + joined["measured_flux_radius_self_minus"]
        )

        total1, total2 = response_columns(joined, "total", args.g)
        self1, self2 = response_columns(joined, "self", args.g)
        neighbour1, neighbour2 = response_columns(joined, "neighbour", args.g)
        joined["R_total_truth"] = total1
        joined["R_total_null"] = total2
        joined["R_self_truth"] = self1
        joined["R_self_null"] = self2
        joined["R_neighbour_coherent_truth"] = neighbour1
        joined["R_neighbour_coherent_null"] = neighbour2

        pair_model = pairs.groupby("anchor_index", sort=False)["response"].sum()
        joined["R_blend_model"] = joined["input_index"].map(pair_model).fillna(0.0)
        pair_estimates = []
        pair_nulls = []
        if pair_modes:
            for mode in pair_modes:
                de1, de2 = response_columns(joined, mode, args.g)
                sums = pairs.groupby("anchor_index", sort=False)[
                    [f"u1_{mode}", f"u2_{mode}"]
                ].sum()
                su1 = joined["input_index"].map(sums[f"u1_{mode}"]).fillna(0.0).to_numpy(float)
                su2 = joined["input_index"].map(sums[f"u2_{mode}"]).fillna(0.0).to_numpy(float)
                estimate = de1 * su1 + de2 * su2
                null = -de1 * su2 + de2 * su1
                joined[f"R_pair_truth_{mode}"] = estimate
                joined[f"R_pair_null_{mode}"] = null
                pair_estimates.append(estimate)
                pair_nulls.append(null)
            joined["R_pair_truth"] = np.mean(pair_estimates, axis=0)
            joined["R_pair_null"] = np.mean(pair_nulls, axis=0)
            pair_estimator = "mean of all-neighbour Hutchinson probes"
        else:
            required_rank_columns = {"rank", "u1", "u2"}
            missing_rank_columns = required_rank_columns - set(pairs.columns)
            if missing_rank_columns:
                raise KeyError(
                    f"case {case}: rank manifest lacks {sorted(missing_rank_columns)}"
                )
            ranks = pairs["rank"].to_numpy(int)
            if len(ranks) and (ranks.min() < 0 or ranks.max() >= len(rank_modes)):
                raise RuntimeError(
                    f"case {case}: manifest ranks outside 0..{len(rank_modes) - 1}"
                )
            if pairs.duplicated(["anchor_index", "rank"]).any():
                raise RuntimeError(f"case {case}: more than one neighbour in an anchor rank")
            for rank, mode in enumerate(rank_modes):
                de1, de2 = response_columns(joined, mode, args.g)
                one = pairs.loc[pairs["rank"] == rank].set_index(
                    "anchor_index", verify_integrity=True,
                )
                u1 = joined["input_index"].map(one["u1"]).to_numpy(float)
                u2 = joined["input_index"].map(one["u2"]).to_numpy(float)
                present = np.isfinite(u1) & np.isfinite(u2)
                estimate = np.zeros(len(joined), dtype=float)
                null = np.zeros(len(joined), dtype=float)
                estimate[present] = (
                    de1[present] * u1[present] + de2[present] * u2[present]
                )
                null[present] = (
                    -de1[present] * u2[present] + de2[present] * u1[present]
                )
                joined[f"R_pair_truth_{mode}"] = estimate
                joined[f"R_pair_null_{mode}"] = null
                pair_estimates.append(estimate)
                pair_nulls.append(null)
            joined["R_pair_truth"] = np.sum(pair_estimates, axis=0)
            joined["R_pair_null"] = np.sum(pair_nulls, axis=0)
            pair_estimator = "sum of one-neighbour-per-rank random projections"
        joined["Delta_emu"] = joined["R_blend_model"] - joined["R_pair_truth"]
        joined["Delta_add_self_neighbour"] = (
            joined["R_total_truth"] - joined["R_self_truth"]
            - joined["R_neighbour_coherent_truth"]
        )
        joined["Delta_add_neighbours"] = (
            joined["R_neighbour_coherent_truth"] - joined["R_pair_truth"]
        )
        joined["Delta_add"] = (
            joined["R_total_truth"] - joined["R_self_truth"]
            - joined["R_pair_truth"]
        )
        joined.insert(0, "case", case)
        parts.append(joined)
        case_summary.append({
            "case": case, "n_manifest_anchors": len(anchor_ids),
            "n_common_anchors": len(joined), "coverage": coverage,
            "R_total_truth": float(joined["R_total_truth"].mean()),
            "R_self_truth": float(joined["R_self_truth"].mean()),
            "R_neighbour_coherent_truth": float(joined["R_neighbour_coherent_truth"].mean()),
            "R_pair_truth": float(joined["R_pair_truth"].mean()),
            "R_blend_model": float(joined["R_blend_model"].mean()),
            "Delta_emu": float(joined["Delta_emu"].mean()),
            "Delta_add": float(joined["Delta_add"].mean()),
            "R_pair_null": float(joined["R_pair_null"].mean()),
        })
        print(
            f"case {case}: common={len(joined):,}/{len(anchor_ids):,} ({coverage:.1%}) "
            f"Rtot={joined['R_total_truth'].mean():+.5f} "
            f"Rself={joined['R_self_truth'].mean():+.5f} "
            f"Rneigh={joined['R_neighbour_coherent_truth'].mean():+.5f} "
            f"Rpair={joined['R_pair_truth'].mean():+.5f} "
            f"emu={joined['R_blend_model'].mean():+.5f} "
            f"Dadd={joined['Delta_add'].mean():+.5f} null={joined['R_pair_null'].mean():+.5f}",
            flush=True,
        )

    out = pd.concat(parts, ignore_index=True)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_feather(args.output)
    sidecar = os.path.splitext(args.output)[0] + ".json"
    with open(sidecar, "x", encoding="utf-8") as handle:
        json.dump({
            "g": args.g, "cases": args.cases, "modes": sorted(bases),
            "pair_truth_estimator": pair_estimator,
            "n_rows": len(out), "n_cases": out["case"].nunique(),
            "common_detection_policy": "inner intersection across every mode and sign",
            "flux_definition": (
                "absolute intrinsic flux in simulation counts, zero point 30, all rendered "
                "sources except anchor, shells [0,1),[1,3),[3,10] arcsec"
            ),
            "case_summary": case_summary,
        }, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"wrote {args.output}: {len(out):,} common anchors")
    print("V22_MATCHED_DECOMP_EXTRACT_DONE", flush=True)


if __name__ == "__main__":
    main()
