"""Run matched coherent and per-neighbour noiseless toys for one anchor case.

Each postage-stamp scene contains the exact latent primary and the exact
deployed V2.2 neighbours of a localized coherent anchor.  Magnitudes, circular
half-light radii, Sersic indices, axis ratios, position angles, and relative
positions are copied from the renderer catalogue.  The primary is never
sheared and catalogue positions remain fixed.

For both g1 and g2, the script measures (1) all neighbours coherently sheared
and (2) every neighbour sheared alone while all other scene members remain
present and unsheared.  The image is noiseless and the same seeded ngmix fit
initialization is used for every arm of an anchor.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import galsim
import numpy as np
import pandas as pd

from archive.toy_blend_linearity import BETA, PIX, PSF_FWHM, _PSF, measure


KEY = ["case", "input_index"]
ZERO_POINT = 30.0
Q_MIN, Q_MAX = 0.05, 1.0
SERSIC_N_MIN, SERSIC_N_MAX = 0.3, 6.2
LATENT_COLUMNS = [
    "index", "RA", "DEC", "r", "Re", "axis_ratio",
    "position_angle", "sersic_n",
]


def tangent_offsets(frame: pd.DataFrame, primary_index: int) -> tuple[np.ndarray, np.ndarray]:
    """Small-angle source offsets in arcsec, centered on the primary."""
    by_id = frame.set_index("index", verify_integrity=True)
    primary = by_id.loc[int(primary_index)]
    cosdec = np.cos(np.deg2rad(float(primary.DEC)))
    x = (frame.RA.to_numpy(float) - float(primary.RA)) * cosdec * 3600.0
    y = (frame.DEC.to_numpy(float) - float(primary.DEC)) * 3600.0
    return x, y


def draw_source(row: pd.Series, x: float, y: float, g1: float,
                g2: float, stamp: int) -> np.ndarray:
    """Render one source with the production Sersic/q/PA convention."""
    q = float(np.clip(row.axis_ratio, Q_MIN, Q_MAX))
    n = float(np.clip(row.sersic_n, SERSIC_N_MIN, SERSIC_N_MAX))
    re = float(row.Re) * np.sqrt(q)
    flux = 10.0 ** (-0.4 * (float(row.r) - ZERO_POINT))
    obj = galsim.Sersic(n=n, half_light_radius=re, flux=flux, trunc=0)
    obj = obj.shear(q=q, beta=float(row.position_angle) * galsim.degrees)
    if g1 or g2:
        obj = obj.shear(g1=float(g1), g2=float(g2))
    image = galsim.ImageF(stamp, stamp, scale=PIX)
    galsim.Convolve([obj, _PSF]).drawImage(
        image=image, add_to_image=True, offset=(float(x) / PIX, float(y) / PIX),
    )
    return image.array.astype(np.float64, copy=True)


def measure_shape(image: np.ndarray, seed: int) -> np.ndarray:
    value = np.asarray(measure(np.ascontiguousarray(image), seed), dtype=float)
    if value.shape != (2,) or not np.isfinite(value).all():
        raise RuntimeError("non-finite ngmix shape")
    return value


def central_column(plus: np.ndarray, minus: np.ndarray, g: float) -> np.ndarray:
    """One response-matrix column from matched antithetic shapes."""
    plus = np.asarray(plus, dtype=float)
    minus = np.asarray(minus, dtype=float)
    if plus.shape != (2,) or minus.shape != (2,):
        raise ValueError("central response requires two-component shapes")
    return (plus - minus) / (2.0 * float(g))


def compose_arm_images(source_images: list[dict[str, np.ndarray]]) -> tuple[
        dict[str, np.ndarray], list[dict[str, np.ndarray]]]:
    """Compose coherent and one-active-neighbour images from cached sources."""
    if len(source_images) < 2:
        raise ValueError("scene needs one primary and at least one neighbour")
    primary = source_images[0]["zero"]
    neighbours = source_images[1:]
    base = primary + np.sum([item["zero"] for item in neighbours], axis=0)
    coherent = {
        arm: primary + np.sum([item[arm] for item in neighbours], axis=0)
        for arm in ("g1_plus", "g1_minus", "g2_plus", "g2_minus")
    }
    individual = []
    for item in neighbours:
        individual.append({
            arm: base - item["zero"] + item[arm]
            for arm in ("g1_plus", "g1_minus", "g2_plus", "g2_minus")
        })
    return coherent, individual


def response_matrix(images: dict[str, np.ndarray], g: float, seed: int) -> np.ndarray:
    e1p = measure_shape(images["g1_plus"], seed)
    e1m = measure_shape(images["g1_minus"], seed)
    e2p = measure_shape(images["g2_plus"], seed)
    e2m = measure_shape(images["g2_minus"], seed)
    return np.column_stack([
        central_column(e1p, e1m, g), central_column(e2p, e2m, g),
    ])


def stable_fit_seed(case: int, input_index: int) -> int:
    return int((int(case) * 1_000_003 + int(input_index) * 9_176 + 42) % (2**32 - 1))


def choose_design(paths: list[str], case: int) -> dict:
    matches = []
    for path in paths:
        with open(path, encoding="utf-8") as handle:
            design = json.load(handle)
        cases = {int(item["case"]) for item in design["per_case"]}
        if case in cases:
            design["_path"] = os.path.abspath(path)
            matches.append(design)
    if len(matches) != 1:
        raise RuntimeError(f"case {case} matched {len(matches)} pair designs")
    if matches[0].get("catalogue_source") != "rendered":
        raise RuntimeError("toy requires renderer-catalogue pair manifests")
    return matches[0]


def scene_for_anchor(latent: pd.DataFrame, local_pairs: pd.DataFrame,
                     anchor_index: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    pairs = local_pairs.loc[local_pairs.anchor_index == int(anchor_index)].copy()
    if pairs.empty:
        raise RuntimeError("anchor has no deployed pair")
    if pairs.secondary_index.duplicated().any():
        raise RuntimeError("duplicate deployed secondary")
    ids = np.r_[int(anchor_index), pairs.secondary_index.to_numpy(np.int64)]
    by_id = latent.set_index("index", verify_integrity=True)
    missing = ids[~np.isin(ids, by_id.index.to_numpy(np.int64))]
    if len(missing):
        raise RuntimeError(f"latent scene lacks ids {missing[:5].tolist()}")
    scene = by_id.loc[ids].reset_index()
    return scene, pairs.reset_index(drop=True)


def run_anchor(row: pd.Series, latent: pd.DataFrame, pairs: pd.DataFrame,
               g: float, stamp: int) -> tuple[dict, list[dict]]:
    anchor = int(row.input_index)
    scene, deployed = scene_for_anchor(latent, pairs, anchor)
    x, y = tangent_offsets(scene, anchor)
    observed_distance = np.hypot(x[1:], y[1:])
    distance_error = float(np.max(np.abs(
        observed_distance - deployed.distance.to_numpy(float)
    )))
    if distance_error > 2.0e-3:
        raise RuntimeError(f"pair distance replay differs by {distance_error:.3g} arcsec")
    prediction_error = abs(float(deployed.response.sum()) - float(row.scene_prediction))
    if prediction_error > 2.0e-6:
        raise RuntimeError(f"pair prediction replay differs by {prediction_error:.3g}")

    source_images = []
    for index, source in scene.iterrows():
        item = {"zero": draw_source(source, x[index], y[index], 0.0, 0.0, stamp)}
        if index > 0:
            item.update({
                "g1_plus": draw_source(source, x[index], y[index], +g, 0.0, stamp),
                "g1_minus": draw_source(source, x[index], y[index], -g, 0.0, stamp),
                "g2_plus": draw_source(source, x[index], y[index], 0.0, +g, stamp),
                "g2_minus": draw_source(source, x[index], y[index], 0.0, -g, stamp),
            })
        source_images.append(item)
    coherent_images, individual_images = compose_arm_images(source_images)
    seed = stable_fit_seed(int(row.case), anchor)
    coherent = response_matrix(coherent_images, g, seed)
    individual = np.stack([
        response_matrix(images, g, seed) for images in individual_images
    ])
    summed = individual.sum(axis=0)
    model = float(row.scene_prediction)
    coherent_trace = float(0.5 * np.trace(coherent))
    summed_trace = float(0.5 * np.trace(summed))
    anchor_result = {
        "case": int(row.case),
        "input_index": anchor,
        "success": True,
        "error_type": "",
        "n_deployed_pairs": int(len(deployed)),
        "compact_dominant_secondary": bool(row.compact_dominant_secondary),
        "dominant_secondary_size": float(row.dominant_secondary_size),
        "R_model_sum": model,
        "R_original_coherent_g1": float(row.R_blend_truth),
        "original_truth_minus_model_g1": float(row.bias_truth_minus_model),
        "R11_coherent": float(coherent[0, 0]),
        "R12_coherent": float(coherent[0, 1]),
        "R21_coherent": float(coherent[1, 0]),
        "R22_coherent": float(coherent[1, 1]),
        "R_trace_coherent": coherent_trace,
        "R11_individual_sum": float(summed[0, 0]),
        "R12_individual_sum": float(summed[0, 1]),
        "R21_individual_sum": float(summed[1, 0]),
        "R22_individual_sum": float(summed[1, 1]),
        "R_trace_individual_sum": summed_trace,
        "additivity_gap_g1": float(coherent[0, 0] - summed[0, 0]),
        "additivity_gap_trace": float(coherent_trace - summed_trace),
        "individual_truth_minus_model_g1": float(summed[0, 0] - model),
        "individual_truth_minus_model_trace": float(summed_trace - model),
        "coherent_truth_minus_model_g1": float(coherent[0, 0] - model),
        "coherent_truth_minus_model_trace": float(coherent_trace - model),
        "max_pair_distance_replay_error_arcsec": distance_error,
        "pair_prediction_replay_error": prediction_error,
    }
    pair_results = []
    for j, (pair, matrix) in enumerate(zip(deployed.itertuples(index=False), individual)):
        source = scene.iloc[j + 1]
        pair_results.append({
            "case": int(row.case), "input_index": anchor,
            "secondary_index": int(pair.secondary_index),
            "distance": float(pair.distance),
            "R_model_pair": float(pair.response),
            "secondary_mag": float(source.r),
            "secondary_size": float(source.Re),
            "secondary_sersic_n": float(source.sersic_n),
            "R11_individual": float(matrix[0, 0]),
            "R12_individual": float(matrix[0, 1]),
            "R21_individual": float(matrix[1, 0]),
            "R22_individual": float(matrix[1, 1]),
            "R_trace_individual": float(0.5 * np.trace(matrix)),
        })
    return anchor_result, pair_results


def failure_row(row: pd.Series, error: Exception) -> dict:
    return {
        "case": int(row.case), "input_index": int(row.input_index),
        "success": False, "error_type": type(error).__name__,
        "n_deployed_pairs": int(row.n_deployed_pairs),
        "compact_dominant_secondary": bool(row.compact_dominant_secondary),
        "dominant_secondary_size": float(row.dominant_secondary_size),
        "R_model_sum": float(row.scene_prediction),
        "R_original_coherent_g1": float(row.R_blend_truth),
        "original_truth_minus_model_g1": float(row.bias_truth_minus_model),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--pair-design", nargs="+", required=True)
    ap.add_argument("--case", type=int, required=True)
    ap.add_argument("--g", type=float, default=0.02)
    ap.add_argument("--stamp", type=int, default=48)
    ap.add_argument("--max-anchors", type=int, default=0)
    ap.add_argument("--output-anchor", required=True)
    ap.add_argument("--output-pair", required=True)
    ap.add_argument("--output-json", required=True)
    args = ap.parse_args()
    for path in (args.output_anchor, args.output_pair, args.output_json):
        if os.path.exists(path):
            raise FileExistsError(f"refusing existing output {path}")
    if args.g <= 0 or args.stamp < 32 or args.stamp % 2:
        raise ValueError("g must be positive and stamp must be even and >=32")

    selected = pd.read_feather(args.manifest)
    selected = selected.loc[selected.case == args.case].sort_values("input_index")
    if args.max_anchors:
        selected = selected.iloc[:args.max_anchors]
    if selected.empty:
        raise RuntimeError(f"case {args.case} has no toy anchors")
    design = choose_design(args.pair_design, args.case)
    base = Path(design["manifest_dir"])
    latent = pd.read_feather(
        base / f"gals{args.case}_{float(design['g'])}.feather",
        columns=LATENT_COLUMNS,
    )
    pair_path = base / f"{design.get('pair_prefix', 'pairs')}_case{args.case}.feather"
    pairs = pd.read_feather(pair_path)
    pairs = pairs.loc[pairs.anchor_index.isin(selected.input_index)].copy()

    anchors = []
    pair_rows = []
    errors = []
    for count, row in enumerate(selected.itertuples(index=False), start=1):
        try:
            anchor, local_pairs = run_anchor(
                pd.Series(row._asdict()), latent, pairs, args.g, args.stamp,
            )
            anchors.append(anchor)
            pair_rows.extend(local_pairs)
        except Exception as error:  # preserve the full selected denominator
            anchors.append(failure_row(pd.Series(row._asdict()), error))
            errors.append({
                "input_index": int(row.input_index),
                "error_type": type(error).__name__, "message": str(error)[:500],
            })
        if count % 50 == 0 or count == len(selected):
            print(
                f"case {args.case}: {count}/{len(selected)} anchors; "
                f"failures={len(errors)}", flush=True,
            )

    anchor_frame = pd.DataFrame(anchors)
    pair_frame = pd.DataFrame(pair_rows)
    Path(args.output_anchor).parent.mkdir(parents=True, exist_ok=True)
    anchor_frame.to_feather(args.output_anchor)
    pair_frame.to_feather(args.output_pair)
    payload = {
        "design": (
            "exact latent localized anchor plus exact deployed V2.2 neighbours; "
            "noiseless Moffat-convolved Sersic stamps; matched coherent and "
            "one-neighbour-at-a-time antithetic g1/g2 responses"
        ),
        "case": int(args.case), "g": float(args.g), "stamp": int(args.stamp),
        "manifest": os.path.abspath(args.manifest),
        "pair_design": design["_path"], "latent_base": str(base),
        "pair_path": str(pair_path), "primary_sheared": False,
        "positions_sheared": False, "pixel_noise": 0.0,
        "measurement_center": "true primary center",
        "scene_members": "primary plus exact deployed V2.2 neighbours",
        "n_selected": int(len(selected)),
        "n_success": int(anchor_frame.success.sum()),
        "n_failure": int((~anchor_frame.success).sum()),
        "n_pair_rows": int(len(pair_frame)),
        "failures": errors[:100],
    }
    with open(args.output_json, "x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps({k: payload[k] for k in (
        "case", "n_selected", "n_success", "n_failure", "n_pair_rows"
    )}, indent=2))
    print("LOCALIZED_ANCHOR_NOISELESS_TOY_CASE_DONE", flush=True)


if __name__ == "__main__":
    main()
