"""Run common-noise coherent/per-neighbour repeats for one frozen anchor scene.

Every noise realization is a complete randomized block: the identical Gaussian
sky-noise image and identical ngmix initialization are used for all antithetic,
coherent, and one-neighbour-active counterfactual arms.  Noise and fit seeds vary
independently between realizations and between scenes.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.run_localized_anchor_noiseless_toy_case import (
    LATENT_COLUMNS,
    central_column,
    choose_design,
    compose_arm_images,
    draw_source,
    measure_shape,
    scene_for_anchor,
    tangent_offsets,
)


UINT32_MODULUS = 2**32 - 1


def repeat_seed(case: int, input_index: int, realization: int, salt: int) -> int:
    """Stable, scene-specific seed for one technical replicate."""
    value = (
        int(case) * 1_000_003
        + int(input_index) * 9_176
        + int(realization) * 2_654_435_761
        + int(salt)
    ) % UINT32_MODULUS
    return int(value or 1)


def noisy_response_matrix(images: dict[str, np.ndarray], noise: np.ndarray,
                          g: float, fit_seed: int) -> np.ndarray:
    """Measure a response matrix with one common noise field in all four arms."""
    e1p = measure_shape(images["g1_plus"] + noise, fit_seed)
    e1m = measure_shape(images["g1_minus"] + noise, fit_seed)
    e2p = measure_shape(images["g2_plus"] + noise, fit_seed)
    e2m = measure_shape(images["g2_minus"] + noise, fit_seed)
    return np.column_stack([
        central_column(e1p, e1m, g), central_column(e2p, e2m, g),
    ])


def render_scene(row: pd.Series, latent: pd.DataFrame, pairs: pd.DataFrame,
                 g: float, stamp: int) -> tuple[
                     dict[str, np.ndarray], list[dict[str, np.ndarray]],
                     pd.DataFrame, float, float]:
    """Render and validate all noiseless counterfactual arms once."""
    anchor = int(row.input_index)
    scene, deployed = scene_for_anchor(latent, pairs, anchor)
    x, y = tangent_offsets(scene, anchor)
    observed_distance = np.hypot(x[1:], y[1:])
    distance_error = float(np.max(np.abs(
        observed_distance - deployed.distance.to_numpy(float)
    )))
    if distance_error > 2.0e-3:
        raise RuntimeError(f"pair distance replay differs by {distance_error:.3g}")
    prediction_error = abs(
        float(deployed.response.sum()) - float(row.scene_prediction)
    )
    if prediction_error > 2.0e-6:
        raise RuntimeError(
            f"pair prediction replay differs by {prediction_error:.3g}"
        )

    source_images = []
    for index, source in scene.iterrows():
        item = {"zero": draw_source(
            source, x[index], y[index], 0.0, 0.0, stamp,
        )}
        if index > 0:
            item.update({
                "g1_plus": draw_source(
                    source, x[index], y[index], +g, 0.0, stamp,
                ),
                "g1_minus": draw_source(
                    source, x[index], y[index], -g, 0.0, stamp,
                ),
                "g2_plus": draw_source(
                    source, x[index], y[index], 0.0, +g, stamp,
                ),
                "g2_minus": draw_source(
                    source, x[index], y[index], 0.0, -g, stamp,
                ),
            })
        source_images.append(item)
    coherent, individual = compose_arm_images(source_images)
    return coherent, individual, deployed, distance_error, prediction_error


def successful_draw(row: pd.Series, realization: int, noise_seed: int,
                    fit_seed: int, coherent: dict[str, np.ndarray],
                    individual: list[dict[str, np.ndarray]],
                    deployed: pd.DataFrame, g: float,
                    pixel_rms: float) -> tuple[dict, list[dict]]:
    """Measure one matched noise block and return aggregate and pair rows."""
    rng = np.random.RandomState(noise_seed)
    shape = coherent["g1_plus"].shape
    noise = rng.normal(0.0, float(pixel_rms), shape)
    coherent_matrix = noisy_response_matrix(coherent, noise, g, fit_seed)
    individual_matrices = np.stack([
        noisy_response_matrix(images, noise, g, fit_seed)
        for images in individual
    ])
    summed = individual_matrices.sum(axis=0)
    coherent_trace = float(0.5 * np.trace(coherent_matrix))
    summed_trace = float(0.5 * np.trace(summed))
    draw = {
        "scene_id": int(row.scene_id), "case": int(row.case),
        "input_index": int(row.input_index),
        "realization": int(realization), "noise_seed": int(noise_seed),
        "fit_seed": int(fit_seed), "success": True, "error_type": "",
        "R_model_sum": float(row.scene_prediction),
        "R11_coherent": float(coherent_matrix[0, 0]),
        "R22_coherent": float(coherent_matrix[1, 1]),
        "R_trace_coherent": coherent_trace,
        "R11_individual_sum": float(summed[0, 0]),
        "R22_individual_sum": float(summed[1, 1]),
        "R_trace_individual_sum": summed_trace,
        "additivity_gap_g1": float(coherent_matrix[0, 0] - summed[0, 0]),
        "additivity_gap_trace": float(coherent_trace - summed_trace),
        "coherent_truth_minus_model_g1": float(
            coherent_matrix[0, 0] - row.scene_prediction
        ),
        "individual_truth_minus_model_g1": float(
            summed[0, 0] - row.scene_prediction
        ),
        "coherent_truth_minus_model_trace": float(
            coherent_trace - row.scene_prediction
        ),
        "individual_truth_minus_model_trace": float(
            summed_trace - row.scene_prediction
        ),
    }
    pair_rows = []
    for pair, matrix in zip(deployed.itertuples(index=False), individual_matrices):
        pair_rows.append({
            "scene_id": int(row.scene_id), "case": int(row.case),
            "input_index": int(row.input_index),
            "realization": int(realization),
            "secondary_index": int(pair.secondary_index),
            "distance": float(pair.distance),
            "R_model_pair": float(pair.response),
            "R11_individual": float(matrix[0, 0]),
            "R22_individual": float(matrix[1, 1]),
            "R_trace_individual": float(0.5 * np.trace(matrix)),
        })
    return draw, pair_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--pair-design", nargs="+", required=True)
    parser.add_argument("--scene-id", type=int, required=True)
    parser.add_argument("--nreal", type=int, default=400)
    parser.add_argument("--g", type=float, default=0.02)
    parser.add_argument("--stamp", type=int, default=48)
    parser.add_argument("--pixel-rms", type=float, default=0.312)
    parser.add_argument("--output-draws", required=True)
    parser.add_argument("--output-pairs", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()
    for path in (args.output_draws, args.output_pairs, args.output_json):
        if os.path.exists(path):
            raise FileExistsError(f"refusing existing output {path}")
    if args.nreal < 2 or args.g <= 0 or args.pixel_rms <= 0:
        raise ValueError("nreal >= 2, g > 0, and pixel_rms > 0 are required")
    if args.stamp < 32 or args.stamp % 2:
        raise ValueError("stamp must be even and >= 32")

    manifest = pd.read_feather(args.manifest)
    selected = manifest.loc[manifest.scene_id == args.scene_id]
    if len(selected) != 1:
        raise RuntimeError(
            f"scene id {args.scene_id} matched {len(selected)} manifest rows"
        )
    row = pd.Series(selected.iloc[0])
    case = int(row.case)
    anchor = int(row.input_index)
    design = choose_design(args.pair_design, case)
    base = Path(design["manifest_dir"])
    latent = pd.read_feather(
        base / f"gals{case}_{float(design['g'])}.feather",
        columns=LATENT_COLUMNS,
    )
    pair_path = base / f"{design.get('pair_prefix', 'pairs')}_case{case}.feather"
    pairs = pd.read_feather(pair_path)
    pairs = pairs.loc[pairs.anchor_index == anchor].copy()
    coherent, individual, deployed, distance_error, prediction_error = render_scene(
        row, latent, pairs, args.g, args.stamp,
    )
    if len(deployed) != int(row.n_deployed_pairs):
        raise RuntimeError(
            f"deployed pair count {len(deployed)} != manifest {row.n_deployed_pairs}"
        )

    draws = []
    pair_rows = []
    failures = []
    for realization in range(args.nreal):
        noise_seed = repeat_seed(case, anchor, realization, 17)
        fit_seed = repeat_seed(case, anchor, realization, 1_000_000_007)
        try:
            draw, local_pairs = successful_draw(
                row, realization, noise_seed, fit_seed,
                coherent, individual, deployed, args.g, args.pixel_rms,
            )
            draws.append(draw)
            pair_rows.extend(local_pairs)
        except Exception as error:
            draws.append({
                "scene_id": int(args.scene_id), "case": case,
                "input_index": anchor, "realization": realization,
                "noise_seed": noise_seed, "fit_seed": fit_seed,
                "success": False, "error_type": type(error).__name__,
            })
            failures.append({
                "realization": realization,
                "error_type": type(error).__name__,
                "message": str(error)[:500],
            })
        if (realization + 1) % 50 == 0 or realization + 1 == args.nreal:
            print(
                f"scene {args.scene_id}: {realization + 1}/{args.nreal}; "
                f"failures={len(failures)}", flush=True,
            )

    draw_frame = pd.DataFrame(draws)
    pair_frame = pd.DataFrame(pair_rows)
    Path(args.output_draws).parent.mkdir(parents=True, exist_ok=True)
    draw_frame.to_feather(args.output_draws)
    pair_frame.to_feather(args.output_pairs)
    payload = {
        "design": (
            "exact localized anchor scene; common Gaussian sky-noise and common "
            "ngmix initialization within each coherent/individual antithetic block"
        ),
        "scene_id": int(args.scene_id), "case": case, "input_index": anchor,
        "selection_stratum": str(row.selection_stratum),
        "compact_dominant_secondary": bool(row.compact_dominant_secondary),
        "n_deployed_pairs": int(len(deployed)),
        "n_requested": int(args.nreal),
        "n_success": int(draw_frame.success.sum()),
        "n_failure": int((~draw_frame.success).sum()),
        "g": float(args.g), "stamp": int(args.stamp),
        "pixel_rms": float(args.pixel_rms),
        "primary_sheared": False, "positions_sheared": False,
        "both_shear_axes": True,
        "same_noise_all_counterfactual_arms_within_realization": True,
        "same_fit_seed_all_counterfactual_arms_within_realization": True,
        "noise_and_fit_seeds_vary_between_realizations": True,
        "measurement_center": "true primary center",
        "pair_design": design["_path"], "latent_base": str(base),
        "pair_path": str(pair_path),
        "max_pair_distance_replay_error_arcsec": distance_error,
        "pair_prediction_replay_error": prediction_error,
        "failures": failures[:100],
    }
    with open(args.output_json, "x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps({
        key: payload[key] for key in (
            "scene_id", "case", "input_index", "n_deployed_pairs",
            "n_requested", "n_success", "n_failure",
        )
    }, indent=2))
    print("LOCALIZED_ANCHOR_NOISE_REPEAT_SCENE_DONE", flush=True)


if __name__ == "__main__":
    main()
