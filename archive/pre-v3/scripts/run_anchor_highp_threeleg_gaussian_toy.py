"""Run one controlled three-leg Gaussian anchor-template toy.

Each selected catalogue configuration is circularized and rendered with
Gaussian profiles, matching the ngmix measurement model used by the old clean
additivity tests.  Exact catalogue flux ratios, circularized sizes, and
relative positions are retained.  The primary is fixed and never sheared;
source positions are fixed.

For each nominal primary S/N, the response trace is measured in three matched
legs:

* coherent: all selected neighbours are sheared together;
* context sum: one neighbour is sheared at a time with the others present;
* pair-only sum: one neighbour is sheared with all other neighbours removed.

Every noisy realization uses the same pixel-noise image and ngmix seed in all
counterfactual images, and also across S/N cells.  A deterministic noiseless
control is stored for every cell.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import galsim
import numpy as np
import pandas as pd

from archive.toy_blend_linearity import PIX, _PSF, measure


ARMS = ("g1_plus", "g1_minus", "g2_plus", "g2_minus")
UINT32_MODULUS = 2**32 - 1
NOISE_EQUIVALENT_PIXELS = 20.0


def repeat_seed(template_id: int, realization: int, salt: int) -> int:
    """Stable seed shared by every counterfactual in one matched block."""
    value = (
        int(template_id) * 1_000_003
        + int(realization) * 2_654_435_761
        + int(salt)
    ) % UINT32_MODULUS
    return int(value or 1)


def primary_flux_for_nominal_snr(
    nominal_snr: float,
    pixel_rms: float,
    noise_equivalent_pixels: float = NOISE_EQUIVALENT_PIXELS,
) -> float:
    """Invert the legacy toy's approximate S/N definition."""
    if nominal_snr <= 0 or pixel_rms <= 0 or noise_equivalent_pixels <= 0:
        raise ValueError("S/N, pixel RMS, and noise-equivalent pixels must be positive")
    return float(nominal_snr * pixel_rms * np.sqrt(noise_equivalent_pixels))


def draw_round_gaussian(
    flux: float,
    hlr_arcsec: float,
    x_arcsec: float,
    y_arcsec: float,
    g1: float,
    g2: float,
    stamp: int,
) -> np.ndarray:
    """Render one round Gaussian source convolved by the established Moffat PSF."""
    if flux <= 0 or hlr_arcsec <= 0:
        raise ValueError("source flux and HLR must be positive")
    obj = galsim.Gaussian(half_light_radius=float(hlr_arcsec), flux=float(flux))
    if g1 or g2:
        obj = obj.shear(g1=float(g1), g2=float(g2))
    image = galsim.ImageF(int(stamp), int(stamp), scale=PIX)
    galsim.Convolve([obj, _PSF]).drawImage(
        image=image,
        add_to_image=True,
        offset=(float(x_arcsec) / PIX, float(y_arcsec) / PIX),
    )
    return image.array.astype(np.float64, copy=True)


def compose_three_leg_images(
    source_images: list[dict[str, np.ndarray]],
) -> tuple[
    dict[str, np.ndarray],
    list[dict[str, np.ndarray]],
    list[dict[str, np.ndarray]],
]:
    """Compose coherent, all-present one-active, and isolated-pair images."""
    if len(source_images) < 3:
        raise ValueError("three-leg toy requires a primary and at least two neighbours")
    primary = source_images[0]["zero"]
    neighbours = source_images[1:]
    base = primary + np.sum([item["zero"] for item in neighbours], axis=0)
    coherent = {
        arm: primary + np.sum([item[arm] for item in neighbours], axis=0)
        for arm in ARMS
    }
    contextual = [
        {arm: base - item["zero"] + item[arm] for arm in ARMS}
        for item in neighbours
    ]
    pair_only = [
        {arm: primary + item[arm] for arm in ARMS}
        for item in neighbours
    ]
    return coherent, contextual, pair_only


def measured_shape(image: np.ndarray, fit_seed: int) -> np.ndarray:
    value = np.asarray(measure(np.ascontiguousarray(image), fit_seed), dtype=float)
    if value.shape != (2,) or not np.isfinite(value).all():
        raise RuntimeError("non-finite ngmix shape")
    return value


def response_matrix(
    images: dict[str, np.ndarray],
    g: float,
    fit_seed: int,
    noise: np.ndarray | None = None,
) -> np.ndarray:
    """Central 2x2 measured-shape response using one common noise image."""
    if noise is None:
        noise = np.zeros_like(images["g1_plus"])
    e1p = measured_shape(images["g1_plus"] + noise, fit_seed)
    e1m = measured_shape(images["g1_minus"] + noise, fit_seed)
    e2p = measured_shape(images["g2_plus"] + noise, fit_seed)
    e2m = measured_shape(images["g2_minus"] + noise, fit_seed)
    return np.column_stack([(e1p - e1m) / (2.0 * g), (e2p - e2m) / (2.0 * g)])


def render_cell(
    sources: pd.DataFrame,
    nominal_snr: float,
    pixel_rms: float,
    g: float,
    stamp: int,
) -> tuple[
    dict[str, np.ndarray],
    list[dict[str, np.ndarray]],
    list[dict[str, np.ndarray]],
    list[float],
]:
    """Render and cache all noiseless images for one template/S/N cell."""
    sources = sources.sort_values("source_rank", kind="mergesort")
    primary_mag = float(sources.iloc[0].source_mag)
    primary_flux = primary_flux_for_nominal_snr(nominal_snr, pixel_rms)
    source_images: list[dict[str, np.ndarray]] = []
    fluxes: list[float] = []
    for source in sources.itertuples(index=False):
        flux_ratio = 10.0 ** (-0.4 * (float(source.source_mag) - primary_mag))
        flux = float(primary_flux * flux_ratio)
        fluxes.append(flux)
        item = {
            "zero": draw_round_gaussian(
                flux,
                float(source.source_hlr_circularized_arcsec),
                float(source.x_arcsec),
                float(source.y_arcsec),
                0.0,
                0.0,
                stamp,
            )
        }
        if str(source.source_role) == "neighbour":
            item.update({
                "g1_plus": draw_round_gaussian(
                    flux, source.source_hlr_circularized_arcsec,
                    source.x_arcsec, source.y_arcsec, +g, 0.0, stamp,
                ),
                "g1_minus": draw_round_gaussian(
                    flux, source.source_hlr_circularized_arcsec,
                    source.x_arcsec, source.y_arcsec, -g, 0.0, stamp,
                ),
                "g2_plus": draw_round_gaussian(
                    flux, source.source_hlr_circularized_arcsec,
                    source.x_arcsec, source.y_arcsec, 0.0, +g, stamp,
                ),
                "g2_minus": draw_round_gaussian(
                    flux, source.source_hlr_circularized_arcsec,
                    source.x_arcsec, source.y_arcsec, 0.0, -g, stamp,
                ),
            })
        source_images.append(item)
    coherent, contextual, pair_only = compose_three_leg_images(source_images)
    return coherent, contextual, pair_only, fluxes


def measure_matched_block(
    template: pd.Series,
    sources: pd.DataFrame,
    nominal_snr: float,
    realization: int,
    noise_mode: str,
    noise_seed: int,
    fit_seed: int,
    coherent: dict[str, np.ndarray],
    contextual: list[dict[str, np.ndarray]],
    pair_only: list[dict[str, np.ndarray]],
    g: float,
    pixel_rms: float,
) -> tuple[dict, list[dict]]:
    """Measure one complete A/B/C counterfactual block."""
    if noise_mode == "noisy":
        noise = np.random.RandomState(noise_seed).normal(
            0.0, float(pixel_rms), coherent["g1_plus"].shape
        )
    elif noise_mode == "noiseless":
        noise = np.zeros_like(coherent["g1_plus"])
    else:
        raise ValueError(f"unknown noise mode {noise_mode}")
    matrix_coherent = response_matrix(coherent, g, fit_seed, noise)
    matrices_context = np.stack([
        response_matrix(images, g, fit_seed, noise) for images in contextual
    ])
    matrices_pair = np.stack([
        response_matrix(images, g, fit_seed, noise) for images in pair_only
    ])
    matrix_context_sum = matrices_context.sum(axis=0)
    matrix_pair_sum = matrices_pair.sum(axis=0)

    def trace(matrix: np.ndarray) -> float:
        return float(0.5 * np.trace(matrix))

    r_coherent = trace(matrix_coherent)
    r_context = trace(matrix_context_sum)
    r_pair = trace(matrix_pair_sum)
    delta_add = r_coherent - r_context
    delta_context = r_context - r_pair
    delta_total = r_coherent - r_pair
    draw = {
        "template_id": int(template.template_id),
        "case": int(template.case),
        "input_index": int(template.input_index),
        "multiplicity": int(template.multiplicity),
        "nominal_snr": float(nominal_snr),
        "realization": int(realization),
        "noise_mode": str(noise_mode),
        "noise_seed": int(noise_seed),
        "fit_seed": int(fit_seed),
        "success": True,
        "error_type": "",
        "R11_coherent": float(matrix_coherent[0, 0]),
        "R22_coherent": float(matrix_coherent[1, 1]),
        "R_trace_coherent": r_coherent,
        "R11_context_sum": float(matrix_context_sum[0, 0]),
        "R22_context_sum": float(matrix_context_sum[1, 1]),
        "R_trace_context_sum": r_context,
        "R11_pair_sum": float(matrix_pair_sum[0, 0]),
        "R22_pair_sum": float(matrix_pair_sum[1, 1]),
        "R_trace_pair_sum": r_pair,
        "delta_additivity_trace": delta_add,
        "delta_context_trace": delta_context,
        "delta_total_trace": delta_total,
        "decomposition_replay_trace": float(delta_total - delta_add - delta_context),
    }
    pair_rows = []
    neighbours = sources.loc[sources.source_role == "neighbour"].sort_values(
        "source_rank", kind="mergesort"
    )
    for source, matrix_context, matrix_pair in zip(
        neighbours.itertuples(index=False), matrices_context, matrices_pair
    ):
        context_trace = trace(matrix_context)
        pair_trace = trace(matrix_pair)
        pair_rows.append({
            "template_id": int(template.template_id),
            "case": int(template.case),
            "input_index": int(template.input_index),
            "multiplicity": int(template.multiplicity),
            "nominal_snr": float(nominal_snr),
            "realization": int(realization),
            "noise_mode": str(noise_mode),
            "source_rank": int(source.source_rank),
            "secondary_index": int(source.source_index),
            "distance_arcsec": float(source.distance_arcsec),
            "R_model_pair": float(source.R_model_pair),
            "R_trace_context": context_trace,
            "R_trace_pair_only": pair_trace,
            "delta_context_pair_trace": float(context_trace - pair_trace),
        })
    return draw, pair_rows


def failure_row(
    template: pd.Series,
    nominal_snr: float,
    realization: int,
    noise_mode: str,
    noise_seed: int,
    fit_seed: int,
    error: Exception,
) -> dict:
    return {
        "template_id": int(template.template_id),
        "case": int(template.case),
        "input_index": int(template.input_index),
        "multiplicity": int(template.multiplicity),
        "nominal_snr": float(nominal_snr),
        "realization": int(realization),
        "noise_mode": str(noise_mode),
        "noise_seed": int(noise_seed),
        "fit_seed": int(fit_seed),
        "success": False,
        "error_type": type(error).__name__,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--templates", required=True)
    parser.add_argument("--sources", required=True)
    parser.add_argument("--template-id", type=int, required=True)
    parser.add_argument("--nominal-snr", nargs="+", type=float, default=[12, 20, 35])
    parser.add_argument("--nreal", type=int, default=200)
    parser.add_argument("--g", type=float, default=0.02)
    parser.add_argument("--stamp", type=int, default=96)
    parser.add_argument("--pixel-rms", type=float, default=10.0)
    parser.add_argument("--output-draws", required=True)
    parser.add_argument("--output-pairs", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()
    for path in (args.output_draws, args.output_pairs, args.output_json):
        if os.path.exists(path):
            raise FileExistsError(f"refusing existing output {path}")
    if args.nreal < 1 or args.g <= 0 or args.pixel_rms <= 0:
        raise ValueError("nreal >=1, g >0, and pixel-rms >0 are required")
    if args.stamp < 64 or args.stamp % 2:
        raise ValueError("stamp must be even and at least 64 pixels")
    snr_levels = tuple(float(value) for value in args.nominal_snr)
    if len(set(snr_levels)) != len(snr_levels) or min(snr_levels) <= 0:
        raise ValueError("nominal S/N levels must be distinct and positive")

    templates = pd.read_feather(args.templates)
    selected = templates.loc[templates.template_id == args.template_id]
    if len(selected) != 1:
        raise RuntimeError(
            f"template id {args.template_id} matched {len(selected)} rows"
        )
    template = pd.Series(selected.iloc[0])
    sources = pd.read_feather(args.sources)
    sources = sources.loc[sources.template_id == args.template_id].sort_values(
        "source_rank", kind="mergesort"
    )
    if len(sources) != int(template.multiplicity) + 1:
        raise RuntimeError("source count does not equal primary plus multiplicity")
    if (sources.iloc[0].source_role != "primary") or not (
        sources.iloc[1:].source_role == "neighbour"
    ).all():
        raise RuntimeError("source ordering/roles are invalid")

    rendered = {}
    cell_fluxes = {}
    for snr in snr_levels:
        coherent, contextual, pair_only, fluxes = render_cell(
            sources, snr, args.pixel_rms, args.g, args.stamp
        )
        rendered[snr] = (coherent, contextual, pair_only)
        cell_fluxes[str(snr)] = [float(value) for value in fluxes]

    draws: list[dict] = []
    pair_rows: list[dict] = []
    failures: list[dict] = []
    # One deterministic noiseless control per S/N cell.
    noiseless_fit_seed = repeat_seed(args.template_id, -1, 1_000_000_007)
    for snr in snr_levels:
        coherent, contextual, pair_only = rendered[snr]
        try:
            draw, local_pairs = measure_matched_block(
                template, sources, snr, -1, "noiseless", 0,
                noiseless_fit_seed, coherent, contextual, pair_only,
                args.g, args.pixel_rms,
            )
            draws.append(draw)
            pair_rows.extend(local_pairs)
        except Exception as error:
            draws.append(failure_row(
                template, snr, -1, "noiseless", 0,
                noiseless_fit_seed, error,
            ))
            failures.append({
                "nominal_snr": snr, "realization": -1,
                "noise_mode": "noiseless", "error_type": type(error).__name__,
                "message": str(error)[:500],
            })

    for realization in range(args.nreal):
        noise_seed = repeat_seed(args.template_id, realization, 17)
        fit_seed = repeat_seed(
            args.template_id, realization, 1_000_000_007
        )
        for snr in snr_levels:
            coherent, contextual, pair_only = rendered[snr]
            try:
                draw, local_pairs = measure_matched_block(
                    template, sources, snr, realization, "noisy",
                    noise_seed, fit_seed, coherent, contextual, pair_only,
                    args.g, args.pixel_rms,
                )
                draws.append(draw)
                pair_rows.extend(local_pairs)
            except Exception as error:
                draws.append(failure_row(
                    template, snr, realization, "noisy", noise_seed,
                    fit_seed, error,
                ))
                failures.append({
                    "nominal_snr": snr, "realization": realization,
                    "noise_mode": "noisy", "error_type": type(error).__name__,
                    "message": str(error)[:500],
                })
        if (realization + 1) % 25 == 0 or realization + 1 == args.nreal:
            print(
                f"template {args.template_id}: {realization + 1}/{args.nreal}; "
                f"failures={len(failures)}",
                flush=True,
            )

    draw_frame = pd.DataFrame(draws)
    pair_frame = pd.DataFrame(pair_rows)
    Path(args.output_draws).parent.mkdir(parents=True, exist_ok=True)
    draw_frame.to_feather(args.output_draws)
    pair_frame.to_feather(args.output_pairs)
    success = draw_frame.success.astype(bool)
    successful = draw_frame.loc[success]
    replay_max = float(
        successful.decomposition_replay_trace.abs().max()
        if len(successful) else np.nan
    )
    payload = {
        "design": (
            "round-Gaussian catalogue-informed close-neighbour scene; matched "
            "coherent, one-at-a-time with context, and isolated-pair sum"
        ),
        "template_id": int(args.template_id),
        "case": int(template.case),
        "input_index": int(template.input_index),
        "multiplicity": int(template.multiplicity),
        "nominal_snr": list(snr_levels),
        "nreal": int(args.nreal),
        "g": float(args.g),
        "stamp": int(args.stamp),
        "pixel_rms": float(args.pixel_rms),
        "noise_equivalent_pixels": float(NOISE_EQUIVALENT_PIXELS),
        "primary_flux_by_snr": {
            str(snr): primary_flux_for_nominal_snr(snr, args.pixel_rms)
            for snr in snr_levels
        },
        "source_fluxes_by_snr": cell_fluxes,
        "primary_sheared": False,
        "positions_sheared": False,
        "intrinsic_profiles": "round Gaussian",
        "both_shear_axes": True,
        "response_scalar": "0.5 * trace(2x2 central response matrix)",
        "same_noise_all_three_legs_and_shear_arms_within_block": True,
        "same_fit_seed_all_three_legs_and_shear_arms_within_block": True,
        "same_noise_and_fit_seed_across_snr_within_block": True,
        "n_rows": int(len(draw_frame)),
        "n_success": int(success.sum()),
        "n_failure": int((~success).sum()),
        "success_fraction": float(success.mean()),
        "n_pair_rows": int(len(pair_frame)),
        "max_decomposition_replay_abs": replay_max,
        "failures": failures[:100],
        "inputs": {
            "templates": os.path.abspath(args.templates),
            "sources": os.path.abspath(args.sources),
        },
    }
    with open(args.output_json, "x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps({key: payload[key] for key in (
        "template_id", "case", "multiplicity", "n_success", "n_failure",
        "n_pair_rows", "max_decomposition_replay_abs",
    )}, indent=2))
    print("ANCHOR_HIGHP_THREELEG_GAUSSIAN_TEMPLATE_DONE", flush=True)


if __name__ == "__main__":
    main()
