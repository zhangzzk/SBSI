"""Measure azimuth-averaged clean R_blend for sampled half-shear pairs.

Every stamp contains only the sampled primary and secondary.  The primary is
unsheared; the secondary is sheared antithetically along both component axes;
the secondary position is rotated around the primary in 45-degree steps while
both intrinsic morphologies remain fixed.  The reported pair response is the
mean of the eight central two-component response matrices.  Pixel noise is zero
and the same ngmix initialization is used for every arm, angle, and pair,
matching the old clean toy.
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


ZERO_POINT = 30.0
Q_MIN, Q_MAX = 0.05, 1.0
SERSIC_N_MIN, SERSIC_N_MAX = 0.3, 6.2
ARMS = ("g1_plus", "g1_minus", "g2_plus", "g2_minus")


def latent_offset(row: pd.Series) -> tuple[float, float, float]:
    """True secondary offset from the true primary in arcsec."""
    cosdec = np.cos(np.deg2rad(float(row.DEC_input_p)))
    x = (float(row.RA_input_s) - float(row.RA_input_p)) * cosdec * 3600.0
    y = (float(row.DEC_input_s) - float(row.DEC_input_p)) * 3600.0
    return float(x), float(y), float(np.hypot(x, y))


def azimuth_offsets_deg(n_azimuths: int) -> np.ndarray:
    """Even offsets around the primary; eight gives the requested 45-degree grid."""
    if n_azimuths < 1:
        raise ValueError("n_azimuths must be positive")
    return np.arange(n_azimuths, dtype=float) * (360.0 / n_azimuths)


def rotate_offset(x: float, y: float, angle_deg: float) -> tuple[float, float]:
    """Rotate a sky-plane neighbour displacement counter-clockwise."""
    angle = np.deg2rad(float(angle_deg))
    cosine, sine = np.cos(angle), np.sin(angle)
    return (
        float(cosine * x - sine * y),
        float(sine * x + cosine * y),
    )


def draw_source(
    row: pd.Series,
    suffix: str,
    x: float,
    y: float,
    g1: float,
    g2: float,
    stamp: int,
) -> np.ndarray:
    """Render one catalogue source using the established anchor-toy convention."""
    q = float(np.clip(row[f"axis_ratio_input_{suffix}"], Q_MIN, Q_MAX))
    n = float(np.clip(row[f"sersic_n_input_{suffix}"], SERSIC_N_MIN, SERSIC_N_MAX))
    # Re_input is the semi-major value written by MultiBand_ImSim.  GalSim's
    # area-preserving q shear starts from the corresponding circularized HLR.
    re_circularized = float(row[f"Re_input_{suffix}"]) * np.sqrt(q)
    flux = 10.0 ** (-0.4 * (float(row[f"r_input_{suffix}"]) - ZERO_POINT))
    obj = galsim.Sersic(
        n=n, half_light_radius=re_circularized, flux=flux, trunc=0,
    )
    obj = obj.shear(
        q=q, beta=float(row[f"position_angle_input_{suffix}"]) * galsim.degrees,
    )
    if g1 or g2:
        obj = obj.shear(g1=float(g1), g2=float(g2))
    image = galsim.ImageF(stamp, stamp, scale=PIX)
    galsim.Convolve([obj, _PSF]).drawImage(
        image=image, add_to_image=True, offset=(float(x) / PIX, float(y) / PIX),
    )
    return image.array.astype(np.float64, copy=True)


def compose_pair_arms(
    row: pd.Series,
    g: float,
    stamp: int,
    azimuth_offset_deg: float = 0.0,
    primary: np.ndarray | None = None,
) -> tuple[dict, dict]:
    """Return four two-object images at one rotated neighbour position."""
    native_x, native_y, latent_distance = latent_offset(row)
    x, y = rotate_offset(native_x, native_y, azimuth_offset_deg)
    if primary is None:
        primary = draw_source(row, "p", 0.0, 0.0, 0.0, 0.0, stamp)
    secondary = {
        "g1_plus": draw_source(row, "s", x, y, +g, 0.0, stamp),
        "g1_minus": draw_source(row, "s", x, y, -g, 0.0, stamp),
        "g2_plus": draw_source(row, "s", x, y, 0.0, +g, stamp),
        "g2_minus": draw_source(row, "s", x, y, 0.0, -g, stamp),
    }
    return (
        {arm: primary + secondary[arm] for arm in ARMS},
        {
            "native_dx_arcsec": native_x,
            "native_dy_arcsec": native_y,
            "rotated_dx_arcsec": x,
            "rotated_dy_arcsec": y,
            "latent_distance_arcsec": latent_distance,
            "catalogue_minus_latent_distance_arcsec": (
                float(row.distance) - latent_distance
            ),
        },
    )


def measured_shape(image: np.ndarray, fit_seed: int) -> np.ndarray:
    value = np.asarray(measure(np.ascontiguousarray(image), fit_seed), dtype=float)
    if value.shape != (2,) or not np.isfinite(value).all():
        raise RuntimeError("non-finite ngmix shape")
    return value


def response_matrix(images: dict[str, np.ndarray], g: float, fit_seed: int) -> np.ndarray:
    """Central 2x2 response of measured primary shape to secondary shear."""
    g1 = (
        measured_shape(images["g1_plus"], fit_seed)
        - measured_shape(images["g1_minus"], fit_seed)
    ) / (2.0 * g)
    g2 = (
        measured_shape(images["g2_plus"], fit_seed)
        - measured_shape(images["g2_minus"], fit_seed)
    ) / (2.0 * g)
    return np.column_stack([g1, g2])


def row_metadata(row: pd.Series) -> dict:
    return {
        "sample_id": int(row.sample_id),
        "catalogue_row": int(row.catalogue_row),
        "case": int(row.case),
        "input_index": int(row.input_index),
        "R_emulator_v22": float(row.R_emulator_v22),
        "R_label_forward": float(row.R_label_forward),
        "R_label_null": float(row.R_label_null),
        "r_input_p": float(row.r_input_p),
        "r_input_s": float(row.r_input_s),
        "Re_input_p": float(row.Re_input_p),
        "Re_input_s": float(row.Re_input_s),
        "sersic_n_input_p": float(row.sersic_n_input_p),
        "sersic_n_input_s": float(row.sersic_n_input_s),
        "axis_ratio_input_p": float(row.axis_ratio_input_p),
        "axis_ratio_input_s": float(row.axis_ratio_input_s),
        "distance": float(row.distance),
    }


def run_pair(
    row: pd.Series,
    g: float,
    stamp: int,
    fit_seed: int,
    n_azimuths: int,
) -> tuple[dict, list[dict]]:
    """Measure every azimuth and return the matrix-averaged pair result."""
    offsets = azimuth_offsets_deg(n_azimuths)
    primary = draw_source(row, "p", 0.0, 0.0, 0.0, 0.0, stamp)
    rotations = []
    matrices = []
    base = row_metadata(row)
    native_geometry = None
    for rotation_index, offset in enumerate(offsets):
        images, geometry = compose_pair_arms(
            row, g, stamp, float(offset), primary=primary,
        )
        matrix = response_matrix(images, g, fit_seed)
        matrices.append(matrix)
        native_geometry = geometry
        rotations.append({
            **base,
            "rotation_index": int(rotation_index),
            "azimuth_offset_deg": float(offset),
            "R11_toy": float(matrix[0, 0]),
            "R12_toy": float(matrix[0, 1]),
            "R21_toy": float(matrix[1, 0]),
            "R22_toy": float(matrix[1, 1]),
            "R_blend_toy": float(0.5 * np.trace(matrix)),
            **geometry,
        })

    stack = np.stack(matrices)
    mean_matrix = stack.mean(axis=0)
    rotation_responses = np.asarray(
        [item["R_blend_toy"] for item in rotations], dtype=float,
    )
    assert native_geometry is not None
    pair = {
        **base,
        "success": True,
        "error_type": "",
        "error_message": "",
        "n_azimuths": int(n_azimuths),
        "azimuth_step_deg": float(360.0 / n_azimuths),
        "R11_toy": float(mean_matrix[0, 0]),
        "R12_toy": float(mean_matrix[0, 1]),
        "R21_toy": float(mean_matrix[1, 0]),
        "R22_toy": float(mean_matrix[1, 1]),
        "R_blend_toy": float(0.5 * np.trace(mean_matrix)),
        "R_blend_toy_native": float(rotation_responses[0]),
        "R_blend_toy_azimuth_sd": float(
            rotation_responses.std(ddof=1) if n_azimuths > 1 else 0.0
        ),
        "latent_dx_arcsec": float(native_geometry["native_dx_arcsec"]),
        "latent_dy_arcsec": float(native_geometry["native_dy_arcsec"]),
        "latent_distance_arcsec": float(native_geometry["latent_distance_arcsec"]),
        "catalogue_minus_latent_distance_arcsec": float(
            native_geometry["catalogue_minus_latent_distance_arcsec"]
        ),
    }
    return pair, rotations


def failure_row(row: pd.Series, error: Exception) -> dict:
    return {
        **row_metadata(row),
        "success": False,
        "error_type": type(error).__name__,
        "error_message": str(error)[:500],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--n-shards", type=int, default=40)
    parser.add_argument("--g", type=float, default=0.05)
    parser.add_argument("--stamp", type=int, default=112)
    parser.add_argument("--fit-seed", type=int, default=42)
    parser.add_argument("--n-azimuths", type=int, default=8)
    parser.add_argument("--max-pairs", type=int, default=0)
    parser.add_argument("--output-feather", required=True)
    parser.add_argument("--output-rotations-feather", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()
    for path in (
        args.output_feather, args.output_rotations_feather, args.output_json,
    ):
        if os.path.exists(path):
            raise FileExistsError(f"refusing existing output {path}")
    if not 0 <= args.shard_index < args.n_shards:
        raise ValueError("shard index is outside [0, n_shards)")
    if args.g <= 0.0 or args.stamp < 48 or args.stamp % 2:
        raise ValueError("g must be positive and stamp must be even and >=48")
    if args.n_azimuths != 8:
        raise ValueError("this experiment requires exactly eight 45-degree azimuths")

    manifest = pd.read_feather(args.manifest)
    selected = manifest.loc[
        manifest.sample_id.to_numpy(np.int64) % args.n_shards == args.shard_index
    ].sort_values("sample_id", kind="mergesort")
    if args.max_pairs:
        selected = selected.iloc[:args.max_pairs]
    if selected.empty:
        raise RuntimeError(f"shard {args.shard_index} has no sampled pairs")

    rows = []
    rotation_rows = []
    errors = []
    for count, record in enumerate(selected.itertuples(index=False), start=1):
        row = pd.Series(record._asdict())
        try:
            pair, rotations = run_pair(
                row, args.g, args.stamp, args.fit_seed, args.n_azimuths,
            )
            rows.append(pair)
            rotation_rows.extend(rotations)
        except Exception as error:  # retain the exact sampled denominator
            rows.append(failure_row(row, error))
            errors.append({
                "sample_id": int(row.sample_id),
                "error_type": type(error).__name__,
                "message": str(error)[:500],
            })
        if count % 50 == 0 or count == len(selected):
            print(
                f"shard {args.shard_index}: {count}/{len(selected)}; "
                f"failures={len(errors)}",
                flush=True,
            )

    output = pd.DataFrame(rows).sort_values("sample_id", kind="mergesort")
    if output.sample_id.duplicated().any() or len(output) != len(selected):
        raise RuntimeError("shard output does not replay its sampled denominator")
    rotations = pd.DataFrame(rotation_rows).sort_values(
        ["sample_id", "rotation_index"], kind="mergesort",
    )
    expected_rotations = int(output.success.sum()) * args.n_azimuths
    if len(rotations) != expected_rotations:
        raise RuntimeError("rotation rows do not replay successful pairs")
    if rotations[["sample_id", "rotation_index"]].duplicated().any():
        raise RuntimeError("duplicate pair/rotation key")
    Path(args.output_feather).parent.mkdir(parents=True, exist_ok=True)
    output.reset_index(drop=True).to_feather(args.output_feather)
    rotations.reset_index(drop=True).to_feather(args.output_rotations_feather)
    payload = {
        "design": (
            "two-object noiseless Sersic+Moffat stamp; primary unsheared; "
            "secondary antithetically sheared along g1/g2; neighbour position "
            "rotated in eight 45-degree steps with morphologies fixed; pair "
            "R_blend=0.5*trace(mean response matrix)"
        ),
        "manifest": os.path.abspath(args.manifest),
        "shard_index": int(args.shard_index),
        "n_shards": int(args.n_shards),
        "g": float(args.g),
        "stamp": int(args.stamp),
        "pixel_scale_arcsec": float(PIX),
        "pixel_noise": 0.0,
        "fit_seed": int(args.fit_seed),
        "n_azimuths": int(args.n_azimuths),
        "azimuth_offsets_deg": azimuth_offsets_deg(args.n_azimuths).tolist(),
        "position_rotation": "secondary displacement only; source PAs fixed",
        "primary_sheared": False,
        "positions_sheared": False,
        "measurement_center": "true primary center",
        "n_selected": int(len(selected)),
        "n_success": int(output.success.sum()),
        "n_failure": int((~output.success).sum()),
        "n_successful_rotations": int(len(rotations)),
        "failures": errors[:100],
        "output_feather": os.path.abspath(args.output_feather),
        "output_rotations_feather": os.path.abspath(
            args.output_rotations_feather
        ),
    }
    with open(args.output_json, "x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    print("HALFSHEAR_NOISELESS_PAIR_TOY_SHARD_DONE", flush=True)


if __name__ == "__main__":
    main()
