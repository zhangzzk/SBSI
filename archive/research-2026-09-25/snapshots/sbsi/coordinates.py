"""Coordinate helpers for SBS spin-2 quantities.

SBS uses the same canonical sky spin-2 basis as blendemu catalogues for
intrinsic ellipticity, relative-position projections, and shear features:

    (q1, q2) = q * (cos(2 theta), sin(2 theta)).
"""

from __future__ import annotations

import numpy as np


RAW_SHEAR_FEATURES = {
    "gamma1_input_p", "gamma2_input_p",
    "gamma1_input_s", "gamma2_input_s",
}
SKY_SHEAR_FEATURES = {
    "gamma1_sky_p", "gamma2_sky_p",
    "gamma1_sky_s", "gamma2_sky_s",
}
SHEAR_ALIGNMENT_FEATURES = {
    "gamma_parallel_p", "gamma_cross_p",
    "gamma_parallel_s", "gamma_cross_s",
}
SHEAR_FEATURES = RAW_SHEAR_FEATURES | SKY_SHEAR_FEATURES | SHEAR_ALIGNMENT_FEATURES
SHEAR_CONVENTION_COLUMN = "shear_component_convention"
SKY_SHEAR_CONVENTION = "sky_cos_sin"


def angle_to_radians(angle):
    """Return angles in radians, accepting either radians or degree-like input."""
    values = np.asarray(angle, dtype=float)
    finite = np.isfinite(values)
    if np.any(finite) and np.nanpercentile(np.abs(values[finite]), 95) > 2 * np.pi * 1.1:
        return np.deg2rad(values)
    return values


def spin2_components_from_angle(angle, amplitude=1.0):
    """Convert amplitude and orientation angle to canonical spin-2 components."""
    theta = angle_to_radians(angle)
    amp = np.asarray(amplitude, dtype=float)
    return amp * np.cos(2.0 * theta), amp * np.sin(2.0 * theta)


def spin2_angle_from_components(component1, component2):
    """Return the orientation half-angle for canonical spin-2 components."""
    return 0.5 * np.arctan2(component2, component1)


def ellipticity_from_axis_ratio_angle(axis_ratio, position_angle):
    """Convert axis ratio and position angle to canonical ellipticity components."""
    q = np.asarray(axis_ratio, dtype=float)
    eps = (1.0 - q) / (1.0 + q)
    return spin2_components_from_angle(position_angle, amplitude=eps)


def normalize_shear_component_convention(convention):
    if convention in {None, "", SKY_SHEAR_CONVENTION, "usual", "cos_sin"}:
        return SKY_SHEAR_CONVENTION
    raise ValueError(f"Unsupported shear component convention: {convention!r}")


def frame_shear_component_convention(frame, default=SKY_SHEAR_CONVENTION):
    if SHEAR_CONVENTION_COLUMN in frame.columns:
        values = frame[SHEAR_CONVENTION_COLUMN].dropna().unique()
        if len(values):
            return normalize_shear_component_convention(values[0])
    return normalize_shear_component_convention(default)


def blendemu_shear_to_sky(gamma1_input, gamma2_input, convention=SKY_SHEAR_CONVENTION):
    """Convert blendemu raw shear columns to the SBS canonical sky basis."""
    normalize_shear_component_convention(convention)
    return np.asarray(gamma1_input, dtype=float), np.asarray(gamma2_input, dtype=float)


def spin2_project_to_angle(component1, component2, angle):
    """Project canonical spin-2 components onto axes parallel/cross to ``angle``."""
    unit1, unit2 = spin2_components_from_angle(angle, amplitude=1.0)
    c1 = np.asarray(component1, dtype=float)
    c2 = np.asarray(component2, dtype=float)
    parallel = c1 * unit1 + c2 * unit2
    cross = -c1 * unit2 + c2 * unit1
    return parallel, cross


def add_sky_shear_components(frame, suffixes=("p", "s")):
    """Add canonical ``gamma*_sky`` columns from blendemu raw shear columns."""
    convention = frame_shear_component_convention(frame)
    for suffix in suffixes:
        raw1 = f"gamma1_input_{suffix}"
        raw2 = f"gamma2_input_{suffix}"
        if raw1 not in frame.columns or raw2 not in frame.columns:
            continue
        gamma1_sky, gamma2_sky = blendemu_shear_to_sky(
            frame[raw1].array,
            frame[raw2].array,
            convention=convention,
        )
        frame[f"gamma1_sky_{suffix}"] = gamma1_sky
        frame[f"gamma2_sky_{suffix}"] = gamma2_sky


def add_pair_aligned_spin2_components(
    frame,
    component_prefix,
    input1_template,
    input2_template,
    output_parallel_template,
    output_cross_template,
    suffixes=("p", "s"),
    angle_column="polarization_angle",
):
    """Add spin-2 components projected parallel/cross to the pair direction."""
    if angle_column not in frame.columns:
        return
    angle = frame[angle_column].array
    for suffix in suffixes:
        c1_name = input1_template.format(suffix=suffix)
        c2_name = input2_template.format(suffix=suffix)
        if c1_name not in frame.columns or c2_name not in frame.columns:
            continue
        parallel, cross = spin2_project_to_angle(frame[c1_name].array, frame[c2_name].array, angle)
        frame[output_parallel_template.format(prefix=component_prefix, suffix=suffix)] = parallel
        frame[output_cross_template.format(prefix=component_prefix, suffix=suffix)] = cross
