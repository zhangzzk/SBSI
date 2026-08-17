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


def _canonical_shear_and_gradient(frame, grad_df, suffix):
    sky1 = f"gamma1_sky_{suffix}"
    sky2 = f"gamma2_sky_{suffix}"
    dsky1 = f"dPsel_dgamma1_sky_{suffix}"
    dsky2 = f"dPsel_dgamma2_sky_{suffix}"
    if {sky1, sky2}.issubset(frame.columns) and {dsky1, dsky2}.issubset(grad_df.columns):
        return (
            frame[sky1].to_numpy(dtype=float),
            frame[sky2].to_numpy(dtype=float),
            grad_df[dsky1].to_numpy(dtype=float),
            grad_df[dsky2].to_numpy(dtype=float),
        )

    raw1 = f"gamma1_input_{suffix}"
    raw2 = f"gamma2_input_{suffix}"
    draw1 = f"dPsel_dgamma1_input_{suffix}"
    draw2 = f"dPsel_dgamma2_input_{suffix}"
    if {raw1, raw2}.issubset(frame.columns) and {draw1, draw2}.issubset(grad_df.columns):
        convention = frame_shear_component_convention(frame)
        gamma1_sky, gamma2_sky = blendemu_shear_to_sky(
            frame[raw1].array,
            frame[raw2].array,
            convention=convention,
        )
        dgamma1_sky = grad_df[draw1].to_numpy(dtype=float)
        dgamma2_sky = grad_df[draw2].to_numpy(dtype=float)
        return gamma1_sky, gamma2_sky, dgamma1_sky, dgamma2_sky

    raise KeyError(
        f"Need either canonical sky shear columns or raw blendemu shear columns for suffix {suffix!r}"
    )


def add_shear_aligned_gradients(frame, grad_df, suffixes=("p", "s")):
    """Add canonical shear-component and shear-amplitude gradient diagnostics.

    The returned frame keeps the original gradient columns and adds:
    ``dPsel_dgamma1_sky_*``, ``dPsel_dgamma2_sky_*``,
    ``dPsel_dgamma_parallel_*``, and ``dPsel_dgamma_perp_*``.
    For models trained on raw blendemu shear columns, raw and sky components
    are identical under the SBSI/blendemu convention.
    """
    out = grad_df.copy()
    for suffix in suffixes:
        pair_parallel = f"dPsel_dgamma_parallel_{suffix}"
        pair_cross = f"dPsel_dgamma_cross_{suffix}"
        if pair_parallel in out.columns and pair_cross in out.columns:
            out[f"dPsel_dgamma_pair_parallel_{suffix}"] = out[pair_parallel]
            out[f"dPsel_dgamma_pair_cross_{suffix}"] = out[pair_cross]
            # Compatibility for older plotting code. For pair-frame models,
            # "perp" here is the pair-cross spin-2 component.
            out[f"dPsel_dgamma_perp_{suffix}"] = out[pair_cross]
            par_name = f"gamma_parallel_{suffix}"
            cross_name = f"gamma_cross_{suffix}"
            if {par_name, cross_name}.issubset(frame.columns):
                out[f"gamma_abs_{suffix}"] = np.hypot(
                    frame[par_name].to_numpy(dtype=float),
                    frame[cross_name].to_numpy(dtype=float),
                )
            continue

        pair_parallel_blend = f"dPsel_dgamma_parallel_{suffix}_blend"
        pair_cross_blend = f"dPsel_dgamma_cross_{suffix}_blend"
        if pair_parallel_blend in out.columns and pair_cross_blend in out.columns:
            if "neighbored" in frame.columns:
                factor = frame["neighbored"].astype(float).to_numpy()
            else:
                factor = np.ones(len(frame), dtype=float)
            parallel = out[pair_parallel_blend].to_numpy(dtype=float) * factor
            cross = out[pair_cross_blend].to_numpy(dtype=float) * factor
            out[f"dPsel_dgamma_parallel_{suffix}"] = parallel
            out[f"dPsel_dgamma_cross_{suffix}"] = cross
            out[f"dPsel_dgamma_pair_parallel_{suffix}"] = parallel
            out[f"dPsel_dgamma_pair_cross_{suffix}"] = cross
            out[f"dPsel_dgamma_perp_{suffix}"] = cross
            par_name = f"gamma_parallel_{suffix}"
            cross_name = f"gamma_cross_{suffix}"
            if {par_name, cross_name}.issubset(frame.columns):
                out[f"gamma_abs_{suffix}"] = np.hypot(
                    frame[par_name].to_numpy(dtype=float),
                    frame[cross_name].to_numpy(dtype=float),
                )
            continue

        pframe_parallel = f"dPsel_dgamma_pframe_parallel_{suffix}"
        pframe_cross = f"dPsel_dgamma_pframe_cross_{suffix}"
        if pframe_parallel in out.columns and pframe_cross in out.columns:
            out[f"dPsel_dgamma_parallel_{suffix}"] = out[pframe_parallel]
            out[f"dPsel_dgamma_cross_{suffix}"] = out[pframe_cross]
            out[f"dPsel_dgamma_pair_parallel_{suffix}"] = out[pframe_parallel]
            out[f"dPsel_dgamma_pair_cross_{suffix}"] = out[pframe_cross]
            out[f"dPsel_dgamma_perp_{suffix}"] = out[pframe_cross]
            par_name = f"gamma_pframe_parallel_{suffix}"
            cross_name = f"gamma_pframe_cross_{suffix}"
            if {par_name, cross_name}.issubset(frame.columns):
                out[f"gamma_abs_{suffix}"] = np.hypot(
                    frame[par_name].to_numpy(dtype=float),
                    frame[cross_name].to_numpy(dtype=float),
                )
            continue

        pframe_parallel_blend = f"dPsel_dgamma_pframe_parallel_{suffix}_blend"
        pframe_cross_blend = f"dPsel_dgamma_pframe_cross_{suffix}_blend"
        if pframe_parallel_blend in out.columns and pframe_cross_blend in out.columns:
            if "neighbored" in frame.columns:
                factor = frame["neighbored"].astype(float).to_numpy()
            else:
                factor = np.ones(len(frame), dtype=float)
            parallel = out[pframe_parallel_blend].to_numpy(dtype=float) * factor
            cross = out[pframe_cross_blend].to_numpy(dtype=float) * factor
            out[f"dPsel_dgamma_parallel_{suffix}"] = parallel
            out[f"dPsel_dgamma_cross_{suffix}"] = cross
            out[f"dPsel_dgamma_pair_parallel_{suffix}"] = parallel
            out[f"dPsel_dgamma_pair_cross_{suffix}"] = cross
            out[f"dPsel_dgamma_perp_{suffix}"] = cross
            par_name = f"gamma_pframe_parallel_{suffix}"
            cross_name = f"gamma_pframe_cross_{suffix}"
            if {par_name, cross_name}.issubset(frame.columns):
                out[f"gamma_abs_{suffix}"] = np.hypot(
                    frame[par_name].to_numpy(dtype=float),
                    frame[cross_name].to_numpy(dtype=float),
                )
            continue

        gamma1_sky, gamma2_sky, dgamma1_sky, dgamma2_sky = _canonical_shear_and_gradient(
            frame, grad_df, suffix
        )
        gmag = np.hypot(gamma1_sky, gamma2_sky)
        good = np.isfinite(gmag) & (gmag > 1.0e-8)
        unit1 = np.full(len(gmag), np.nan)
        unit2 = np.full(len(gmag), np.nan)
        unit1[good] = gamma1_sky[good] / gmag[good]
        unit2[good] = gamma2_sky[good] / gmag[good]

        out[f"gamma_abs_{suffix}"] = gmag
        out[f"dPsel_dgamma1_sky_{suffix}"] = dgamma1_sky
        out[f"dPsel_dgamma2_sky_{suffix}"] = dgamma2_sky
        out[f"dPsel_dgamma_parallel_{suffix}"] = dgamma1_sky * unit1 + dgamma2_sky * unit2
        out[f"dPsel_dgamma_perp_{suffix}"] = -dgamma1_sky * unit2 + dgamma2_sky * unit1
    return out


def perturb_frame_along_sky_shear(frame, suffix, step, feature_names=None):
    """Return a copy with canonical shear amplitude shifted by ``step``.

    This supports both future canonical-feature models and existing models
    trained directly on blendemu raw shear columns.
    """
    out = frame.copy()
    sky1 = f"gamma1_sky_{suffix}"
    sky2 = f"gamma2_sky_{suffix}"
    raw1 = f"gamma1_input_{suffix}"
    raw2 = f"gamma2_input_{suffix}"

    available_features = set(out.columns if feature_names is None else feature_names)

    if {sky1, sky2}.issubset(available_features):
        gamma1_sky = out[sky1].to_numpy(dtype=float)
        gamma2_sky = out[sky2].to_numpy(dtype=float)
        target1 = sky1
        target2 = sky2
    elif {raw1, raw2}.issubset(available_features):
        convention = frame_shear_component_convention(out)
        gamma1_sky, gamma2_sky = blendemu_shear_to_sky(
            out[raw1].array,
            out[raw2].array,
            convention=convention,
        )
        target1 = raw1
        target2 = raw2
    else:
        raise KeyError(f"Missing shear columns for suffix {suffix!r}")

    gmag = np.hypot(gamma1_sky, gamma2_sky)
    good = np.isfinite(gmag) & (gmag > 1.0e-8)
    unit1 = np.zeros(len(out), dtype=float)
    unit2 = np.zeros(len(out), dtype=float)
    unit1[good] = gamma1_sky[good] / gmag[good]
    unit2[good] = gamma2_sky[good] / gmag[good]

    out[target1] = out[target1].to_numpy(dtype=float) + step * unit1
    out[target2] = out[target2].to_numpy(dtype=float) + step * unit2
    return out
