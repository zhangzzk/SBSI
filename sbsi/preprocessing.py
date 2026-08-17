"""Preprocessing utilities for standalone SBSI calibration models."""

from __future__ import annotations

import numpy as np

from .coordinates import (
    RAW_SHEAR_FEATURES,
    SHEAR_ALIGNMENT_FEATURES,
    SHEAR_FEATURES,
    SKY_SHEAR_FEATURES,
    add_pair_aligned_spin2_components,
    add_sky_shear_components,
    angle_to_radians,
    ellipticity_from_axis_ratio_angle,
)


DEFAULT_CUTS = [[18, 28], [18, 28], [0.1, 1.5], [0.1, 1.5], [0, 5]]
DEFAULT_SELECTION_CUTS = DEFAULT_CUTS


def apply_structure_measurement_noise(frame, photoz_sigma=0.0, sersic_frac=0.0, rng=None, seed=0):
    """Emulate survey measurement error on the shear-EVEN structure conditioners.

    The realistic-conditioning study (WORKLOG cont.24) uses TRUE sersic_n / redshift as
    stand-ins for a survey photo-z and profile fit. This adds Gaussian measurement noise
    so we can bracket m between the perfect-structure (V1) and no-structure (V2) limits.

    - photoz_sigma: additive noise on redshift_input_p with sigma = photoz_sigma*(1+z),
      the standard photo-z scatter parametrisation; clipped to z>=0.
    - sersic_frac:  fractional noise on sersic_n_input_p with sigma = sersic_frac*|n|,
      clipped to the physical sersic range [0.3, 8.0].

    Both quantities are shear-even, so noising them once (held fixed under the +/-delta shear
    perturbation) is consistent with holding a MEASURED value fixed at deployment. Applied
    identically in training and validation so the flow is fed the same noisy channel it learned.
    Returns the frame with columns overwritten in place.
    """
    if rng is None:
        rng = np.random.default_rng(seed)
    if photoz_sigma and "redshift_input_p" in frame.columns:
        z = frame["redshift_input_p"].to_numpy(float)
        frame["redshift_input_p"] = np.clip(z + rng.normal(0.0, photoz_sigma * (1.0 + z)), 0.0, None)
    if sersic_frac and "sersic_n_input_p" in frame.columns:
        n = frame["sersic_n_input_p"].to_numpy(float)
        frame["sersic_n_input_p"] = np.clip(n + rng.normal(0.0, sersic_frac * np.abs(n)), 0.3, 8.0)
    return frame
SHAPE_FEATURES = {
    "e1_input_p", "e2_input_p",
    "e1_input_s", "e2_input_s",
}
RELATIVE_POSITION_FEATURES = {
    "relative_position_angle_cos",
    "relative_position_angle_sin",
    "relative_position_angle_cos2",
    "relative_position_angle_sin2",
}
RELATIVE_ALIGNMENT_FEATURES = {
    "e_parallel_p", "e_cross_p",
    "e_parallel_s", "e_cross_s",
}
PRIMARY_FRAME_FEATURES = {
    "e_abs_p",
    "e_abs_s",
    "e_pframe_parallel_p",
    "e_pframe_cross_p",
    "e_pframe_parallel_s",
    "e_pframe_cross_s",
    "gamma_pframe_parallel_p",
    "gamma_pframe_cross_p",
    "gamma_pframe_parallel_s",
    "gamma_pframe_cross_s",
    "pair_pframe_cos2",
    "pair_pframe_sin2",
}
NEIGHBOR_GATED_BASE_FEATURES = [
    "distance_scaled",
    "Re_input_s_scaled",
    "r_input_s_scaled",
    "flux_ratio",
    "sersic_n_input_s",
    "e_parallel_s",
    "e_cross_s",
    "gamma_parallel_s",
    "gamma_cross_s",
    "e_pframe_parallel_s",
    "e_pframe_cross_s",
    "gamma_pframe_parallel_s",
    "gamma_pframe_cross_s",
    "pair_pframe_cos2",
    "pair_pframe_sin2",
    "redshift_input_s",
]
NEIGHBOR_GATED_FEATURES = {f"{name}_blend" for name in NEIGHBOR_GATED_BASE_FEATURES}


def shear_label(g: float | str) -> str:
    """Compact shear label matching blendemu simulation directory names."""
    label = f"{float(g):.3f}".rstrip("0").rstrip(".")
    if label == "-0":
        label = "0"
    if "." not in label:
        label += ".0"
    return label


def source_select_detection(dataset, cuts=None):
    """Apply the current SExtractor-detection pilot cuts."""
    if cuts is None:
        cuts = DEFAULT_CUTS
    idx_sel = np.where(
        (dataset["r_input_p"] > cuts[1][0]) & (dataset["r_input_p"] < cuts[1][1])
        & (dataset["Re_input_p"] > cuts[3][0]) & (dataset["Re_input_p"] < cuts[3][1])
        & (((dataset["distance"] > cuts[4][0]) & (dataset["distance"] < cuts[4][1]))
           | (~dataset["neighbored"]))
    )[0]
    return dataset.iloc[idx_sel].reset_index(drop=True)


def source_select_selection(dataset, cuts=None):
    """Apply the current selection-model cuts.

    The first selection target is SExtractor detection, so this aliases the
    existing detection pilot cuts.
    """
    return source_select_detection(dataset, cuts=cuts)


def mag2flux(mag, zero_mag):
    return 10 ** (-0.4 * (mag - zero_mag))


def flux2mag(flux, zero_mag):
    return -2.5 * np.log10(flux) + zero_mag


def _angle_to_radians(angle):
    return angle_to_radians(angle)


def _ellipticity_from_axis_ratio_angle(axis_ratio, position_angle):
    return ellipticity_from_axis_ratio_angle(axis_ratio, position_angle)


def _add_shape_components(dataset):
    for suffix in ("p", "s"):
        e1_name = f"e1_input_{suffix}"
        e2_name = f"e2_input_{suffix}"
        rot_e1_name = f"e1_input_rot0_{suffix}"
        rot_e2_name = f"e2_input_rot0_{suffix}"
        if rot_e1_name in dataset.columns and rot_e2_name in dataset.columns:
            dataset[e1_name] = dataset[rot_e1_name]
            dataset[e2_name] = dataset[rot_e2_name]
            continue

        axis_ratio_name = f"axis_ratio_input_{suffix}"
        angle_name = f"position_angle_input_{suffix}"
        if axis_ratio_name in dataset.columns and angle_name in dataset.columns:
            dataset[e1_name], dataset[e2_name] = _ellipticity_from_axis_ratio_angle(
                dataset[axis_ratio_name].array,
                dataset[angle_name].array,
            )


def _add_relative_position_features(dataset):
    if "polarization_angle" not in dataset.columns:
        return

    phi = _angle_to_radians(dataset["polarization_angle"].array)
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)
    cos_2phi = np.cos(2.0 * phi)
    sin_2phi = np.sin(2.0 * phi)

    dataset["relative_position_angle_cos"] = cos_phi
    dataset["relative_position_angle_sin"] = sin_phi
    dataset["relative_position_angle_cos2"] = cos_2phi
    dataset["relative_position_angle_sin2"] = sin_2phi

    for suffix in ("p", "s"):
        e1_name = f"e1_input_{suffix}"
        e2_name = f"e2_input_{suffix}"
        if e1_name not in dataset.columns or e2_name not in dataset.columns:
            continue
        e1 = np.asarray(dataset[e1_name], dtype=float)
        e2 = np.asarray(dataset[e2_name], dtype=float)
        dataset[f"e_parallel_{suffix}"] = e1 * cos_2phi + e2 * sin_2phi
        dataset[f"e_cross_{suffix}"] = -e1 * sin_2phi + e2 * cos_2phi

    add_pair_aligned_spin2_components(
        dataset,
        component_prefix="gamma",
        input1_template="gamma1_sky_{suffix}",
        input2_template="gamma2_sky_{suffix}",
        output_parallel_template="{prefix}_parallel_{suffix}",
        output_cross_template="{prefix}_cross_{suffix}",
    )


def _project_spin2_to_primary_frame(dataset, c1_name, c2_name, out_parallel, out_cross, unit1, unit2):
    if c1_name not in dataset.columns or c2_name not in dataset.columns:
        return
    c1 = np.asarray(dataset[c1_name], dtype=float)
    c2 = np.asarray(dataset[c2_name], dtype=float)
    dataset[out_parallel] = c1 * unit1 + c2 * unit2
    dataset[out_cross] = -c1 * unit2 + c2 * unit1


def _add_primary_frame_features(dataset):
    if "e1_input_p" not in dataset.columns or "e2_input_p" not in dataset.columns:
        return

    e1_p = np.asarray(dataset["e1_input_p"], dtype=float)
    e2_p = np.asarray(dataset["e2_input_p"], dtype=float)
    e_abs = np.hypot(e1_p, e2_p)
    good = np.isfinite(e_abs) & (e_abs > 1.0e-8)
    unit1 = np.ones(len(dataset), dtype=float)
    unit2 = np.zeros(len(dataset), dtype=float)
    unit1[good] = e1_p[good] / e_abs[good]
    unit2[good] = e2_p[good] / e_abs[good]

    dataset["e_abs_p"] = np.where(np.isfinite(e_abs), e_abs, 0.0)
    dataset["e_pframe_parallel_p"] = dataset["e_abs_p"]
    dataset["e_pframe_cross_p"] = 0.0

    if "e1_input_s" in dataset.columns and "e2_input_s" in dataset.columns:
        e1_s = np.asarray(dataset["e1_input_s"], dtype=float)
        e2_s = np.asarray(dataset["e2_input_s"], dtype=float)
        e_abs_s = np.hypot(e1_s, e2_s)
        dataset["e_abs_s"] = np.where(np.isfinite(e_abs_s), e_abs_s, 0.0)

    _project_spin2_to_primary_frame(
        dataset,
        "e1_input_s",
        "e2_input_s",
        "e_pframe_parallel_s",
        "e_pframe_cross_s",
        unit1,
        unit2,
    )

    for suffix in ("p", "s"):
        _project_spin2_to_primary_frame(
            dataset,
            f"gamma1_sky_{suffix}",
            f"gamma2_sky_{suffix}",
            f"gamma_pframe_parallel_{suffix}",
            f"gamma_pframe_cross_{suffix}",
            unit1,
            unit2,
        )

    if "polarization_angle" in dataset.columns:
        phi = _angle_to_radians(dataset["polarization_angle"].array)
        cos_2phi = np.cos(2.0 * phi)
        sin_2phi = np.sin(2.0 * phi)
        dataset["pair_pframe_cos2"] = cos_2phi * unit1 + sin_2phi * unit2
        dataset["pair_pframe_sin2"] = -cos_2phi * unit2 + sin_2phi * unit1


def _add_neighbor_gated_features(dataset):
    if "neighbored" not in dataset.columns:
        return
    active = dataset["neighbored"].astype(bool).to_numpy()
    for base_name in NEIGHBOR_GATED_BASE_FEATURES:
        if base_name not in dataset.columns:
            continue
        values = np.asarray(dataset[base_name], dtype=float)
        dataset[f"{base_name}_blend"] = np.where(active & np.isfinite(values), values, 0.0)


def raw_columns_for_selection_features(features, available_columns=None):
    """Return catalogue columns needed before SBS feature engineering.

    ``available_columns`` lets callers prefer stored ``e1/e2`` columns when a
    catalogue has them, while still supporting older catalogues that only carry
    axis ratio and position angle.
    """
    available = None if available_columns is None else set(available_columns)
    columns = {
        "neighbored",
        "distance",
        "Re_input_p", "Re_input_s",
        "r_input_p", "r_input_s",
    }
    if available is not None and "shear_component_convention" in available:
        columns.add("shear_component_convention")

    def add_shape_raw(suffix):
        e1 = f"e1_input_{suffix}"
        e2 = f"e2_input_{suffix}"
        rot_e1 = f"e1_input_rot0_{suffix}"
        rot_e2 = f"e2_input_rot0_{suffix}"
        if available is not None and {e1, e2}.issubset(available):
            columns.update([e1, e2])
        elif available is not None and {rot_e1, rot_e2}.issubset(available):
            columns.update([rot_e1, rot_e2])
        else:
            columns.update([f"axis_ratio_input_{suffix}", f"position_angle_input_{suffix}"])

    for name in features:
        if name in NEIGHBOR_GATED_FEATURES:
            columns.update(raw_columns_for_selection_features([name[:-len("_blend")]], available_columns=available))
            columns.add("neighbored")
        elif name.endswith("_scaled"):
            if name.startswith("Re_input_p"):
                columns.add("Re_input_p")
            elif name.startswith("Re_input_s"):
                columns.add("Re_input_s")
            elif name.startswith("r_input_p"):
                columns.add("r_input_p")
            elif name.startswith("r_input_s"):
                columns.add("r_input_s")
            elif name == "distance_scaled":
                columns.update(["distance", "Re_input_p"])
            else:
                raise ValueError(f"Unknown scaled feature {name!r}")
        elif name == "flux_ratio":
            columns.update(["r_input_p", "r_input_s"])
        elif name in {"e1_input_p", "e2_input_p"}:
            add_shape_raw("p")
        elif name in {"e1_input_s", "e2_input_s"}:
            add_shape_raw("s")
        elif name in RELATIVE_POSITION_FEATURES:
            columns.add("polarization_angle")
        elif name in {"e_parallel_p", "e_cross_p"}:
            add_shape_raw("p")
            columns.add("polarization_angle")
        elif name in {"e_parallel_s", "e_cross_s"}:
            add_shape_raw("s")
            columns.add("polarization_angle")
        elif name in PRIMARY_FRAME_FEATURES:
            if name == "e_abs_p" or name in {"e_pframe_parallel_p", "e_pframe_cross_p"}:
                add_shape_raw("p")
            elif name == "e_abs_s":
                add_shape_raw("s")
            elif name in {"e_pframe_parallel_s", "e_pframe_cross_s"}:
                add_shape_raw("p")
                add_shape_raw("s")
            elif name.startswith("gamma_pframe_"):
                suffix = name.rsplit("_", 1)[-1]
                add_shape_raw("p")
                columns.update([f"gamma1_input_{suffix}", f"gamma2_input_{suffix}"])
            elif name in {"pair_pframe_cos2", "pair_pframe_sin2"}:
                add_shape_raw("p")
                columns.add("polarization_angle")
        elif name in RAW_SHEAR_FEATURES:
            columns.add(name)
        elif name in SKY_SHEAR_FEATURES:
            suffix = name.rsplit("_", 1)[-1]
            columns.update([f"gamma1_input_{suffix}", f"gamma2_input_{suffix}"])
        elif name in SHEAR_ALIGNMENT_FEATURES:
            suffix = name.rsplit("_", 1)[-1]
            columns.update([f"gamma1_input_{suffix}", f"gamma2_input_{suffix}", "polarization_angle"])
        else:
            columns.add(name)

    return columns


def moffat_fwhm2re(fwhm, beta):
    factor = np.sqrt((2 ** (1 / (beta - 1)) - 1) / (2 ** (1 / beta) - 1)) / 2
    return fwhm * factor


def _convolved_size(r, psf_size):
    return np.sqrt(np.power(r, 2) + np.power(psf_size, 2))


def rescale(
    dataset,
    pixel_rms=0.312,
    pixel_size=0.2,
    zero_mag=30,
    psf_fwhm=0.73,
    moffat_beta=2.224,
):
    """
    Rescale blendemu-style features into approximately survey-independent units.

    This mirrors the feature engineering used by blendemu, but is kept local so
    SBS can stay independent of the blendemu Python package.
    """
    dataset = dataset.copy()
    psf_size = moffat_fwhm2re(psf_fwhm, moffat_beta)
    aperture_rms = pixel_rms * (psf_size / pixel_size) ** 2 * np.pi

    post_re_p = _convolved_size(dataset["Re_input_p"].array, psf_size)
    post_re_s = _convolved_size(dataset["Re_input_s"].array, psf_size)

    dataset["distance_scaled"] = dataset["distance"] / post_re_p
    dataset["Re_input_p_scaled"] = dataset["Re_input_p"] / post_re_p
    dataset["Re_input_s_scaled"] = dataset["Re_input_s"] / post_re_s

    flux_input_p = mag2flux(dataset["r_input_p"], zero_mag)
    flux_input_s = mag2flux(dataset["r_input_s"], zero_mag)
    dataset["flux_ratio"] = np.log10(flux_input_p / flux_input_s)

    dataset["r_input_p_scaled"] = flux2mag(flux_input_p / aperture_rms, zero_mag)
    dataset["r_input_s_scaled"] = flux2mag(flux_input_s / aperture_rms, zero_mag)
    _add_shape_components(dataset)
    add_sky_shear_components(dataset)
    _add_relative_position_features(dataset)
    _add_primary_frame_features(dataset)
    _add_neighbor_gated_features(dataset)
    return dataset


def intrinsic_e_abs(axis_ratio):
    """|e_intrinsic| in the blendemu `angle2e` convention.

    That convention is e = (1-q)/(1+q) * (cos 2theta, sin 2theta), so the MAGNITUDE depends only on
    the axis ratio -- the position angle cancels. Lives here, rather than in either caller, because
    the per-object response target (scripts/build_perobj_target.py) is FIT on this feature and the
    trainer must DERIVE the identical quantity on the g=0 catalogue, which carries `axis_ratio_input_p`
    but no `e_abs` column. Two private copies of this formula drifting apart would silently corrupt
    the supervision signal with nothing raising.
    """
    q = np.asarray(axis_ratio, dtype=float)
    return (1.0 - q) / (1.0 + q)
