"""Blend-aware detection-classifier preparation and response diagnostics.

This module owns only SBSI-side neighbour selection, feature preparation, and
validation statistics.  Image detection labels and truth catalogues remain
caller-supplied BlendEMU products.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.spatial import KDTree
import torch
import torch.nn.functional as F

from .coordinates import (
    ellipticity_from_axis_ratio_angle,
    spin2_angle_from_components,
)
from .forward_catalogue import REQUIRED_CONDITIONS, validate_input_catalogue
from .preprocessing import rescale
from .shear_map import apply_shear_to_ellipticity


BASE_DETECTION_FEATURES = (
    "Re_input_p_scaled",
    "Re_input_s_scaled",
    "r_input_p_scaled",
    "r_input_s_scaled",
    "sersic_n_input_p",
    "sersic_n_input_s",
    "distance_scaled",
)
AXIS_RATIO_FEATURES = (
    "axis_ratio_input_p",
    "axis_ratio_input_s",
)
PAIR_FRAME_SHAPE_FEATURES = (
    "e_parallel_p",
    "e_cross_p",
    "e_parallel_s",
    "e_cross_s",
)
SHEARED_ELLIPTICITY_FEATURES = (
    "e1_input_p",
    "e2_input_p",
    "e1_input_s",
    "e2_input_s",
) + PAIR_FRAME_SHAPE_FEATURES


def discordant_transition_loss(
    logits0: torch.Tensor,
    logitsg: torch.Tensor,
    labels0: torch.Tensor,
    labelsg: torch.Tensor,
    *,
    eligible: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, int]:
    """Conditional logistic loss for paired detection-label transitions.

    Conditional on exactly one leg being detected, ``sigmoid(logitsg-logits0)``
    is the model probability that the detected leg is the sheared one.  The
    optional eligibility mask excludes pairs whose supplied morphology did not
    change between legs.
    """

    if not (
        logits0.shape == logitsg.shape == labels0.shape == labelsg.shape
    ):
        raise ValueError("paired transition tensors must have equal shape")
    if eligible is None:
        eligible = torch.ones_like(labels0, dtype=torch.bool)
    elif eligible.shape != labels0.shape:
        raise ValueError("transition eligibility mask must match paired labels")
    discordant = eligible.bool() & (labels0.bool() != labelsg.bool())
    count = int(discordant.sum().detach().cpu())
    if not count:
        return (logits0.sum() + logitsg.sum()) * 0.0, 0
    target = (labelsg[discordant] > labels0[discordant]).to(logits0.dtype)
    loss = F.binary_cross_entropy_with_logits(
        logitsg[discordant] - logits0[discordant], target
    )
    return loss, count


@dataclass(frozen=True)
class NeighbourChoice:
    """One representative neighbour per requested primary."""

    primary_row: np.ndarray
    secondary_row: np.ndarray
    distance_arcsec: np.ndarray
    log_impact: np.ndarray
    candidate_count: np.ndarray


def log_neighbour_impact(
    secondary_magnitude: np.ndarray,
    secondary_size_arcsec: np.ndarray,
    distance_arcsec: np.ndarray,
    exponent: float,
) -> np.ndarray:
    """Log of ``flux_s * (Re_s / distance)**exponent`` up to a constant."""

    magnitude = np.asarray(secondary_magnitude, dtype=float)
    size = np.asarray(secondary_size_arcsec, dtype=float)
    distance = np.asarray(distance_arcsec, dtype=float)
    if not np.isfinite(exponent) or exponent < 0:
        raise ValueError("impact exponent must be finite and non-negative")
    if (
        not np.isfinite(magnitude).all()
        or not np.isfinite(size).all()
        or (size <= 0).any()
        or not np.isfinite(distance).all()
        or (distance <= 0).any()
    ):
        raise ValueError(
            "impact score requires finite magnitudes and positive finite sizes "
            "and separations"
        )
    return (
        -0.4 * np.log(10.0) * magnitude
        + float(exponent) * (np.log(size) - np.log(distance))
    )


def render_catalogue_shape_shear(
    catalogue,
    *,
    g1_column: str = "g1",
    g2_column: str = "g2",
) -> pd.DataFrame:
    """Fold per-row applied shear into axis ratio and position angle.

    BlendEMU truth files retain intrinsic ``axis_ratio``/``position_angle`` and
    store the rendered shear separately.  A classifier intended to respond via
    shape must be conditioned on the composed, rendered shape.
    """

    frame = validate_input_catalogue(catalogue)
    missing = sorted({g1_column, g2_column, "position_angle"} - set(frame.columns))
    if missing:
        raise KeyError(f"catalogue is missing shear-rendering columns: {missing}")
    e1, e2 = ellipticity_from_axis_ratio_angle(
        frame["axis_ratio"].to_numpy(dtype=float),
        frame["position_angle"].to_numpy(dtype=float),
    )
    sheared1, sheared2 = apply_shear_to_ellipticity(
        e1,
        e2,
        frame[g1_column].to_numpy(dtype=float),
        frame[g2_column].to_numpy(dtype=float),
    )
    amplitude = np.hypot(sheared1, sheared2)
    if (amplitude >= 1.0).any():
        raise ValueError("rendered ellipticity left the physical unit disc")
    frame["axis_ratio"] = (1.0 - amplitude) / (1.0 + amplitude)
    angle = spin2_angle_from_components(sheared1, sheared2)
    raw_angle = frame["position_angle"].to_numpy(dtype=float)
    finite = np.isfinite(raw_angle)
    uses_degrees = bool(
        finite.any()
        and np.nanpercentile(np.abs(raw_angle[finite]), 95) > 2 * np.pi * 1.1
    )
    frame["position_angle"] = np.rad2deg(angle) if uses_degrees else angle
    return frame


def choose_representative_neighbours(
    catalogue,
    primary_rows: Sequence[int],
    *,
    radius_arcsec: float,
    neighbour_selection: str = "nearest",
    impact_exponent: float = 2.0,
) -> NeighbourChoice:
    """Find all in-aperture candidates and select one without a k-neighbour cap."""

    if neighbour_selection not in {"nearest", "impact"}:
        raise ValueError("neighbour_selection must be 'nearest' or 'impact'")
    key = "nearest" if neighbour_selection == "nearest" else f"impact_a{impact_exponent:g}"
    return choose_neighbour_ladder(
        catalogue,
        primary_rows,
        radius_arcsec=radius_arcsec,
        impact_exponents=(impact_exponent,),
    )[key]


def choose_neighbour_ladder(
    catalogue,
    primary_rows: Sequence[int],
    *,
    radius_arcsec: float,
    impact_exponents: Sequence[float] = (1.0, 2.0, 4.0),
) -> dict[str, NeighbourChoice]:
    """Build nearest and several impact-ranked choices with one exact tree query."""

    galaxies = validate_input_catalogue(catalogue)
    primary = np.asarray(primary_rows, dtype=np.int64)
    if primary.ndim != 1 or ((primary < 0) | (primary >= len(galaxies))).any():
        raise ValueError("primary_rows are invalid")
    if len(np.unique(primary)) != len(primary):
        raise ValueError("primary_rows must be unique")
    if radius_arcsec <= 0:
        raise ValueError("radius_arcsec must be positive")
    exponents = tuple(float(value) for value in impact_exponents)
    if len(set(exponents)) != len(exponents):
        raise ValueError("impact exponents must be unique")
    for exponent in exponents:
        if not np.isfinite(exponent) or exponent < 0:
            raise ValueError("impact exponent must be finite and non-negative")

    positions = galaxies[["RA", "DEC"]].to_numpy(dtype=float)
    neighbours = KDTree(positions).query_ball_point(
        positions[primary], r=float(radius_arcsec) / 3600.0, workers=-1
    )
    keys = ("nearest",) + tuple(f"impact_a{value:g}" for value in exponents)
    selected = {key: np.full(len(primary), -1, dtype=np.int64) for key in keys}
    selected_distance = {
        key: np.full(len(primary), np.nan, dtype=float) for key in keys
    }
    selected_log_impact = {
        key: np.full(len(primary), np.nan, dtype=float) for key in keys
    }
    counts = np.zeros(len(primary), dtype=np.int64)
    magnitudes = galaxies["r"].to_numpy(dtype=float)
    sizes = galaxies["Re"].to_numpy(dtype=float)

    for output_row, (primary_row, candidate_value) in enumerate(
        zip(primary, neighbours)
    ):
        candidate = np.asarray(candidate_value, dtype=np.int64)
        candidate = candidate[candidate != primary_row]
        if not len(candidate):
            continue
        delta = (positions[candidate] - positions[primary_row]) * 3600.0
        distance = np.hypot(delta[:, 0], delta[:, 1])
        positive = distance > 0
        candidate, distance = candidate[positive], distance[positive]
        inside = distance < radius_arcsec
        candidate, distance = candidate[inside], distance[inside]
        counts[output_row] = len(candidate)
        if not len(candidate):
            continue
        nearest_order = np.lexsort((candidate, distance))
        nearest_winner = int(nearest_order[0])
        selected["nearest"][output_row] = candidate[nearest_winner]
        selected_distance["nearest"][output_row] = distance[nearest_winner]
        nearest_impact = log_neighbour_impact(
            magnitudes[candidate], sizes[candidate], distance, 2.0
        )
        selected_log_impact["nearest"][output_row] = nearest_impact[
            nearest_winner
        ]
        for exponent in exponents:
            key = f"impact_a{exponent:g}"
            impact = log_neighbour_impact(
                magnitudes[candidate], sizes[candidate], distance, exponent
            )
            order = np.lexsort((candidate, distance, -impact))
            winner = int(order[0])
            selected[key][output_row] = candidate[winner]
            selected_distance[key][output_row] = distance[winner]
            selected_log_impact[key][output_row] = impact[winner]

    return {
        key: NeighbourChoice(
            primary.copy(),
            selected[key],
            selected_distance[key],
            selected_log_impact[key],
            counts.copy(),
        )
        for key in keys
    }


def build_detection_feature_frame(
    catalogue,
    choice: NeighbourChoice,
    *,
    conditions: Mapping[str, float],
) -> pd.DataFrame:
    """Build a rescaled classifier table for a precomputed neighbour choice."""

    galaxies = validate_input_catalogue(catalogue)
    missing_conditions = sorted(set(REQUIRED_CONDITIONS) - set(conditions))
    if missing_conditions:
        raise KeyError(f"observing conditions are missing: {missing_conditions}")
    primary_rows = np.asarray(choice.primary_row, dtype=np.int64)
    secondary_rows = np.asarray(choice.secondary_row, dtype=np.int64)
    n = len(primary_rows)
    if secondary_rows.shape != (n,) or choice.distance_arcsec.shape != (n,):
        raise ValueError("neighbour choice arrays are not aligned")
    if ((primary_rows < 0) | (primary_rows >= len(galaxies))).any():
        raise ValueError("neighbour choice has an invalid primary row")
    matched = secondary_rows >= 0
    if (secondary_rows[matched] >= len(galaxies)).any():
        raise ValueError("neighbour choice has an invalid secondary row")

    frame = galaxies.iloc[primary_rows].reset_index(drop=True).add_suffix("_input_p")
    frame.insert(0, "primary_row", primary_rows)
    if "index" in galaxies:
        frame.insert(
            1,
            "input_index",
            galaxies.iloc[primary_rows]["index"].to_numpy(copy=True),
        )
    else:
        frame.insert(1, "input_index", primary_rows)
    frame["secondary_row"] = secondary_rows
    frame["neighbored"] = matched
    frame["distance"] = np.asarray(choice.distance_arcsec, dtype=float)
    frame["neighbour_log_impact"] = np.asarray(choice.log_impact, dtype=float)

    dx = np.full(n, np.nan, dtype=float)
    dy = np.full(n, np.nan, dtype=float)
    if matched.any():
        dx[matched] = (
            galaxies.iloc[secondary_rows[matched]]["RA"].to_numpy(dtype=float)
            - galaxies.iloc[primary_rows[matched]]["RA"].to_numpy(dtype=float)
        ) * 3600.0
        dy[matched] = (
            galaxies.iloc[secondary_rows[matched]]["DEC"].to_numpy(dtype=float)
            - galaxies.iloc[primary_rows[matched]]["DEC"].to_numpy(dtype=float)
        ) * 3600.0
    frame["polarization_angle"] = np.arctan2(dy, dx)

    for column in galaxies.columns:
        values = np.full(n, np.nan, dtype=object)
        if matched.any():
            values[matched] = galaxies.iloc[secondary_rows[matched]][column].to_numpy()
        frame[f"{column}_input_s"] = values
    for column in galaxies.select_dtypes(include=[np.number, "bool"]).columns:
        frame[f"{column}_input_s"] = pd.to_numeric(
            frame[f"{column}_input_s"], errors="coerce"
        )

    e1, e2 = ellipticity_from_axis_ratio_angle(
        galaxies.iloc[primary_rows]["axis_ratio"].to_numpy(dtype=float),
        galaxies.iloc[primary_rows]["position_angle"].to_numpy(dtype=float),
    )
    frame["e1_input_rot0_p"] = e1
    frame["e2_input_rot0_p"] = e2
    return rescale(
        frame.set_index("primary_row", drop=True),
        pixel_rms=float(conditions["pixel_rms"]),
        pixel_size=float(conditions["pixel_size"]),
        zero_mag=float(conditions["zero_point"]),
        psf_fwhm=float(conditions["psf_fwhm"]),
        moffat_beta=float(conditions["moffat_beta"]),
    )


def detection_selection_response(
    e0_parallel: np.ndarray,
    eg_parallel: np.ndarray,
    weights0: np.ndarray,
    weightsg: np.ndarray,
    shear: np.ndarray,
) -> float:
    """Centered forward detection-selection response for paired objects."""

    e0 = np.asarray(e0_parallel, dtype=float)
    eg = np.asarray(eg_parallel, dtype=float)
    w0 = np.asarray(weights0, dtype=float)
    wg = np.asarray(weightsg, dtype=float)
    g = np.asarray(shear, dtype=float)
    if not (e0.shape == eg.shape == w0.shape == wg.shape == g.shape):
        raise ValueError("paired response arrays must have equal shape")
    valid = (
        np.isfinite(e0)
        & np.isfinite(eg)
        & np.isfinite(w0)
        & np.isfinite(wg)
        & np.isfinite(g)
        & (g > 0)
        & (w0 >= 0)
        & (wg >= 0)
    )
    if not valid.any() or w0[valid].sum() <= 0 or wg[valid].sum() <= 0:
        raise ValueError("response population or selected weight is empty")

    def shift(shape, weight):
        return np.sum(weight * shape) / np.sum(weight) - np.mean(shape)

    nominal_g = float(np.median(g[valid]))
    if not np.allclose(g[valid], nominal_g, rtol=0, atol=1.0e-6):
        raise ValueError("response estimator requires one nonzero shear magnitude")
    return float(
        (shift(eg[valid], wg[valid]) - shift(e0[valid], w0[valid])) / nominal_g
    )


__all__ = [
    "AXIS_RATIO_FEATURES",
    "BASE_DETECTION_FEATURES",
    "PAIR_FRAME_SHAPE_FEATURES",
    "SHEARED_ELLIPTICITY_FEATURES",
    "NeighbourChoice",
    "build_detection_feature_frame",
    "choose_neighbour_ladder",
    "choose_representative_neighbours",
    "detection_selection_response",
    "discordant_transition_loss",
    "log_neighbour_impact",
    "render_catalogue_shape_shear",
]
