"""Population helpers for the fixed zero-shear measured-selection domain.

The domain is anchored once on the measured zero-shear leg.  It is not a
truth-property cut and it is never re-evaluated on a sheared leg.  The helper
functions here are intentionally small so catalogue preparation, training,
and tests share exactly the same strict inequalities and key matching.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from .shear_map import apply_shear_to_ellipticity


PIXEL_SCALE_ARCSEC = 0.2
ZERO_POINT = 30.0
MAG_AUTO_MAX = 25.8
FLUX_RADIUS_MIN_ARCSEC = 0.6
# This is the exact catalogue-coordinate boundary.  Deriving it with the
# binary-float division 0.6 / 0.2 yields 2.9999999999999996 and incorrectly
# admits stored FLUX_RADIUS == 3.000 despite the declared strict inequality.
FLUX_RADIUS_MIN_PIXELS = 3.0

KEY_COLUMNS = ("case", "input_index")
FLOW_FEATURES = (
    "e1_input_p",
    "e2_input_p",
    "sersic_n_input_p",
    "r_input_p",
    "circularized_Re_input_p",
    "nbr_flux_near",
    "nbr_flux_far",
    "nbr_flux_max",
)
FLOW_TARGETS = (
    "measured_ngmix_g1",
    "measured_ngmix_g2",
    "measured_flux_radius",
    "measured_flux_from_mag_auto",
)


def _require_columns(frame: pd.DataFrame, names: Iterable[str]) -> None:
    missing = sorted(set(names) - set(frame.columns))
    if missing:
        raise KeyError(f"catalogue is missing columns: {missing}")


def fixed_g0_anchor_mask(
    frame: pd.DataFrame,
    *,
    mag_max: float = MAG_AUTO_MAX,
    radius_min_pixels: float | None = FLUX_RADIUS_MIN_PIXELS,
) -> np.ndarray:
    """Return the strict fixed-cohort mask evaluated on the g=0 leg only.

    Detection is required when a ``detected`` column is present.  No truth
    magnitude, truth size, neighbour, or ellipticity cut is applied.
    """

    _require_columns(frame, ("measured_mag_auto", "measured_flux_radius"))
    mag = frame["measured_mag_auto"].to_numpy(dtype=float, copy=False)
    radius = frame["measured_flux_radius"].to_numpy(dtype=float, copy=False)
    keep = np.isfinite(mag) & np.isfinite(radius)
    keep &= mag < float(mag_max)
    if radius_min_pixels is not None:
        keep &= radius > float(radius_min_pixels)
    if "detected" in frame:
        keep &= frame["detected"].fillna(False).to_numpy(dtype=bool, copy=False)
    return keep


def packed_keys(case, input_index) -> np.ndarray:
    """Pack non-negative ``(case, input_index)`` pairs into unique uint64 keys."""

    case = np.asarray(case)
    index = np.asarray(input_index)
    if case.shape != index.shape:
        raise ValueError("case and input_index must have matching shapes")
    if not np.issubdtype(case.dtype, np.integer):
        if not np.isfinite(case).all() or not np.equal(case, np.floor(case)).all():
            raise ValueError("case must be finite integers")
    if not np.issubdtype(index.dtype, np.integer):
        if not np.isfinite(index).all() or not np.equal(index, np.floor(index)).all():
            raise ValueError("input_index must be finite integers")
    case64 = case.astype(np.int64, copy=False)
    index64 = index.astype(np.int64, copy=False)
    if np.any(case64 < 0) or np.any(case64 >= 2**32):
        raise ValueError("case values must fit unsigned 32 bits")
    if np.any(index64 < 0) or np.any(index64 >= 2**32):
        raise ValueError("input_index values must fit unsigned 32 bits")
    return (case64.astype(np.uint64) << np.uint64(32)) | index64.astype(np.uint64)


def anchor_membership(frame: pd.DataFrame, anchor_keys: np.ndarray) -> np.ndarray:
    """Match rows to a sorted, unique fixed-g0 anchor without fuzzy joins."""

    _require_columns(frame, KEY_COLUMNS)
    anchor = np.asarray(anchor_keys, dtype=np.uint64)
    if anchor.ndim != 1 or (len(anchor) > 1 and np.any(anchor[1:] <= anchor[:-1])):
        raise ValueError("anchor keys must be a sorted unique one-dimensional array")
    return np.isin(
        packed_keys(frame["case"].to_numpy(), frame["input_index"].to_numpy()),
        anchor,
        assume_unique=False,
    )


def response_primary_target_mask(input_index, total_input_count: int) -> np.ndarray:
    """Select the structural primary half used by BlendEMU response targets.

    This reproduces BlendEMU's generated-catalogue split by stable input ID.
    It is an inherited simulator role boundary, not a measured-shape or truth
    analysis cut.
    """

    index = np.asarray(input_index)
    if not np.issubdtype(index.dtype, np.integer):
        if not np.isfinite(index).all() or not np.equal(index, np.floor(index)).all():
            raise ValueError("input_index must contain finite integers")
    index = index.astype(np.int64, copy=False)
    total = int(total_input_count)
    if total <= 1:
        raise ValueError("total_input_count must exceed one")
    if np.any(index < 0) or np.any(index >= total):
        raise ValueError("input_index lies outside the generated catalogue")
    return index < (total // 2)


def matched_key_indices(
    left_case,
    left_index,
    right_case,
    right_index,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Return exact one-to-one key intersections and explicit drop counts."""

    left = packed_keys(left_case, left_index)
    right = packed_keys(right_case, right_index)
    if len(np.unique(left)) != len(left):
        raise ValueError("duplicate key in left catalogue")
    if len(np.unique(right)) != len(right):
        raise ValueError("duplicate key in right catalogue")
    _, li, ri = np.intersect1d(left, right, assume_unique=True, return_indices=True)
    order = np.argsort(left[li], kind="stable")
    li = li[order].astype(np.int64, copy=False)
    ri = ri[order].astype(np.int64, copy=False)
    return (
        li,
        ri,
        {
            "left_rows": int(len(left)),
            "right_rows": int(len(right)),
            "matched_rows": int(len(li)),
            "left_unmatched": int(len(left) - len(li)),
            "right_unmatched": int(len(right) - len(ri)),
        },
    )


def circularized_radius(frame: pd.DataFrame) -> np.ndarray:
    """Recover circularized intrinsic Re without imposing a size cut."""

    _require_columns(frame, ("Re_input_p", "e1_input_rot0_p", "e2_input_rot0_p"))
    re_major = frame["Re_input_p"].to_numpy(dtype=float, copy=False)
    if "axis_ratio_input_p" in frame:
        q = frame["axis_ratio_input_p"].to_numpy(dtype=float, copy=False)
    else:
        e1 = frame["e1_input_rot0_p"].to_numpy(dtype=float, copy=False)
        e2 = frame["e2_input_rot0_p"].to_numpy(dtype=float, copy=False)
        modulus = np.hypot(e1, e2)
        q = (1.0 - modulus) / (1.0 + modulus)
    with np.errstate(invalid="ignore"):
        return re_major * np.sqrt(q)


@dataclass(frozen=True)
class FlowRows:
    """Finite usable flow rows derived without any truth-domain selection."""

    case: np.ndarray
    input_index: np.ndarray
    context: np.ndarray
    target: np.ndarray
    gamma: np.ndarray
    usable_mask: np.ndarray


def make_flow_rows(frame: pd.DataFrame, *, zero_point: float = ZERO_POINT) -> FlowRows:
    """Build physical-flow arrays and reject only invalid measurement rows.

    The raw truth columns are used as conditions over their inherited parent
    support.  Applied catalogue shear is folded into intrinsic ellipticity.
    The returned ``usable_mask`` is a measurement-validity mask, not a truth
    or measured-selection cut.
    """

    required = (
        *KEY_COLUMNS,
        "e1_input_rot0_p",
        "e2_input_rot0_p",
        "gamma1_input_p",
        "gamma2_input_p",
        "sersic_n_input_p",
        "r_input_p",
        "Re_input_p",
        "nbr_flux_near",
        "nbr_flux_far",
        "nbr_flux_max",
        "measured_ngmix_g1",
        "measured_ngmix_g2",
        "measured_mag_auto",
        "measured_flux_radius",
    )
    _require_columns(frame, required)
    e1 = frame["e1_input_rot0_p"].to_numpy(dtype=float, copy=False)
    e2 = frame["e2_input_rot0_p"].to_numpy(dtype=float, copy=False)
    g1 = frame["gamma1_input_p"].to_numpy(dtype=float, copy=False)
    g2 = frame["gamma2_input_p"].to_numpy(dtype=float, copy=False)
    folded1, folded2 = apply_shear_to_ellipticity(e1, e2, g1, g2)
    context = np.column_stack(
        (
            folded1,
            folded2,
            frame["sersic_n_input_p"].to_numpy(dtype=float, copy=False),
            frame["r_input_p"].to_numpy(dtype=float, copy=False),
            circularized_radius(frame),
            frame["nbr_flux_near"].to_numpy(dtype=float, copy=False),
            frame["nbr_flux_far"].to_numpy(dtype=float, copy=False),
            frame["nbr_flux_max"].to_numpy(dtype=float, copy=False),
        )
    )
    mag = frame["measured_mag_auto"].to_numpy(dtype=float, copy=False)
    with np.errstate(over="ignore", invalid="ignore"):
        flux = 10.0 ** (-0.4 * (mag - float(zero_point)))
    target = np.column_stack(
        (
            frame["measured_ngmix_g1"].to_numpy(dtype=float, copy=False),
            frame["measured_ngmix_g2"].to_numpy(dtype=float, copy=False),
            frame["measured_flux_radius"].to_numpy(dtype=float, copy=False),
            flux,
        )
    )
    gamma = np.column_stack((g1, g2))
    usable = np.isfinite(context).all(axis=1)
    usable &= np.isfinite(target).all(axis=1)
    usable &= np.isfinite(gamma).all(axis=1)
    usable &= np.square(target[:, 0]) + np.square(target[:, 1]) < 1.0
    usable &= (target[:, 2] > 0.0) & (target[:, 3] > 0.0)
    if "detected" in frame:
        usable &= frame["detected"].fillna(False).to_numpy(dtype=bool, copy=False)
    return FlowRows(
        case=frame["case"].to_numpy(dtype=np.int32, copy=False)[usable],
        input_index=frame["input_index"].to_numpy(dtype=np.int64, copy=False)[usable],
        context=context[usable].astype(np.float32, copy=False),
        target=target[usable].astype(np.float64, copy=False),
        gamma=gamma[usable].astype(np.float64, copy=False),
        usable_mask=usable,
    )


def split_group_values(values, seed: int, validation_size: float) -> tuple[np.ndarray, np.ndarray]:
    """Match the historical grouped ShuffleSplit convention exactly."""

    groups = np.asarray(sorted(set(np.asarray(values).tolist())), dtype=np.int64)
    if groups.size < 2:
        raise ValueError("grouped split needs at least two distinct values")
    if not (0.0 < validation_size < 1.0):
        raise ValueError("validation_size must be in (0, 1)")
    n_validation = int(np.ceil(validation_size * groups.size))
    permutation = np.random.RandomState(int(seed)).permutation(groups.size)
    return groups[permutation[n_validation:]], groups[permutation[:n_validation]]


__all__ = [
    "FLUX_RADIUS_MIN_ARCSEC",
    "FLUX_RADIUS_MIN_PIXELS",
    "FLOW_FEATURES",
    "FLOW_TARGETS",
    "KEY_COLUMNS",
    "MAG_AUTO_MAX",
    "PIXEL_SCALE_ARCSEC",
    "ZERO_POINT",
    "FlowRows",
    "anchor_membership",
    "circularized_radius",
    "fixed_g0_anchor_mask",
    "make_flow_rows",
    "matched_key_indices",
    "packed_keys",
    "response_primary_target_mask",
    "split_group_values",
]
