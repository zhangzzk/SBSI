"""Historical V2.1 truth-level population cuts for archived anchor products."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import domain


@dataclass(frozen=True)
class PairPopulationCuts:
    primary_mag: tuple[float, float] = (18.0, 28.0)
    primary_re: tuple[float, float] = (0.1, 1.5)
    secondary_mag: tuple[float, float] = (18.0, 28.0)
    secondary_re: tuple[float, float] = (0.1, 1.5)
    separation: tuple[float, float] = (0.0, 10.0)


LSST_PAIR_CUTS = PairPopulationCuts()
EXTENDED_PAIR_CUTS = PairPopulationCuts(
    secondary_mag=(13.0, 29.0),
    secondary_re=(0.0, 10.0),
    separation=(0.0, 10.0),
)

REQUIRED_COLUMNS = (
    "r_input_p", "Re_input_p", "r_input_s", "Re_input_s", "distance",
)


def _strict_between(values, limits):
    x = np.asarray(values, dtype=float)
    return np.isfinite(x) & (x > limits[0]) & (x < limits[1])


def primary_mask(frame, cuts: PairPopulationCuts = LSST_PAIR_CUTS):
    """Historical primary support box intersected with the V2.1 domain."""
    for col in ("r_input_p", "Re_input_p"):
        if col not in frame.columns:
            raise KeyError(f"primary_mask requires {col!r}")
    mag = frame["r_input_p"].to_numpy(dtype=float)
    re = frame["Re_input_p"].to_numpy(dtype=float)
    return (
        _strict_between(mag, cuts.primary_mag)
        & _strict_between(re, cuts.primary_re)
        & domain.in_domain(mag, re)
    )


def pair_mask(frame, cuts: PairPopulationCuts = LSST_PAIR_CUTS):
    """Historical intrinsic primary+secondary+separation pair mask."""
    missing = [col for col in REQUIRED_COLUMNS if col not in frame.columns]
    if missing:
        raise KeyError(f"pair_mask missing required columns: {missing}")
    return (
        primary_mask(frame, cuts=cuts)
        & _strict_between(frame["r_input_s"].to_numpy(dtype=float), cuts.secondary_mag)
        & _strict_between(frame["Re_input_s"].to_numpy(dtype=float), cuts.secondary_re)
        & _strict_between(frame["distance"].to_numpy(dtype=float), cuts.separation)
    )


def describe(cuts: PairPopulationCuts = LSST_PAIR_CUTS):
    return (
        f"primary: {cuts.primary_mag[0]}<r_p<{cuts.primary_mag[1]}, "
        f"{cuts.primary_re[0]}<Re_p<{cuts.primary_re[1]} arcsec, "
        f"Re_p>{domain.V21_RE_MIN} arcsec, intrinsic S/N>{domain.V21_SN_MIN}; "
        f"secondary: {cuts.secondary_mag[0]}<r_s<{cuts.secondary_mag[1]}, "
        f"{cuts.secondary_re[0]}<Re_s<{cuts.secondary_re[1]} arcsec; "
        f"{cuts.separation[0]}<separation<{cuts.separation[1]} arcsec"
    )
