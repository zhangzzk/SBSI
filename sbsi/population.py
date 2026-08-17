"""Truth-level population cuts for SBSI catalogue comparisons.

These cuts define which simulated scenes belong to the data domain.  They do
not inspect detection state or any measured quantity.  Keeping this separate
from :func:`sbsi.preprocessing.source_select_selection` is deliberate:
that function carries the historical 5-arcsec selection-model convention,
whereas a comparison of the flow half-shear, R_blend half-shear, and constgold
must apply one identical intrinsic mask to all three catalogues.

The standard LSST pair support mirrors ``blendemu/configs/fs2_lsst_r.yaml``::

    [r_s, r_p, Re_s, Re_p, separation]
      [18,28], [18,28], [0.1,1.5], [0.1,1.5], [0,10]

The caller may intersect the primary with an explicit trained-model domain.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .domain import Domain


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


def primary_mask(
    frame,
    domain: Domain = None,
    cuts: PairPopulationCuts = LSST_PAIR_CUTS,
):
    """Primary support box, optionally intersected with a model domain."""
    for col in ("r_input_p", "Re_input_p"):
        if col not in frame.columns:
            raise KeyError(f"primary_mask requires {col!r}")
    mag = frame["r_input_p"].to_numpy(dtype=float)
    re = frame["Re_input_p"].to_numpy(dtype=float)
    mask = (
        _strict_between(mag, cuts.primary_mag)
        & _strict_between(re, cuts.primary_re)
    )
    if domain is not None:
        if not isinstance(domain, Domain):
            raise TypeError("domain must be a Domain or None")
        mask &= domain.mask(mag, re)
    return mask


def pair_mask(frame, domain: Domain = None, cuts: PairPopulationCuts = LSST_PAIR_CUTS):
    """One identical intrinsic primary+secondary+separation mask.

    Isolated rows have NaN secondary properties and therefore fail.  This is
    the same semantics as BlendEMU's regression pair selection: this function
    defines a pair population, not a detection or isolated-galaxy population.
    """
    missing = [col for col in REQUIRED_COLUMNS if col not in frame.columns]
    if missing:
        raise KeyError(f"pair_mask missing required columns: {missing}")
    return (
        primary_mask(frame, domain=domain, cuts=cuts)
        & _strict_between(frame["r_input_s"].to_numpy(dtype=float), cuts.secondary_mag)
        & _strict_between(frame["Re_input_s"].to_numpy(dtype=float), cuts.secondary_re)
        & _strict_between(frame["distance"].to_numpy(dtype=float), cuts.separation)
    )


def describe(domain: Domain = None, cuts: PairPopulationCuts = LSST_PAIR_CUTS):
    primary_mag = (
        (domain.magnitude_min, domain.magnitude_max)
        if domain is not None else cuts.primary_mag
    )
    primary_re = (
        (domain.half_light_radius_min, domain.half_light_radius_max)
        if domain is not None else cuts.primary_re
    )
    return (
        f"primary: {primary_mag[0]}<r_p<{primary_mag[1]}, "
        f"{primary_re[0]}<Re_p<{primary_re[1]} arcsec; "
        f"secondary: {cuts.secondary_mag[0]}<r_s<{cuts.secondary_mag[1]}, "
        f"{cuts.secondary_re[0]}<Re_s<{cuts.secondary_re[1]} arcsec; "
        f"{cuts.separation[0]}<separation<{cuts.separation[1]} arcsec"
    )
