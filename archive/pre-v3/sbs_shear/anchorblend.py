"""Utilities for a neighbour-only coherent-shear response simulation.

Sparse ``anchor`` galaxies are kept unsheared while every other source in the
field receives the same signed shear.  If anchors are separated by more than
twice the response aperture, every neighbour of an anchor is sheared and the
anchor's antithetic shape difference isolates the scene-level blend response.
"""
from __future__ import annotations

import numpy as np


def sparse_anchor_mask(ra_deg, dec_deg, eligible, min_separation_arcsec=20.0):
    """Greedily select deterministic eligible anchors with a guarded spacing.

    Input row order is the deterministic priority.  Positions are projected to
    a local tangent plane; the one-degree FS2 fields make this approximation
    accurate far below the spacing tolerance.
    """
    ra = np.asarray(ra_deg, dtype=float)
    dec = np.asarray(dec_deg, dtype=float)
    ok = np.asarray(eligible, dtype=bool)
    if not (ra.shape == dec.shape == ok.shape):
        raise ValueError("ra, dec, and eligible must have the same shape")
    sep = float(min_separation_arcsec)
    if not np.isfinite(sep) or sep <= 0:
        raise ValueError("min_separation_arcsec must be positive and finite")
    if len(ra) == 0:
        return np.zeros(0, dtype=bool)

    dec0 = float(np.nanmedian(dec))
    x = (ra - np.nanmin(ra)) * np.cos(np.deg2rad(dec0)) * 3600.0
    y = (dec - np.nanmin(dec)) * 3600.0
    finite = ok & np.isfinite(x) & np.isfinite(y)
    chosen = np.zeros(len(ra), dtype=bool)
    cells = {}
    sep2 = sep * sep
    for i in np.flatnonzero(finite):
        cell = (int(np.floor(x[i] / sep)), int(np.floor(y[i] / sep)))
        conflict = False
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for j in cells.get((cell[0] + dx, cell[1] + dy), ()):
                    if (x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2 <= sep2:
                        conflict = True
                        break
                if conflict:
                    break
            if conflict:
                break
        if not conflict:
            chosen[i] = True
            cells.setdefault(cell, []).append(i)
    return chosen


def assert_anchor_spacing(ra_deg, dec_deg, anchor, min_separation_arcsec=20.0):
    """Return the minimum anchor separation and refuse a spacing violation."""
    from scipy.spatial import cKDTree

    ra = np.asarray(ra_deg, dtype=float)
    dec = np.asarray(dec_deg, dtype=float)
    anchor = np.asarray(anchor, dtype=bool)
    dec0 = float(np.nanmedian(dec))
    xy = np.column_stack([
        (ra[anchor] - np.nanmin(ra)) * np.cos(np.deg2rad(dec0)) * 3600.0,
        (dec[anchor] - np.nanmin(dec)) * 3600.0,
    ])
    if len(xy) < 2:
        return float("inf")
    dist = cKDTree(xy).query(xy, k=2)[0][:, 1]
    minimum = float(dist.min())
    if minimum <= float(min_separation_arcsec) - 1.0e-8:
        raise RuntimeError(
            f"anchor spacing {minimum:.6f} arcsec violates "
            f"minimum {min_separation_arcsec:.6f}"
        )
    return minimum
