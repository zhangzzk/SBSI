"""Spin-2 helpers in the canonical sky basis shared with BlendEMU catalogues:

    (q1, q2) = q * (cos(2 theta), sin(2 theta)).
"""

from __future__ import annotations

import numpy as np


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


def ellipticity_from_axis_ratio_angle(axis_ratio, position_angle):
    """Convert axis ratio and position angle to canonical ellipticity components."""
    q = np.asarray(axis_ratio, dtype=float)
    eps = (1.0 - q) / (1.0 + q)
    return spin2_components_from_angle(position_angle, amplitude=eps)
