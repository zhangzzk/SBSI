"""Explicit invariant size for area-preserving shear flows.

The radius and axis ratio here are BOTH pre-shear catalogue properties. Never
combine the original semi-major radius with a post-shear axis ratio. This
module does not change catalogue cuts, the response grid, or measured outputs.
"""

import numpy as np

CIRCULARIZED_FEATURE = "circularized_Re_input_p"
SEMIMAJOR_FEATURE = "Re_input_p"


def circularized_radius(semimajor, unsheared_axis_ratio):
    """Return a0 sqrt(q0), rejecting invalid truth instead of repairing it."""
    a, q = np.broadcast_arrays(
        np.asarray(semimajor, dtype=np.float64), np.asarray(unsheared_axis_ratio, dtype=np.float64)
    )
    if (
        not np.isfinite(a).all()
        or not np.isfinite(q).all()
        or np.any(a <= 0)
        or np.any(q <= 0)
        or np.any(q > 1)
    ):
        raise ValueError("finite positive pre-shear semi-major radius and 0 < q0 <= 1 required")
    return a * np.sqrt(q)
