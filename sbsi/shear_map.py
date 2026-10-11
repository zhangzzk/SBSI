"""The reduced-shear map of an intrinsic ellipticity.

Shear is applied to the true scene before rendering, so it enters the model only
through this closed-form map of each galaxy's intrinsic shape.
"""

from __future__ import annotations

import numpy as np


def apply_shear_to_ellipticity(e1, e2, g1, g2):
    """Mobius reduced-shear transform eps' = (eps + g) / (1 + conj(g) eps).

    Parameters and returns are real spin-2 components in the canonical basis.
    Broadcasts over arrays.
    """
    eps = np.asarray(e1, dtype=float) + 1j * np.asarray(e2, dtype=float)
    g = np.asarray(g1, dtype=float) + 1j * np.asarray(g2, dtype=float)
    out = (eps + g) / (1.0 + np.conj(g) * eps)
    return np.real(out), np.imag(out)
