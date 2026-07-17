"""Analytic shear-distortion map S_gamma for the true scene (SBI_shear.md §2).

Shear is applied to the *true scene before rendering*, so the learned catalogue
density is shear-free and shear enters only through this closed-form, differentiable
map of the intrinsic properties.  This module implements the pieces SBSI needs:

* ``apply_shear_to_ellipticity`` -- the Mobius (reduced-shear) transform of the
  spin-2 ellipticity ``eps' = (eps + g) / (1 + conj(g) eps)``, in the SBSI/blendemu
  canonical sky basis ``(e1, e2) = e (cos 2theta, sin 2theta)``.
* ``inverse_shear_to_ellipticity`` -- its inverse (the prior shift ``S_{-g}``).
* ``shear_jacobian_at_zero`` -- ``d eps'/d g`` of the *true scene* shape at ``g = 0``;
  its orientation average is the identity (unit responsivity of the BARE scene shape).
  NOTE: this is NOT the measured responsivity. The MEASURED first-moment response is
  ``R = M_data * <1 - eps^2> ~ 0.233`` (PSF dilution + measurement shrink), derived by
  finite-differencing the trained flow, not from this scene Jacobian. Do not wire this
  function into a calibration as if it were the measured R.
* ``shear_separation_matrix`` / ``magnification`` -- the A-matrix distortion of the
  pair-separation vector and the flux/size magnification, used for exact
  finite-shear injection in validation.

Everything is plain NumPy and broadcasts over arrays of galaxies.  The core science uses
``apply_shear_to_ellipticity`` (the finite-shear map S_delta applied to the intrinsic shape
inside the response loss / diagnostics); ``shear_jacobian_at_zero`` is a reference quantity
used only in tests, not in the trained-response calibration.
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


def inverse_shear_to_ellipticity(e1, e2, g1, g2):
    """Inverse map eps = (eps' - g) / (1 - conj(g) eps') (i.e. apply -g undoing)."""
    return apply_shear_to_ellipticity(e1, e2, -np.asarray(g1, dtype=float), -np.asarray(g2, dtype=float))


def shear_jacobian_at_zero(e1, e2):
    """Return d(eps')/d(g) at g=0 as a real 2x2 Jacobian per object.

    For eps' = (eps + g)/(1 + conj(g) eps), the Wirtinger derivative at g=0 is
    ``d eps' = dg - eps**2 d conj(g)``, which in real components is

        J = [[1 - a, -b], [-b, 1 + a]],   a = e1**2 - e2**2,  b = 2 e1 e2.

    Returns
    -------
    np.ndarray
        Shape ``(..., 2, 2)`` with J[...,0,0]=de1'/dg1, [...,0,1]=de1'/dg2,
        [...,1,0]=de2'/dg1, [...,1,1]=de2'/dg2.
    """
    e1 = np.asarray(e1, dtype=float)
    e2 = np.asarray(e2, dtype=float)
    a = e1 * e1 - e2 * e2
    b = 2.0 * e1 * e2
    J = np.empty(np.shape(a) + (2, 2), dtype=float)
    J[..., 0, 0] = 1.0 - a
    J[..., 0, 1] = -b
    J[..., 1, 0] = -b
    J[..., 1, 1] = 1.0 + a
    return J


def shear_matrix(g1, g2, kappa=0.0):
    """Return the lensing distortion matrix A = (1-kappa) I - Gamma per object.

    Gamma = [[g1, g2], [g2, -g1]].  With kappa=0 this is the reduced-shear
    A-matrix used to distort the pair-separation vector.
    """
    g1 = np.asarray(g1, dtype=float)
    g2 = np.asarray(g2, dtype=float)
    kappa = np.asarray(kappa, dtype=float)
    A = np.empty(np.broadcast(g1, g2, kappa).shape + (2, 2), dtype=float)
    A[..., 0, 0] = (1.0 - kappa) - g1
    A[..., 0, 1] = -g2
    A[..., 1, 0] = -g2
    A[..., 1, 1] = (1.0 - kappa) + g1
    return A


def shear_separation(dtheta, g1, g2, kappa=0.0):
    """Apply the A-matrix to a separation vector dtheta=(d1,d2). Returns (d1',d2')."""
    A = shear_matrix(g1, g2, kappa=kappa)
    d1 = np.asarray(dtheta[0], dtype=float)
    d2 = np.asarray(dtheta[1], dtype=float)
    out1 = A[..., 0, 0] * d1 + A[..., 0, 1] * d2
    out2 = A[..., 1, 0] * d1 + A[..., 1, 1] * d2
    return out1, out2


def magnification(g1, g2, kappa=0.0):
    """Flux/area magnification mu = 1 / det A = 1 / ((1-kappa)^2 - |g|^2)."""
    g1 = np.asarray(g1, dtype=float)
    g2 = np.asarray(g2, dtype=float)
    kappa = np.asarray(kappa, dtype=float)
    det = (1.0 - kappa) ** 2 - (g1 * g1 + g2 * g2)
    return 1.0 / det
