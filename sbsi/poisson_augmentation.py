"""Exact density tilt for one-point augmentation of an ordered Poisson law.

For normalized h, on positive-base-density scenes, uniform insertion of one
independent h point gives q_aug(E)/q_env(E) = sum_{u in E} h(u)/lambda(u).
This module evaluates that valid-scene tilt only. It neither draws h nor
establishes normalization or global support. If h has mass outside lambda,
that mass requires separate handling: the explicitly flagged zero-target
initialization/positive-beta-resampling contract in informed_environment is
one allowed path. Never discard, retry, or renormalize such raw initial draws.
"""
from __future__ import annotations

import numpy as np


def point_log_augmentation_ratio(log_h, log_intensity):
    """Return log h - log lambda for points on positive base-intensity support.

    Both densities must use the same native-mark/type and Cartesian-position
    measure. ``log_h=-inf`` is allowed and contributes zero to the augmented
    law. A zero base intensity is rejected, whether h is positive (singular
    augmentation mass) or zero (a point outside both laws). Such configurations
    cannot use this finite tilt. Handle them separately, for example as the
    explicit zero-target initial rows in the informed_environment contract.
    Pointwise checks cannot certify global h << lambda or integral(h)=1.
    """
    proposal, intensity = np.asarray(log_h, float), np.asarray(log_intensity, float)
    if (proposal.ndim != 1 or proposal.shape != intensity.shape
            or np.isnan(proposal).any() or np.isposinf(proposal).any()):
        raise ValueError('aligned one-dimensional finite or -inf log h densities required')
    if not np.isfinite(intensity).all():
        raise ValueError('positive finite base intensity required; singular or zero-support points cannot use a tilt')
    with np.errstate(over='ignore'):
        ratio = proposal-intensity
    if np.isposinf(ratio).any() or np.any(np.isneginf(ratio) & np.isfinite(proposal)):
        raise FloatingPointError('log density ratio overflowed; do not clip or retry')
    return ratio


def poisson_augmentation_log_tilt(log_h_over_intensity, indptr, mixture_probability):
    """Log of (1-a) + a * sum_j h(u_j)/lambda(u_j), one value per CSR scene.

    ``indptr`` begins at zero, ends at the point count and is nondecreasing.
    Empty scenes have exactly log(1-a). A zero-point, zero-scene batch with
    indptr=[0] is permitted. The sum is invariant to neighbour ordering.

    ``0 < a < 1`` preserves all base-law configurations, including N=0.
    The caller must supply a known normalized h, and either certify h << lambda
    or separately retain off-support draws under an audited execution contract.
    -inf ratios mean h=0 on positive lambda support; +inf or NaN may indicate
    singular augmentation mass and are rejected, never silently removed.
    """
    ratios, pointers = np.asarray(log_h_over_intensity, float), np.asarray(indptr)
    if (ratios.ndim != 1 or np.isnan(ratios).any() or np.isposinf(ratios).any()):
        raise ValueError('finite or -inf point log ratios required; singular mass is unsupported')
    if (pointers.ndim != 1 or len(pointers) < 1 or pointers.dtype.kind not in 'iu'
            or pointers[0] != 0 or pointers[-1] != len(ratios)
            or np.any(pointers[1:] < pointers[:-1])):
        raise ValueError('nondecreasing integer CSR pointers from zero to the point count required')
    if (not np.isscalar(mixture_probability) or not np.isfinite(mixture_probability)
            or not 0 < mixture_probability < 1):
        raise ValueError('finite augmentation mixture probability strictly between zero and one required')
    a = float(mixture_probability)
    logarithmic_sum = np.full(len(pointers)-1, -np.inf)
    nonempty = np.flatnonzero(pointers[1:] > pointers[:-1])
    if len(nonempty):
        # Removing empty segments leaves each start immediately after the
        # preceding nonempty segment, making reduceat valid for ragged CSR.
        logarithmic_sum[nonempty] = np.logaddexp.reduceat(
            ratios, pointers[nonempty].astype(np.intp, copy=False))
    return np.logaddexp(np.log1p(-a), np.log(a)+logarithmic_sum)
