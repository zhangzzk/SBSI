"""Lossless radial information for model-generated shapes near the disk edge.

The extra value is log(1-|e|²), computed before rounding the Cartesian shape.
It describes the same two-dimensional measurement, not a new random target.
It cannot be reconstructed from a rounded boundary observation.
"""
import math

import numpy as np
import torch
from torch.nn import functional as F

from .flow_disk import _radius_parts, radial_tanh


LOG_GAP_COLUMN = 'measured_shape_log_disk_gap'


def radial_log_gap(coordinates):
    """Analytic log(sech²|u|), including the cancellation-free origin series."""
    _, r2, small, r = _radius_parts(coordinates)
    series = r2 * (-1 + r2 * (1 / 6 - r2 * 2 / 45))
    return torch.where(small, series, 2 * (math.log(2.) - r - F.softplus(-2*r)))


def validate_shape_encoding(shape, log_gap):
    """Reject missing/inconsistent metadata rather than repairing observations."""
    shape, log_gap = np.asarray(shape, float), np.asarray(log_gap, float)
    if shape.shape != (*log_gap.shape, 2):
        raise ValueError('shape and disk log-gap must align')
    expected = -np.expm1(log_gap)
    norm2 = np.square(shape).sum(-1)
    valid = (np.isfinite(shape).all(-1) & np.isfinite(log_gap) & (log_gap <= 0)
             & (np.abs(norm2-expected) <= 64*np.finfo(float).eps))
    if not valid.all():
        raise ValueError('inconsistent or invalid disk log-gap encoding')


def transport_with_log_gap(shape, log_gap, velocity, *, inverse=False):
    """Möbius transport retaining the radial gap and its exact Jacobian.

    1-|T(e)|² = (1-|e|²)(1-|t|²)/|1+conj(t)e|².
    Cartesian roundoff is permitted; the independently propagated gap remains
    strictly positive mathematically even when exp(log_gap) underflows.
    """
    from .disk_response_transport import mobius_translate
    shift, _ = radial_tanh(velocity)
    if inverse:
        shift = -shift
    transformed, determinant = mobius_translate(shape, shift)
    # Use the separately computed change; subtracting two huge negative gaps
    # would lose the (small) transport Jacobian.
    gap_change = determinant / 2
    norm2 = transformed.square().sum(-1)
    interior = norm2 < .5
    interior_gap = torch.log1p(-torch.where(interior, norm2, torch.zeros_like(norm2)))
    gap = torch.where(interior, interior_gap, log_gap + gap_change)
    return transformed, gap, determinant


def physical_log_prob_with_gap(model, physical, log_gap, context):
    """Physical density of a ConditionalPhysicalDiskFlow with radial metadata.

    Inputs have already been validated. No shape clipping or density floor.
    The base network uses its original precision and unchanged parameters.
    """
    shape = physical[..., :2].double()
    norm = torch.hypot(shape[..., 0], shape[..., 1])
    radius = torch.sqrt(-torch.expm1(log_gap))
    unbounded_radius = torch.log1p(radius) - .5*log_gap
    safe_norm = torch.where(norm > 0, norm, torch.ones_like(norm))
    unbounded = shape * (unbounded_radius/safe_norm)[..., None]
    small = unbounded_radius < 1.e-4
    # log(r/tanh(r)) = r²/3 - 7r⁴/90 + 62r⁶/2835 + O(r⁸).
    r2 = unbounded_radius.square()
    series = r2 * (1/3 + r2 * (-7/90 + r2 * 62/2835))
    ratio_log = torch.log(torch.where(small, torch.ones_like(radius), unbounded_radius))
    ratio_log -= torch.log(torch.where(small, torch.ones_like(radius), radius))
    inverse_shape_ld = -log_gap + torch.where(small, series, ratio_log)
    wshape = (unbounded-model.disk_coordinate_means)/model.disk_coordinate_scales
    positive = physical[..., 2:].double()/model.physical_units
    u = positive + torch.log(-torch.expm1(-positive))
    waux = (u-model.physical_coordinate_means)/model.physical_coordinate_scales
    w = torch.cat((wshape, waux), -1).to(dtype=next(model.base.parameters()).dtype)
    z, inverse_base_ld = model.base.inverse(w, context)
    density = -.5*(z.double().square()+math.log(2*math.pi)).sum(-1) + inverse_base_ld.double()
    inverse_aux_ld = -(model.physical_units.log() + model.physical_coordinate_scales.log()
                       + F.logsigmoid(u)).sum(-1)
    return density + inverse_shape_ld - model.disk_coordinate_scales.log().sum() + inverse_aux_ld
