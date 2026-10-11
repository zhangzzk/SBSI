"""The disk-preserving move of a measured shape by the blending emulator.

The emulator predicts a velocity for each simulated measurement; the measured
shape is moved by the Mobius map with the bounded shift radial_tanh(velocity),
so it always stays inside the unit disk.
"""
import torch

from .flow_disk import radial_tanh


def mobius_translate(shape, shift):
    """Apply (e+t)/(1+conj(t)*e), returning log|det d(output)/d(e)|.

Inputs broadcast with a final two-component axis. ``shift`` must lie strictly
inside the disk and be held fixed when taking this Jacobian. Mathematically
the map preserves the disk. Floating-point boundary rounding is neither
clipped nor repaired, as for the underlying physical flow.
"""
    shape, shift = torch.broadcast_tensors(shape.double(), shift.double())
    if shape.shape[-1] != 2:
        raise ValueError('two-component shapes and shifts required')
    t2 = shift.square().sum(-1)
    if not bool(torch.isfinite(shift).all() and (t2 < 1).all()):
        raise ValueError('finite strictly interior transport shift required')
    # Complex multiplication written as real arithmetic for real autograd.
    dr = 1+(shape*shift).sum(-1)
    di = shift[..., 0]*shape[..., 1]-shift[..., 1]*shape[..., 0]
    numerator = shape+shift
    denominator = dr.square()+di.square()
    result = torch.stack((numerator[..., 0]*dr+numerator[..., 1]*di,
                          numerator[..., 1]*dr-numerator[..., 0]*di), dim=-1)/denominator[..., None]
    logdet = 2*torch.log1p(-t2)-2*torch.log(denominator)
    return result, logdet


def disk_response_transport(shape, velocity, *, inverse=False):
    """Map a shape using a bounded translation derived from an R² velocity.

The radial tanh bounds the translation parameter, not the measured shape.
There is no Jacobian for the velocity map: velocity is a fixed condition,
not an output integrated over in this density. Inverse uses the same fixed
velocity with the sign of its disk shift reversed.
"""
    shift, _ = radial_tanh(velocity)
    return mobius_translate(shape, -shift if inverse else shift)
