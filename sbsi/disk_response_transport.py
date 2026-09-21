"""Experimental disk-preserving response transport with an exact density.

The velocity is a newly fitted transport parameter, not an existing physical
R_blend coefficient. For fixed auxiliary outputs it must not depend on shape;
that restriction gives the triangular joint-density Jacobian used here.
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


def infinitesimal_response_matrix(shape):
    """Derivative with respect to the two velocity components at velocity=0."""
    shape = shape.double()
    if shape.shape[-1] != 2:
        raise ValueError('two-component shapes required')
    a, b = shape.unbind(-1)
    return torch.stack((1-a*a+b*b, -2*a*b, -2*a*b, 1+a*a-b*b), dim=-1).reshape(*shape.shape[:-1], 2, 2)


def directional_response_gain(shape, direction):
    """uᵀ L(e) u; positive for interior e and a nonzero direction u.

For unit u the gain lies in [1-|e|², 1+|e|²]. This is the physical derivative
needed when fitting transport response coefficients from measured labels.
"""
    matrix = infinitesimal_response_matrix(shape)
    direction = direction.double()
    return torch.einsum('...i,...ij,...j->...', direction, matrix, direction)


def transported_physical_log_prob(physical, velocity, base_log_prob):
    """Evaluate a transported four-output density in physical coordinates.

``base_log_prob`` accepts (..., 4) tensors ordered e1,e2,radius,flux and
returns their normalized physical log density. ``velocity`` must be the
same coefficient used to generate the forward draw, depending only on the
unchanged radius/flux and external conditions, never on measured shape.
The radius/flux dependence leaves the joint Jacobian triangular, including
piecewise constant tree coefficients. Invalid physical outputs get -inf.
This experimental primitive does not include usability or selection factors.
"""
    physical = physical.double()
    if physical.shape[-1] != 4:
        raise ValueError('four physical outputs required')
    valid = (torch.isfinite(physical).all(-1)
             & (physical[..., :2].square().sum(-1) < 1)
             & (physical[..., 2:] > 0).all(-1))
    placeholder = physical.new_tensor([0., 0., 1., 1.])
    safe = torch.where(valid[..., None], physical, placeholder)
    shape, inverse_logdet = disk_response_transport(safe[..., :2], velocity, inverse=True)
    original = torch.cat((shape, safe[..., 2:]), dim=-1)
    log_prob = base_log_prob(original)
    if log_prob.shape != physical.shape[:-1]:
        raise ValueError('base density must return one value per physical row')
    result = log_prob+inverse_logdet
    return torch.where(valid, result, torch.full_like(result, -torch.inf))
