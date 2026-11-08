"""Own-normalized per-leg response moments and memory-bounded gradient replay.

The full-parent moment order is D, A1, A2, B1, B2, H11, H12, H21, H22.
D=E[U0*w0], A=E[U0*w0*e0], B=E[(Ug*wg-U0*w0)*design],
H=E[(Ug*wg*eg-U0*w0*e0) outer design]. The response of the model's
selected mean is (H-(A/D) outer B)/D. D and A remain differentiable;
they are not replaced by observed constants or used as fitted corrections.

Finite integrated moments give a plug-in ratio, not an unbiased ratio
estimator. Before training, reference integration and independent gradient
replicas must establish adequate numerical accuracy. No optimizer lives here.
"""
import numpy as np
import torch

from sbsi.flow_guard_response import physical_context_draws, torch_guard_weights


def moment_contributions(model, population, indices, *, draws, seed):
    """Return one nine-moment vector per parent and cut, retaining own-leg U."""
    pair = population.pairs[indices]
    values = physical_context_draws(model, [population.context[pair[:, leg]] for leg in (0, 1)], draws, seed)
    mass, shape = [], []
    for leg, output in enumerate(values):
        weights = torch_guard_weights(output, population.cuts, radius_softness=population.radius_softness,
                                      magnitude_softness=population.magnitude_softness,
                                      zero_point=population.zero_point).double()
        weights = weights*population.usable[indices, leg, None, None]
        mass.append(weights.mean(1))
        shape.append((weights[..., None]*output[..., None, :2].double()).mean(1))
    design = population.design[indices].double()
    selection_slope = (mass[1]-mass[0])[..., None]*design[:, None, :]
    shape_slope = (shape[1]-shape[0])[..., None]*design[:, None, None, :]
    result = torch.cat((mass[0][..., None], shape[0], selection_slope,
                        shape_slope.flatten(-2)), dim=-1)
    if not bool(torch.isfinite(result).all()):
        raise ValueError('finite full-parent response moments required')
    return result


def response_from_moments(moments):
    """Differentiate the model's own baseline mean and mass as part of R."""
    if (moments.ndim != 2 or moments.shape[1] != 9 or not len(moments)
            or not bool(torch.isfinite(moments).all() and (moments[:, 0] > 0).all())):
        raise ValueError('finite per-cut nine-moment vectors with positive baseline mass required')
    mass = moments[:, 0]
    mean = moments[:, 1:3]/mass[:, None]
    selection_slope = moments[:, 3:5]
    shape_slope = moments[:, 5:9].reshape(-1, 2, 2)
    return ((shape_slope-mean[..., None]*selection_slope[:, None, :])/mass[:, None, None]).flatten(1)


def loss_cotangent(moments, target, response_scale):
    """Exact first derivative of squared response error with respect to moments."""
    integrated = moments.detach().double().requires_grad_(True)
    target = torch.as_tensor(target, dtype=integrated.dtype, device=integrated.device)
    scale = torch.as_tensor(response_scale, dtype=integrated.dtype, device=integrated.device)
    response = response_from_moments(integrated)
    if (target.shape != response.shape or scale.shape not in ((), (len(integrated), 1))
            or not bool(torch.isfinite(target).all() and torch.isfinite(scale).all() and (scale > 0).all())):
        raise ValueError('aligned finite response target and positive scalar/per-cut scales required')
    loss = ((response-target)/scale).square().mean()
    gradient, = torch.autograd.grad(loss, integrated)
    return float(loss.detach()), gradient.detach()


def sampled_moment_backward(model, population, indices, integrated_moments, *, draws, batch_size, seed):
    """Replay raw-moment derivatives with the global ratio-loss cotangent.

``integrated_moments`` must be integrated at the current model parameters on
the same full-parent population. Sampled indices may repeat. Cotangents
include normalization derivatives before detaching for the replay; freezing
the normalization inside response_from_moments would give a different loss.
The gradient is a sampled Jacobian times a finite-integration cotangent.
"""
    indices = np.asarray(indices)
    if (indices.ndim != 1 or not len(indices) or indices.dtype.kind not in 'iu'
            or indices.min() < 0 or indices.max() >= len(population.pairs)
            or draws < 2 or draws % 2 or batch_size < 1):
        raise ValueError('valid nonempty parent indices, positive batch and even draws required')
    integrated_moments = torch.as_tensor(integrated_moments, dtype=torch.float64, device=population.context.device)
    loss, cotangent = loss_cotangent(integrated_moments, population.target, population.response_scale)
    total = torch.zeros_like(cotangent)
    for block, start in enumerate(range(0, len(indices), batch_size)):
        rows = torch.as_tensor(indices[start:start+batch_size], device=population.context.device)
        contribution = moment_contributions(model, population, rows, draws=draws, seed=seed+block)
        moments = contribution.sum(0)/len(indices)
        (moments*cotangent).sum().backward()
        total += moments.detach()
    return {'loss_at_integrated_moments': loss, 'sampled_moments': total.cpu().numpy(),
            'moment_cotangent': cotangent.cpu().numpy()}
