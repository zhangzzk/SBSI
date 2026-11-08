"""Differentiable own-leg probability moments for training diagnostics only."""
import numpy as np
import torch

from sbsi.flow_guard_response import physical_context_draws, torch_guard_weights
from scripts.guard_selection_mass import primary_cut_index


def sampled_soft_mass_backward(model, population, indices, *, draws, batch_size, seed):
    indices = np.asarray(indices)
    if (indices.ndim != 1 or not len(indices) or indices.dtype.kind not in 'iu'
            or indices.min() < 0 or indices.max() >= len(population.pairs)
            or draws < 1 or batch_size < 1):
        raise ValueError('nonempty valid parent indices and positive draw/batch counts required')
    cut = population.cuts[primary_cut_index(population)]
    total = 0.
    for block, start in enumerate(range(0, len(indices), batch_size)):
        rows = torch.as_tensor(indices[start:start+batch_size], device=population.context.device)
        values = physical_context_draws(model, [population.context[population.pairs[rows, 0]]],
                                        draws, seed+block)[0]
        weights = torch_guard_weights(values, (cut,), radius_softness=population.radius_softness,
            magnitude_softness=population.magnitude_softness, zero_point=population.zero_point)[..., 0].mean(1)
        mass = (weights*population.usable[rows, 0]).double().sum()/len(indices)
        if not bool(torch.isfinite(mass)):
            raise ValueError('nonfinite sampled soft mass')
        mass.backward()
        total += float(mass.detach())
    return total


def common_descent(response_gradient, probability_gradient):
    """Angle-bisecting descent; probability gradient must include residual sign.

This diagnoses first-order feasibility, not a line-search or accepted update.
The norm matches response-only descent for a meaningful direction comparison.
"""
    response, probability = map(lambda x: np.asarray(x, dtype=np.float64),
                                (response_gradient, probability_gradient))
    if response.ndim != 1 or response.shape != probability.shape or not np.isfinite([response, probability]).all():
        raise ValueError('aligned finite gradient vectors required')
    nr, np_ = np.linalg.norm(response), np.linalg.norm(probability)
    if min(nr, np_) <= 0:
        raise ValueError('two nonzero objective gradients required')
    combined = response/nr+probability/np_
    norm = np.linalg.norm(combined)
    cosine = float(np.dot(response/nr, probability/np_))
    if norm < 1e-12:
        return None, {'cosine': cosine, 'strict_common_descent': False}
    direction = -nr*combined/norm
    sr, sp = float(response@direction), float(probability@direction)
    return direction, {'cosine': cosine, 'strict_common_descent': sr < 0 and sp < 0,
                       'response_only_probability_slope': float(-probability@response),
                       'common_response_slope': sr, 'common_probability_slope': sp,
                       'response_gradient_norm': float(nr), 'probability_gradient_norm': float(np_)}
