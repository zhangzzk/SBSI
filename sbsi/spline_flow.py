"""Conditional rational-quadratic neural spline flow (Durkan et al. 2019).

A drop-in, more expressive alternative to the affine-coupling flow in
``measurement_model.py``.  Affine coupling shrinks the conditional mean toward the
marginal (the recovery diagnostics showed M_model/M_data ~ 0.8); monotonic
rational-quadratic spline couplings model the conditional shape far more faithfully.

The public class ``ConditionalSplineFlow`` matches the ConditionalAffineFlow API
(``forward``, ``inverse``, ``log_prob``, ``sample``) so the trainer/bundle can use
either via a ``flow_type`` switch.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

DEFAULT_MIN_BIN_WIDTH = 1e-3
DEFAULT_MIN_BIN_HEIGHT = 1e-3
DEFAULT_MIN_DERIVATIVE = 1e-3


def _searchsorted(bin_locations, inputs, eps=1e-6):
    bin_locations = bin_locations.clone()
    bin_locations[..., -1] += eps
    return torch.sum(inputs[..., None] >= bin_locations, dim=-1) - 1


def unconstrained_rational_quadratic_spline(
    inputs, unnormalized_widths, unnormalized_heights, unnormalized_derivatives,
    inverse=False, tail_bound=5.0,
    min_bin_width=DEFAULT_MIN_BIN_WIDTH, min_bin_height=DEFAULT_MIN_BIN_HEIGHT,
    min_derivative=DEFAULT_MIN_DERIVATIVE,
):
    """Monotonic RQ spline on [-B, B] with linear (identity-slope) tails outside."""
    inside = (inputs >= -tail_bound) & (inputs <= tail_bound)
    outside = ~inside
    out = torch.zeros_like(inputs)
    logabsdet = torch.zeros_like(inputs)

    # Linear tails: identity, zero log-det.
    out[outside] = inputs[outside]
    logabsdet[outside] = 0.0

    if inside.any():
        # pad derivatives with the constant giving slope 1 in the boundary knots
        ud = unnormalized_derivatives
        constant = np.log(np.exp(1.0 - min_derivative) - 1.0)
        ud = F.pad(ud, pad=(1, 1), value=constant)
        o, lad = _rational_quadratic_spline(
            inputs=inputs[inside],
            unnormalized_widths=unnormalized_widths[inside, :],
            unnormalized_heights=unnormalized_heights[inside, :],
            unnormalized_derivatives=ud[inside, :],
            inverse=inverse,
            left=-tail_bound, right=tail_bound, bottom=-tail_bound, top=tail_bound,
            min_bin_width=min_bin_width, min_bin_height=min_bin_height, min_derivative=min_derivative,
        )
        out[inside] = o
        logabsdet[inside] = lad
    return out, logabsdet


def _rational_quadratic_spline(
    inputs, unnormalized_widths, unnormalized_heights, unnormalized_derivatives,
    inverse, left, right, bottom, top, min_bin_width, min_bin_height, min_derivative,
):
    num_bins = unnormalized_widths.shape[-1]

    widths = F.softmax(unnormalized_widths, dim=-1)
    widths = min_bin_width + (1 - min_bin_width * num_bins) * widths
    cumwidths = torch.cumsum(widths, dim=-1)
    cumwidths = F.pad(cumwidths, pad=(1, 0), value=0.0)
    cumwidths = (right - left) * cumwidths + left
    cumwidths[..., 0] = left
    cumwidths[..., -1] = right
    widths = cumwidths[..., 1:] - cumwidths[..., :-1]

    derivatives = min_derivative + F.softplus(unnormalized_derivatives)

    heights = F.softmax(unnormalized_heights, dim=-1)
    heights = min_bin_height + (1 - min_bin_height * num_bins) * heights
    cumheights = torch.cumsum(heights, dim=-1)
    cumheights = F.pad(cumheights, pad=(1, 0), value=0.0)
    cumheights = (top - bottom) * cumheights + bottom
    cumheights[..., 0] = bottom
    cumheights[..., -1] = top
    heights = cumheights[..., 1:] - cumheights[..., :-1]

    if inverse:
        bin_idx = _searchsorted(cumheights, inputs)[..., None]
    else:
        bin_idx = _searchsorted(cumwidths, inputs)[..., None]

    input_cumwidths = cumwidths.gather(-1, bin_idx)[..., 0]
    input_bin_widths = widths.gather(-1, bin_idx)[..., 0]
    input_cumheights = cumheights.gather(-1, bin_idx)[..., 0]
    delta = heights / widths
    input_delta = delta.gather(-1, bin_idx)[..., 0]
    input_derivatives = derivatives.gather(-1, bin_idx)[..., 0]
    input_derivatives_plus_one = derivatives[..., 1:].gather(-1, bin_idx)[..., 0]
    input_heights = heights.gather(-1, bin_idx)[..., 0]

    if inverse:
        a = (inputs - input_cumheights) * (input_derivatives + input_derivatives_plus_one - 2 * input_delta) \
            + input_heights * (input_delta - input_derivatives)
        b = input_heights * input_derivatives - (inputs - input_cumheights) * (
            input_derivatives + input_derivatives_plus_one - 2 * input_delta)
        c = -input_delta * (inputs - input_cumheights)
        discriminant = b.pow(2) - 4 * a * c
        discriminant = torch.clamp(discriminant, min=0.0)
        root = (2 * c) / (-b - torch.sqrt(discriminant))
        outputs = root * input_bin_widths + input_cumwidths

        theta_one_minus_theta = root * (1 - root)
        denominator = input_delta + (
            (input_derivatives + input_derivatives_plus_one - 2 * input_delta) * theta_one_minus_theta)
        derivative_numerator = input_delta.pow(2) * (
            input_derivatives_plus_one * root.pow(2)
            + 2 * input_delta * theta_one_minus_theta
            + input_derivatives * (1 - root).pow(2))
        logabsdet = torch.log(derivative_numerator) - 2 * torch.log(denominator)
        return outputs, -logabsdet
    else:
        theta = (inputs - input_cumwidths) / input_bin_widths
        theta_one_minus_theta = theta * (1 - theta)
        numerator = input_heights * (input_delta * theta.pow(2) + input_derivatives * theta_one_minus_theta)
        denominator = input_delta + (
            (input_derivatives + input_derivatives_plus_one - 2 * input_delta) * theta_one_minus_theta)
        outputs = input_cumheights + numerator / denominator
        derivative_numerator = input_delta.pow(2) * (
            input_derivatives_plus_one * theta.pow(2)
            + 2 * input_delta * theta_one_minus_theta
            + input_derivatives * (1 - theta).pow(2))
        logabsdet = torch.log(derivative_numerator) - 2 * torch.log(denominator)
        return outputs, logabsdet


def _activation(name):
    return {"silu": nn.SiLU, "gelu": nn.GELU, "tanh": nn.Tanh}[name.lower()]


class SplineCoupling(nn.Module):
    def __init__(self, target_dim, context_dim, num_bins=8, hidden_dim=256, n_layers=3,
                 mask=None, tail_bound=5.0, activation="silu"):
        super().__init__()
        if mask is None:
            mask = [(i % 2) for i in range(target_dim)]
        mask = torch.as_tensor(mask, dtype=torch.float32)
        self.register_buffer("mask", mask)
        self.num_bins = int(num_bins)
        self.tail_bound = float(tail_bound)
        self.target_dim = target_dim
        n_identity = int(mask.sum().item())
        n_transform = target_dim - n_identity
        self.transform_idx = torch.where(mask == 0)[0]
        self.identity_idx = torch.where(mask == 1)[0]
        self.n_transform = n_transform
        params_per = 3 * self.num_bins - 1
        act = _activation(activation)
        layers, dim = [], n_identity + context_dim
        for _ in range(n_layers):
            layers += [nn.Linear(dim, hidden_dim), act()]
            dim = hidden_dim
        layers += [nn.Linear(dim, n_transform * params_per)]
        self.net = nn.Sequential(*layers)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def _params(self, x_identity, context):
        h = self.net(torch.cat([x_identity, context], dim=-1))
        h = h.view(h.shape[0], self.n_transform, 3 * self.num_bins - 1)
        w = h[..., :self.num_bins]
        ht = h[..., self.num_bins:2 * self.num_bins]
        d = h[..., 2 * self.num_bins:]
        return w, ht, d

    def _transform(self, y, context, inverse):
        x_id = y[:, self.identity_idx]
        y_tr = y[:, self.transform_idx]
        w, ht, d = self._params(x_id, context)
        out_tr, logabsdet = unconstrained_rational_quadratic_spline(
            y_tr, w, ht, d, inverse=inverse, tail_bound=self.tail_bound)
        out = y.clone()
        out[:, self.transform_idx] = out_tr
        return out, logabsdet.sum(dim=-1)

    def forward(self, z, context):
        return self._transform(z, context, inverse=False)

    def inverse(self, x, context):
        return self._transform(x, context, inverse=True)


class ConditionalSplineFlow(nn.Module):
    def __init__(self, target_dim, context_dim, hidden_dim=256, n_layers=3, n_flows=8,
                 num_bins=8, tail_bound=5.0, activation="silu", **kwargs):
        super().__init__()
        if target_dim < 2:
            raise ValueError("target_dim must be >= 2")
        self.target_dim = int(target_dim)
        self.context_dim = int(context_dim)
        base = np.array([(i % 2) for i in range(target_dim)], dtype=np.float32)
        self.layers = nn.ModuleList([
            SplineCoupling(target_dim, context_dim, num_bins=num_bins, hidden_dim=hidden_dim,
                           n_layers=n_layers, mask=(base if i % 2 == 0 else 1.0 - base),
                           tail_bound=tail_bound, activation=activation)
            for i in range(n_flows)
        ])

    def inverse(self, x, context):
        logdet = torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)
        z = x
        for layer in reversed(self.layers):
            z, ldj = layer.inverse(z, context)
            logdet = logdet + ldj
        return z, logdet

    def forward(self, z, context):
        logdet = torch.zeros(z.shape[0], dtype=z.dtype, device=z.device)
        x = z
        for layer in self.layers:
            x, ldj = layer(x, context)
            logdet = logdet + ldj
        return x, logdet

    def log_prob(self, x, context):
        z, logdet = self.inverse(x, context)
        log_base = -0.5 * (z.pow(2) + np.log(2.0 * np.pi)).sum(dim=-1)
        return log_base + logdet

    def sample(self, context, n_samples=1):
        if context.ndim != 2:
            raise ValueError("context must be (batch, context_dim)")
        batch = context.shape[0]
        ctx = context[:, None, :].expand(batch, n_samples, self.context_dim).reshape(-1, self.context_dim)
        z = torch.randn(batch * n_samples, self.target_dim, dtype=context.dtype, device=context.device)
        x, _ = self.forward(z, ctx)
        return x.reshape(batch, n_samples, self.target_dim)
