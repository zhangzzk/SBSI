"""Distribution finite-difference supervision from matched measurement pairs.

These primitives do not apply selection cuts or modify inference. Outputs
must use the same coordinate transform in data and both model draw groups.
The two groups must be independent; common random numbers belong WITHIN
each group's two shear legs, not between groups.
"""

import math

import torch
from torch import nn


class GaussianFourierFeatures(nn.Module):
    """Fixed finite features for an equally weighted mixture of RBF kernels.

    For bandwidth h, frequencies have covariance I/h**2. Both sine and
    cosine are retained, with normalization giving phi(y).phi(y) == 1.
    Bandwidths are in the input coordinates (e.g. standardized log radius).
    This finite approximation does not uniquely identify every distribution.
    """

    def __init__(self, input_dim, *, bandwidths=(0.25, 0.5, 1.0, 2.0),
                 frequencies_per_band=32, seed=501):
        super().__init__()
        if not isinstance(input_dim, int) or input_dim < 1:
            raise ValueError("input_dim must be a positive integer")
        if not isinstance(frequencies_per_band, int) or frequencies_per_band < 1:
            raise ValueError("frequencies_per_band must be a positive integer")
        widths = tuple(float(v) for v in bandwidths)
        if not widths or any(not math.isfinite(v) or v <= 0 for v in widths):
            raise ValueError("bandwidths must be finite and positive")
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        # Generate in double precision so analytic tests and saved frequencies
        # share one definition; callers explicitly choose training dtype/device.
        frequency = torch.cat([
            torch.randn(frequencies_per_band, input_dim, generator=generator,
                        dtype=torch.float64) / width for width in widths
        ])
        self.register_buffer("frequencies", frequency)

    def forward(self, values):
        if values.ndim < 2 or values.shape[-1] != self.frequencies.shape[1]:
            raise ValueError("values must end in the configured input dimension")
        phase = values @ self.frequencies.T
        return torch.cat((phase.cos(), phase.sin()), dim=-1) / math.sqrt(len(self.frequencies))


def feature_secant(features0, features1, shear):
    """Per-row feature changes at the actual paired shear separation.

    Inputs are (row, feature) expectations or observed feature vectors.
    Shear is a positive scalar or a positive vector with one entry per row;
    the sign and applied direction are represented by the ordered contexts.
    """
    if features0.ndim != 2 or features0.shape != features1.shape:
        raise ValueError("aligned (row, feature) tensors are required")
    h = torch.as_tensor(shear, dtype=features0.dtype, device=features0.device)
    if h.ndim > 1 or (h.ndim == 1 and len(h) != len(features0)):
        raise ValueError("shear must be scalar or have one entry per row")
    if not bool(torch.isfinite(h).all()) or not bool((h > 0).all()):
        raise ValueError("shear must be finite and positive")
    return (features1 - features0) / (h[:, None] if h.ndim else h)


def draw_feature_secant(feature_map, samples0, samples1, shear):
    """Transform each draw BEFORE averaging; returns (row, feature)."""
    if samples0.ndim != 3 or samples0.shape != samples1.shape or samples0.shape[1] < 1:
        raise ValueError("aligned nonempty (row, draw, output) samples are required")
    return feature_secant(feature_map(samples0).mean(dim=1),
                          feature_map(samples1).mean(dim=1), shear)


def independent_response_score(estimate_a, estimate_b, target):
    """Per-row squared conditional-mean error, up to a data-only constant.

    A and B must be conditionally independent unbiased estimates of the same
    model feature secant, and independent of target measurement noise.
    E[A.B - (A+B).T | x] = ||M(x)-E[T|x]||**2 - ||E[T|x]||**2.
    Thus optimization has no extra penalty on model Monte Carlo variance.
    Negative scores are valid; target-squared is omitted for lower noise.
    Dot products SUM feature coordinates, preserving kernel normalization.
    """
    if estimate_a.ndim != 2 or estimate_a.shape != estimate_b.shape or estimate_a.shape != target.shape:
        raise ValueError("aligned (row, feature) estimates and targets are required")
    return (estimate_a * estimate_b - (estimate_a + estimate_b) * target.detach()).sum(dim=-1)
