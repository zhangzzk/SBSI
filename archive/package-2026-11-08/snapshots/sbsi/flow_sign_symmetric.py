"""Conditional physical flow equivariant to independent shape sign flips.

The first two context coordinates are signed, zero-centred input shapes.
Each affine conditioner sees invariants of the context and its unchanged
coordinates. Shape shifts have the corresponding context shape as a factor.
No group averaging, absolute-value folding, or duplicated density call is used.
"""

import numpy as np
import torch
from torch import nn

from .measurement_model import ConditionalAffineCoupling, ConditionalAffineFlow
from .flow_physical_disk import ConditionalPhysicalDiskFlow


class SignSymmetricCoupling(ConditionalAffineCoupling):
    def __init__(self, target_dim, context_dim, invariant_features="bounded", **kwargs):
        if target_dim != 4 or context_dim < 2:
            raise ValueError("sign symmetry requires four outputs and two shape conditions")
        if invariant_features not in ("bounded", "asinh"):
            raise ValueError("unknown sign-symmetric invariant feature map")
        # Two extra invariants retain relative signs between unchanged shape
        # coordinates and their corresponding conditioning shapes.
        super().__init__(target_dim, context_dim + 2, **kwargs)
        self.invariant_features = invariant_features

    def _shift_log_scale(self, x, context):
        masked = x * self.mask
        # A smooth odd bijection to (-1, 1) prevents quadratic amplification
        # of internal tail coordinates. Only conditioner features are mapped;
        # the invertible flow coordinates, density and data are not clipped.
        if self.invariant_features == "asinh":
            # Logarithmic growth preserves tail distinctions while avoiding
            # raw quadratic amplification. Oddness preserves both symmetries.
            odd_x, odd_c = torch.asinh(masked[..., :2]), torch.asinh(context[..., :2])
        else:
            odd_x = masked[..., :2] / torch.hypot(masked[..., :2], torch.ones_like(masked[..., :2]))
            odd_c = context[..., :2] / torch.hypot(context[..., :2], torch.ones_like(context[..., :2]))
        invariant_x = torch.cat((odd_x.square(), masked[..., 2:]), -1)
        invariant_context = torch.cat((odd_c.square(), context[..., 2:]), -1)
        relative = odd_x * odd_c
        parameters = self.net(torch.cat((invariant_x, invariant_context, relative), -1))
        coefficients, log_scale = parameters.chunk(2, -1)
        shift = torch.cat((coefficients[..., :2] * context[..., :2], coefficients[..., 2:]), -1)
        transformed = 1.0 - self.mask
        return shift * transformed, torch.tanh(log_scale) * self.scale_limit * transformed


class SignSymmetricAffineFlow(ConditionalAffineFlow):
    def __init__(self, target_dim, context_dim, hidden_dim=256, n_layers=3,
                 n_flows=8, scale_limit=3.0, activation="silu", invariant_features="bounded", **kwargs):
        nn.Module.__init__(self)
        if n_flows < 1 or target_dim != 4:
            raise ValueError("positive layer count and four targets required")
        self.target_dim, self.context_dim = int(target_dim), int(context_dim)
        mask = np.array([0, 1, 0, 1], dtype=np.float32)
        self.layers = nn.ModuleList([
            SignSymmetricCoupling(target_dim, context_dim, hidden_dim=hidden_dim,
                n_layers=n_layers, mask=mask if i % 2 == 0 else 1 - mask,
                scale_limit=scale_limit, activation=activation, invariant_features=invariant_features)
            for i in range(n_flows)
        ])


class SignSymmetricPhysicalDiskFlow(ConditionalPhysicalDiskFlow):
    """Four physical outputs with Z2 x Z2 conditional shape equivariance."""

    def __init__(self, target_dim, context_dim, invariant_features="bounded", **kwargs):
        super().__init__(target_dim, context_dim, **kwargs)
        self.invariant_features = invariant_features
        self.base = SignSymmetricAffineFlow(target_dim, context_dim,
            invariant_features=invariant_features, **kwargs)
        self.validate_symmetry_coordinates()

    def validate_symmetry_coordinates(self):
        for name in ("disk_shape_means", "disk_coordinate_means"):
            if not bool((getattr(self, name) == 0).all()):
                raise ValueError(f"sign-symmetric flow requires zero {name}")

    def validate_condition_preprocessor(self, preprocessor):
        self.validate_symmetry_coordinates()
        if preprocessor.feature_names[:2] != ["e1_input_p", "e2_input_p"]:
            raise ValueError("sign symmetry requires e1_input_p/e2_input_p first")
        if preprocessor.output_dim != self.context_dim:
            raise ValueError("sign-symmetric context dimensions differ")
        if (np.any(preprocessor.means[:2] != 0)
                or np.any(preprocessor.fill_values[:2] != 0)
                or set(preprocessor.feature_names[:2]) & set(preprocessor.log_features)):
            raise ValueError("sign-symmetric shape conditions require zero mean/fill and no log")


def validate_symmetric_bundle(model, preprocessor):
    if isinstance(model, SignSymmetricPhysicalDiskFlow):
        model.validate_condition_preprocessor(preprocessor)
