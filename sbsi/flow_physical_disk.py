"""Disk shapes and positive physical radius/flux, with a full density Jacobian."""

import torch
from torch.nn import functional as F
from .flow_disk import ConditionalDiskAffineFlow, radial_tanh
from .measurement_model import _qmc_normal


def positive_inverse(value):
    """Inverse softplus for strictly positive float64 values, without clipping."""
    return value + torch.log(-torch.expm1(-value))


class ConditionalPhysicalDiskFlow(ConditionalDiskAffineFlow):
    physical_buffer_names = (
        "physical_means",
        "physical_scales",
        "physical_units",
        "physical_coordinate_means",
        "physical_coordinate_scales",
    )
    coordinate_buffer_names = ConditionalDiskAffineFlow.coordinate_buffer_names + physical_buffer_names

    def __init__(
        self,
        *args,
        physical_map,
        physical_means,
        physical_scales,
        physical_units,
        physical_coordinate_means,
        physical_coordinate_scales,
        **kwargs,
    ):
        if physical_map != "softplus":
            raise ValueError("physical map must be softplus")
        super().__init__(*args, **kwargs)
        for name, value in zip(
            self.physical_buffer_names,
            (
                physical_means,
                physical_scales,
                physical_units,
                physical_coordinate_means,
                physical_coordinate_scales,
            ),
        ):
            value = torch.as_tensor(value, dtype=torch.float64)
            if value.shape != (2,) or not bool(torch.isfinite(value).all()):
                raise ValueError(name)
            if ("scales" in name or name == "physical_units") and not bool((value > 0).all()):
                raise ValueError(name)
            self.register_buffer(name, value.clone())

    def forward_physical(self, z, context):
        """Return physical draws directly, avoiding subtract/add cancellation.

        The determinant here is with respect to physical e1,e2,r,F. No draw
        is clipped; softplus underflow and disk rounding remain observable.
        """
        self._require_precision()
        w, ld = self.base.forward(z, context)
        e, shape_ld = radial_tanh(
            w[..., :2].double() * self.disk_coordinate_scales + self.disk_coordinate_means
        )
        u = w[..., 2:].double() * self.physical_coordinate_scales + self.physical_coordinate_means
        p = self.physical_units * torch.logaddexp(u, torch.zeros_like(u))
        extra = (self.physical_units.log() + self.physical_coordinate_scales.log() + F.logsigmoid(u)).sum(-1)
        return torch.cat(
            (e, p), dim=-1
        ), ld.double() + shape_ld + self.disk_coordinate_scales.log().sum() + extra

    def forward(self, z, context):
        physical, ld = self.forward_physical(z, context)
        means = torch.cat((self.disk_shape_means, self.physical_means))
        scales = torch.cat((self.disk_shape_scales, self.physical_scales))
        return (physical - means) / scales, ld - scales.log().sum()

    def sample_physical(self, context, n_samples=1, qmc=False):
        """Physical samples using the same base draw convention as sample()."""
        if n_samples < 1 or context.ndim != 2:
            raise ValueError("positive draws and matrix contexts required")
        batch = len(context)
        expanded = (
            context[:, None, :].expand(batch, n_samples, self.context_dim).reshape(-1, self.context_dim)
        )
        z = (
            _qmc_normal(batch, n_samples, self.target_dim, context.dtype, context.device)
            if qmc
            else torch.randn(batch * n_samples, self.target_dim, dtype=context.dtype, device=context.device)
        )
        physical, _ = self.forward_physical(z, expanded)
        return physical.reshape(batch, n_samples, self.target_dim)

    def inverse(self, y, context):
        self._require_precision()
        if y.dtype != torch.float64:
            raise ValueError("physical density requires float64 targets")
        p = y[..., 2:] * self.physical_scales + self.physical_means
        valid = torch.isfinite(y).all(-1) & (p > 0).all(-1)
        safe = torch.where(valid[..., None], p, torch.ones_like(p))
        u = positive_inverse(safe / self.physical_units)
        w = (u - self.physical_coordinate_means) / self.physical_coordinate_scales
        z, ld = super().inverse(torch.cat((y[..., :2], w), dim=-1), context)
        extra = (
            self.physical_scales.log()
            - self.physical_units.log()
            - self.physical_coordinate_scales.log()
            - F.logsigmoid(u)
        ).sum(-1)
        return z, torch.where(valid, ld + extra, torch.full_like(ld, -torch.inf))
