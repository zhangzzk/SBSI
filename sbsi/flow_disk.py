"""A plain conditional density with physical ngmix shape support.

The first two affine outputs are standardized *unbounded coordinates*, not
ellipticities.  A radial tanh bijection maps each draw into the physical disk
before any expectation or shear difference.  Remaining targets retain their
existing standardization.  There is no mean head or response correction.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

from .measurement_model import ConditionalAffineFlow, _qmc_normal


def _radius_parts(x):
    if x.shape[-1] != 2:
        raise ValueError("disk coordinates must have final dimension two")
    x = x.to(torch.float64)
    radius2 = x.square().sum(dim=-1)
    small = radius2 < 1.e-6
    # The inactive sqrt/hypot branch must also have a finite derivative at zero.
    safe = torch.where(small[..., None], torch.ones_like(x), x)
    radius = torch.hypot(safe[..., 0], safe[..., 1])
    series_r2 = torch.where(small, radius2, torch.zeros_like(radius2))
    return x, series_r2, small, radius


def radial_tanh(x):
    """R^2 -> open disk, with analytic log|det De/Dx| in float64.

    At large radii (about 19 or more), floating-point tanh can round to 1.
    Samples are returned at their representable value; they are never clipped
    or contracted.  The analytic log Jacobian remains finite there.  A rounded
    boundary sample cannot be inverted with finite precision.  Likelihoods
    must use original, strictly interior measured shapes.
    """
    x, r2, small, r = _radius_parts(x)
    ratio_small = 1 + r2 * (-1 / 3 + r2 * (2 / 15 - r2 * 17 / 315))
    ratio = torch.where(small, ratio_small, torch.tanh(r) / r)
    log_radial_small = r2 * (-1 + r2 * (1 / 6 - r2 * 2 / 45))
    log_radial = torch.where(small, log_radial_small,
                             2 * (math.log(2.) - r - F.softplus(-2 * r)))
    return x * ratio[..., None], log_radial + torch.log(ratio)


def radial_atanh(e, *, validate=True):
    """Open disk -> R^2 and inverse log Jacobian, without boundary repair."""
    e, s2, small, s = _radius_parts(e)
    valid = torch.isfinite(e).all(dim=-1) & (e.square().sum(dim=-1) < 1)
    if validate and not bool(valid.all()):
        raise ValueError("radial_atanh requires finite, strictly interior shapes")
    # s is artificial in the small branch; atanh must never see that value.
    s = torch.where(small, torch.full_like(s, .5), s)
    # Use the squared-radius gap, which can remain representable even if
    # hypot rounds to one for a shape only one ulp inside the boundary.
    actual_s2 = e.square().sum(dim=-1)
    safe_s2 = torch.where(small, torch.full_like(s2, .25), actual_s2)
    r = torch.log1p(s) - .5 * torch.log1p(-safe_s2)
    ratio_small = 1 + s2 * (1 / 3 + s2 * (1 / 5 + s2 / 7))
    ratio = torch.where(small, ratio_small, r / s)
    log_radial = -torch.log1p(-actual_s2)
    return e * ratio[..., None], log_radial + torch.log(ratio)


class ConditionalDiskAffineFlow(nn.Module):
    """Affine flow followed by a physical disk bijection and standardization.

    Parameters/contexts can stay float32.  Shape coordinates and Jacobians are
    float64, as are standardized physical inputs and outputs.  The density
    reference measure is the *original standardized measurement* measure;
    both shape standardizations appear in the Jacobian explicitly.
    """

    coordinate_buffer_names = (
        "disk_shape_means", "disk_shape_scales",
        "disk_coordinate_means", "disk_coordinate_scales",
    )

    def __init__(self, target_dim, context_dim, *, disk_map,
                 disk_shape_means, disk_shape_scales,
                 disk_coordinate_means, disk_coordinate_scales, **kwargs):
        super().__init__()
        if target_dim < 2 or disk_map != "radial_tanh":
            raise ValueError("disk_affine requires at least two targets and disk_map='radial_tanh'")
        if kwargs.get("mean_hidden", 0) != 0 or any(k.startswith("disk_") for k in kwargs):
            raise ValueError("disk_affine has no mean head or additional disk options")
        self.target_dim = int(target_dim)
        self.context_dim = int(context_dim)
        self.base = ConditionalAffineFlow(target_dim, context_dim, **kwargs)
        values = (disk_shape_means, disk_shape_scales,
                  disk_coordinate_means, disk_coordinate_scales)
        for name, value in zip(self.coordinate_buffer_names, values):
            value = torch.as_tensor(value, dtype=torch.float64)
            if value.shape != (2,) or not bool(torch.isfinite(value).all()):
                raise ValueError(f"{name} must contain two finite values")
            if name.endswith("scales") and not bool((value > 0).all()):
                raise ValueError(f"{name} must be strictly positive")
            self.register_buffer(name, value.clone())

    def _require_precision(self):
        if any(getattr(self, name).dtype != torch.float64 for name in self.coordinate_buffer_names):
            raise ValueError("disk coordinate buffers must remain float64; do not call model.float()")

    def forward(self, z, context):
        self._require_precision()
        w, logdet = self.base.forward(z, context)
        u = w[..., :2].double() * self.disk_coordinate_scales + self.disk_coordinate_means
        e, disk_ldj = radial_tanh(u)
        yshape = (e - self.disk_shape_means) / self.disk_shape_scales
        y = torch.cat((yshape, w[..., 2:].double()), dim=-1)
        scale_ldj = (self.disk_coordinate_scales.log() - self.disk_shape_scales.log()).sum()
        return y, logdet.double() + disk_ldj + scale_ldj

    def inverse(self, y, context):
        self._require_precision()
        if y.dtype != torch.float64:
            raise ValueError("disk density requires float64 standardized original targets")
        e = y[..., :2] * self.disk_shape_scales + self.disk_shape_means
        valid = torch.isfinite(y).all(dim=-1) & (e.square().sum(dim=-1) < 1)
        # Invalid observations get density zero.  Evaluate safe inactive branches
        # to keep valid rows' gradients finite, without moving invalid data inside.
        safe_e = torch.where(valid[..., None], e, torch.zeros_like(e))
        u, disk_ldj = radial_atanh(safe_e, validate=False)
        wshape = (u - self.disk_coordinate_means) / self.disk_coordinate_scales
        w = torch.cat((wshape, torch.where(valid[..., None], y[..., 2:], torch.zeros_like(y[..., 2:]))), dim=-1)
        w = w.to(dtype=next(self.base.parameters()).dtype)
        z, logdet = self.base.inverse(w, context)
        scale_ldj = (self.disk_shape_scales.log() - self.disk_coordinate_scales.log()).sum()
        logdet = logdet.double() + disk_ldj + scale_ldj
        return z, torch.where(valid, logdet, torch.full_like(logdet, -torch.inf))

    def log_prob(self, y, context):
        z, logdet = self.inverse(y, context)
        log_base = -.5 * (z.double().square() + math.log(2 * math.pi)).sum(dim=-1)
        return log_base + logdet

    def sample(self, context, n_samples=1, qmc=False):
        if n_samples < 1 or context.ndim != 2:
            raise ValueError("positive n_samples and a matrix of contexts are required")
        batch = len(context)
        flat_context = context[:, None, :].expand(batch, n_samples, self.context_dim).reshape(-1, self.context_dim)
        if qmc:
            z = _qmc_normal(batch, n_samples, self.target_dim, context.dtype, context.device)
        else:
            z = torch.randn(batch * n_samples, self.target_dim, dtype=context.dtype, device=context.device)
        y, _ = self.forward(z, flat_context)
        return y.reshape(batch, n_samples, self.target_dim)
