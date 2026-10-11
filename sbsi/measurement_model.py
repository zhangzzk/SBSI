"""The conditional measurement flow: loading, drawing and evaluating it.

The flow is the normalized likelihood p(measured shape, size, flux | truth,
crowding) of a galaxy that is detected with a usable measurement. It does not
include that detection probability; the classifier gives it separately.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from .nn_utils import activation_class as _activation
from .selection_model import TabularPreprocessor


@dataclass
class TargetStandardizer:
    target_names: list[str]
    means: np.ndarray
    scales: np.ndarray
    dtype: str = "float32"

    def __post_init__(self):
        if self.dtype not in ("float32", "float64"):
            raise ValueError("target dtype must be float32 or float64")

    @property
    def dim(self):
        return len(self.target_names)

    def transform_frame(self, frame):
        values = frame[self.target_names].to_numpy(dtype=self.dtype, copy=True)
        return ((values - self.means) / self.scales).astype(self.dtype, copy=False)

    @classmethod
    def from_state(cls, state):
        dtype = state.get("dtype", "float32")
        return cls(
            list(state["target_names"]),
            np.asarray(state["means"], dtype=dtype),
            np.asarray(state["scales"], dtype=dtype),
            dtype=dtype,
        )


class ConditionalAffineCoupling(nn.Module):
    def __init__(
        self,
        target_dim,
        context_dim,
        hidden_dim=256,
        n_layers=3,
        mask=None,
        scale_limit=3.0,
        activation="silu",
    ):
        super().__init__()
        if n_layers < 1:
            raise ValueError("n_layers must be >= 1")
        if mask is None:
            mask = self._alternating_mask(target_dim)
        mask = torch.as_tensor(mask, dtype=torch.float32)
        if mask.numel() != target_dim:
            raise ValueError("mask length must match target_dim")
        if float(mask.sum()) <= 0.0 or float(mask.sum()) >= float(target_dim):
            raise ValueError("coupling mask must leave at least one identity and transformed dimension")
        self.register_buffer("mask", mask)
        self.scale_limit = float(scale_limit)

        act = _activation(activation)
        layers = []
        dim = target_dim + context_dim
        for _ in range(n_layers):
            layers.append(nn.Linear(dim, hidden_dim))
            layers.append(act())
            dim = hidden_dim
        layers.append(nn.Linear(dim, 2 * target_dim))
        self.net = nn.Sequential(*layers)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    @staticmethod
    def _alternating_mask(target_dim):
        return [(i % 2) for i in range(target_dim)]

    def _shift_log_scale(self, x, context):
        masked_x = x * self.mask
        params = self.net(torch.cat([masked_x, context], dim=-1))
        shift, log_scale = params.chunk(2, dim=-1)
        transform_mask = 1.0 - self.mask
        shift = shift * transform_mask
        log_scale = torch.tanh(log_scale) * self.scale_limit * transform_mask
        return shift, log_scale

    def forward(self, z, context):
        shift, log_scale = self._shift_log_scale(z, context)
        x = z * self.mask + (1.0 - self.mask) * (z * torch.exp(log_scale) + shift)
        logdet = log_scale.sum(dim=-1)
        return x, logdet

    def inverse(self, x, context):
        shift, log_scale = self._shift_log_scale(x, context)
        z = x * self.mask + (1.0 - self.mask) * ((x - shift) * torch.exp(-log_scale))
        logdet = -log_scale.sum(dim=-1)
        return z, logdet


class ConditionalAffineFlow(nn.Module):
    def __init__(
        self,
        target_dim,
        context_dim,
        hidden_dim=256,
        n_layers=3,
        n_flows=8,
        scale_limit=3.0,
        activation="silu",
        **kwargs,  # checkpoint configs also carry keys of retired flow types
    ):
        super().__init__()
        if target_dim < 2:
            raise ValueError("target_dim must be >= 2 for affine coupling flows")
        if n_flows < 1:
            raise ValueError("n_flows must be >= 1")
        masks = []
        base = np.array([(i % 2) for i in range(target_dim)], dtype=np.float32)
        for i in range(n_flows):
            masks.append(base if i % 2 == 0 else 1.0 - base)
        self.target_dim = int(target_dim)
        self.context_dim = int(context_dim)
        self.layers = nn.ModuleList(
            [
                ConditionalAffineCoupling(
                    target_dim,
                    context_dim,
                    hidden_dim=hidden_dim,
                    n_layers=n_layers,
                    mask=mask,
                    scale_limit=scale_limit,
                    activation=activation,
                )
                for mask in masks
            ]
        )

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


class MeasurementModelBundle:
    def __init__(self, model, condition_preprocessor, target_transform, metadata=None, device="cpu"):
        _validate_physical_contract(model, target_transform)
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.model.eval()
        self.condition_preprocessor = condition_preprocessor
        self.target_transform = target_transform
        self.metadata = metadata or {}

    def context_tensor(self, frame):
        """Return the standardized flow conditions of ``frame`` on the bundle device."""

        context = self.condition_preprocessor.transform_frame(frame)
        return torch.as_tensor(context, dtype=torch.float32, device=self.device)

    @torch.no_grad()
    def log_prob(self, frame, batch_size=65536):
        """Log likelihood of each row's measured targets given its conditions."""

        values = []
        for start in range(0, len(frame), batch_size):
            batch = frame.iloc[start : start + batch_size]
            target = torch.as_tensor(self.target_transform.transform_frame(batch), device=self.device)
            values.append(self.model.log_prob(target, self.context_tensor(batch)).cpu().numpy())
        return np.concatenate(values) if values else np.array([], dtype=self.target_transform.dtype)

    @torch.no_grad()
    def sample(self, condition_frame, n_samples=1, batch_size=65536):
        """Draw measured targets, shape ``(rows, n_samples, targets)``, in physical units."""

        samples = []
        for start in range(0, len(condition_frame), batch_size):
            context = self.context_tensor(condition_frame.iloc[start : start + batch_size])
            samples.append(self.model.sample_physical(context, n_samples=n_samples).cpu().numpy())
        if not samples:
            return np.empty((0, n_samples, self.target_transform.dim), dtype=self.target_transform.dtype)
        return np.concatenate(samples, axis=0)


def build_flow(model_config):
    """Construct the physical-disk flow from a checkpoint's config."""

    from .flow_physical_disk import ConditionalPhysicalDiskFlow

    cfg = dict(model_config)
    flow_type = cfg.pop("flow_type", None)
    # Training-only response-loss metadata does not affect the density.
    for key in ("response_difference", "response_error", "response_components"):
        cfg.pop(key, None)
    if flow_type != "physical_disk_affine":
        raise ValueError(f"unsupported flow type {flow_type!r}; only physical_disk_affine is shipped")
    return ConditionalPhysicalDiskFlow(**cfg)


def _validate_physical_contract(model, target_transform, model_config=None):
    """Check that the flow's stored standardization matches its target metadata."""

    if target_transform is None or target_transform.dtype != "float64":
        raise ValueError("the physical flow requires float64 target metadata")
    if target_transform.target_names != [
        "measured_ngmix_g1", "measured_ngmix_g2", "measured_flux_radius", "measured_flux_from_mag_auto",
    ]:
        raise ValueError("the physical flow requires ngmix g1/g2, flux radius and flux targets")
    if target_transform.dim != model.target_dim:
        raise ValueError("flow and target dimensions differ")
    for name, value in (
        ("disk_shape_means", target_transform.means[:2]),
        ("disk_shape_scales", target_transform.scales[:2]),
        ("physical_means", target_transform.means[2:]),
        ("physical_scales", target_transform.scales[2:]),
    ):
        if not np.array_equal(getattr(model, name).detach().cpu().numpy(), value):
            raise ValueError(f"flow {name} differs from the target metadata")
    if model_config is not None:
        if model_config.get("disk_map") != "radial_tanh" or model_config.get("physical_map") != "softplus":
            raise ValueError("the physical flow requires radial_tanh and softplus maps")
        for name in model.coordinate_buffer_names:
            if not np.array_equal(getattr(model, name).detach().cpu().numpy(), model_config.get(name)):
                raise ValueError(f"flow state and config disagree on {name}")


def load_measurement_model(path, device="cpu"):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    condition_preprocessor = TabularPreprocessor.from_state(checkpoint["condition_preprocessor"])
    target_transform = TargetStandardizer.from_state(checkpoint["target_transform"])
    model = build_flow(checkpoint["model_config"])
    model.load_state_dict(checkpoint["state_dict"])
    _validate_physical_contract(model, target_transform, checkpoint["model_config"])
    return MeasurementModelBundle(
        model,
        condition_preprocessor,
        target_transform,
        metadata=checkpoint.get("metadata", {}),
        device=device,
    )
