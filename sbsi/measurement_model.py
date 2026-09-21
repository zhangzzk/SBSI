"""Conditional density model for selected measured catalogue properties.

The measurement model represents the normalized likelihood

    p_meas(xhat | truth, neighbour, shear, s=1)

for rows that already satisfy the current SBSI source cuts and selection event.
It intentionally does not include the Bernoulli selection factor; compose this
with the selection classifier when using the unnormalized catalogue density.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn

from .nn_utils import activation_class as _activation
from .coordinates import angle_to_radians
from .selection_model import TabularPreprocessor

_ENGINEERED_TARGET_RAW_COLUMNS = {
    "measured_log_flux_auto": {"measured_flux_auto"},
    "measured_log_flux_radius": {"measured_flux_radius"},
    "measured_log_a_image": {"measured_a_image"},
    "measured_log_b_image": {"measured_b_image"},
    "measured_e1_image": {"measured_a_image", "measured_b_image", "measured_theta_image"},
    "measured_e2_image": {"measured_a_image", "measured_b_image", "measured_theta_image"},
}


def _safe_log_positive(values):
    values = np.asarray(values, dtype=float)
    out = np.full(values.shape, np.nan, dtype=float)
    good = np.isfinite(values) & (values > 0.0)
    out[good] = np.log(values[good])
    return out


def raw_columns_for_measurement_targets(targets: Iterable[str]) -> set[str]:
    """Return catalogue columns needed before target feature engineering."""
    columns: set[str] = set()
    for name in targets:
        if name in _ENGINEERED_TARGET_RAW_COLUMNS:
            columns.update(_ENGINEERED_TARGET_RAW_COLUMNS[name])
        else:
            columns.add(name)
    return columns


def add_measurement_target_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add stable continuous measured-property targets in-place and return frame."""
    if "measured_flux_auto" in frame.columns:
        frame["measured_log_flux_auto"] = _safe_log_positive(frame["measured_flux_auto"].array)
    if "measured_flux_radius" in frame.columns:
        frame["measured_log_flux_radius"] = _safe_log_positive(frame["measured_flux_radius"].array)
    if "measured_a_image" in frame.columns:
        frame["measured_log_a_image"] = _safe_log_positive(frame["measured_a_image"].array)
    if "measured_b_image" in frame.columns:
        frame["measured_log_b_image"] = _safe_log_positive(frame["measured_b_image"].array)

    needed = {"measured_a_image", "measured_b_image", "measured_theta_image"}
    if needed.issubset(frame.columns):
        a = np.asarray(frame["measured_a_image"], dtype=float)
        b = np.asarray(frame["measured_b_image"], dtype=float)
        theta = angle_to_radians(frame["measured_theta_image"].array)
        denom = a + b
        eps = np.full(len(frame), np.nan, dtype=float)
        good = (
            np.isfinite(a)
            & np.isfinite(b)
            & np.isfinite(theta)
            & (a > 0.0)
            & (b > 0.0)
            & np.isfinite(denom)
            & (denom > 0.0)
        )
        eps[good] = (a[good] - b[good]) / denom[good]
        frame["measured_e1_image"] = eps * np.cos(2.0 * theta)
        frame["measured_e2_image"] = eps * np.sin(2.0 * theta)
    return frame


@dataclass
class TargetStandardizer:
    target_names: list[str]
    means: np.ndarray
    scales: np.ndarray
    dtype: str = "float32"

    def __post_init__(self):
        if self.dtype not in ("float32", "float64"):
            raise ValueError("target dtype must be float32 or float64")

    @classmethod
    def fit(cls, frame: pd.DataFrame, target_names: Sequence[str], *, dtype="float32"):
        if dtype not in ("float32", "float64"):
            raise ValueError("target dtype must be float32 or float64")
        missing = [name for name in target_names if name not in frame.columns]
        if missing:
            raise KeyError(f"Missing target columns: {missing}")
        values = frame[list(target_names)].to_numpy(dtype=dtype, copy=True)
        finite = np.isfinite(values)
        if not np.all(finite):
            bad = {
                name: int((~finite[:, i]).sum())
                for i, name in enumerate(target_names)
                if np.any(~finite[:, i])
            }
            raise ValueError(f"Cannot fit target transform with non-finite values: {bad}")
        means = values.mean(axis=0, dtype=np.float64).astype(dtype)
        scales = values.std(axis=0, ddof=1).astype(dtype)
        scales[~np.isfinite(scales) | (scales < 1e-6)] = 1.0
        return cls(list(target_names), means, scales, dtype=dtype)

    @property
    def dim(self):
        return len(self.target_names)

    def transform_frame(self, frame: pd.DataFrame):
        raw = frame[self.target_names].to_numpy(dtype=self.dtype, copy=True)
        return self.transform_array(raw)

    def transform_array(self, values):
        values = np.asarray(values, dtype=self.dtype)
        return ((values - self.means) / self.scales).astype(self.dtype, copy=False)

    def inverse_transform_array(self, values):
        values = np.asarray(values, dtype=self.dtype)
        return (values * self.scales + self.means).astype(self.dtype, copy=False)

    def to_state(self):
        state = {
            "target_names": self.target_names,
            "means": self.means,
            "scales": self.scales,
        }
        # Preserve legacy checkpoint metadata unless double precision is explicit.
        if self.dtype != "float32":
            state["dtype"] = self.dtype
        return state

    @classmethod
    def from_state(cls, state):
        dtype = state.get("dtype", "float32")
        if dtype not in ("float32", "float64"):
            raise ValueError("target dtype must be float32 or float64")
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


def _qmc_normal(batch, n_samples, dim, dtype, device):
    """Randomized-QMC standard normals, shape (batch*n_samples, dim), row-major (obj, sample).

    The per-object integral we need is E[f(x)] over the flow's predictive distribution, estimated
    from n_samples draws. Plain `torch.randn` converges as 1/sqrt(n); a low-discrepancy (Sobol) set
    covers the unit cube far more evenly and converges nearer 1/n on smooth integrands, so the same
    accuracy needs far fewer draws.

    THE RANDOM SHIFT IS NOT OPTIONAL. If every object reused the SAME Sobol points, their quadrature
    errors would be perfectly correlated and would NOT average away over the ~1e6 objects -- that
    turns a variance into a BIAS, which is exactly the failure mode this estimator cannot tolerate.
    Giving each object its own uniform shift mod 1 (a randomly-shifted digital net) keeps the
    low-discrepancy structure WITHIN an object's n_samples while decorrelating objects, so the
    estimator stays unbiased and the object-average still converges.

    n_samples should be a power of two: Sobol's equidistribution guarantees hold on 2^k points and
    degrade for other counts.
    """
    eng = torch.quasirandom.SobolEngine(dimension=dim, scramble=True, seed=_QMC_SEED)
    u = eng.draw(n_samples).to(device=device, dtype=torch.float32)  # (n_samples, dim)
    shift = torch.rand(batch, 1, dim, device=device, dtype=torch.float32)  # per-object shift
    u = torch.remainder(u[None, :, :] + shift, 1.0)  # (batch, n_samples, dim)
    # ndtri(0)= -inf, ndtri(1)=+inf; clamp just inside the open interval.
    u = u.clamp_(1e-7, 1.0 - 1e-7)
    z = torch.special.ndtri(u)
    return z.reshape(batch * n_samples, dim).to(dtype)


_QMC_SEED = 0


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
        **kwargs,  # ignore spline-only keys so a single config dict works for both
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

    def log_prob(self, x, context):
        z, logdet = self.inverse(x, context)
        log_base = -0.5 * (z.pow(2) + np.log(2.0 * np.pi)).sum(dim=-1)
        return log_base + logdet

    def sample(self, context, n_samples=1, qmc=False):
        if n_samples < 1:
            raise ValueError("n_samples must be >= 1")
        if context.ndim != 2:
            raise ValueError("context must have shape (batch, context_dim)")
        batch = context.shape[0]
        expanded_context = context[:, None, :].expand(batch, n_samples, self.context_dim)
        flat_context = expanded_context.reshape(batch * n_samples, self.context_dim)
        if qmc:
            z = _qmc_normal(batch, n_samples, self.target_dim, context.dtype, context.device)
        else:
            z = torch.randn(batch * n_samples, self.target_dim, dtype=context.dtype, device=context.device)
        flat_x, _ = self.forward(z, flat_context)
        return flat_x.reshape(batch, n_samples, self.target_dim)


class MeasurementModelBundle:
    def __init__(self, model, condition_preprocessor, target_transform, metadata=None, device="cpu"):
        _validate_disk_contract(model, target_transform)
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.model.eval()
        self.condition_preprocessor = condition_preprocessor
        self.target_transform = target_transform
        self.metadata = metadata or {}

    def _context_from_frame(self, frame):
        context = self.condition_preprocessor.transform_frame(frame)
        return torch.as_tensor(context, dtype=torch.float32, device=self.device)

    def _targets_from_frame(self, frame):
        missing = [name for name in self.target_transform.target_names if name not in frame.columns]
        if missing:
            frame = add_measurement_target_features(frame.copy())
        targets = self.target_transform.transform_frame(frame)
        return torch.as_tensor(targets, device=self.device)

    def context_tensor(self, frame):
        """Return standardized flow conditions on the bundle device.

        Catalogue inference uses this lower-level interface to transform a
        model view once, keep it resident on the accelerator, and gather rows
        without rebuilding a pandas frame for every atom block.
        """

        return self._context_from_frame(frame)

    def target_tensor(self, frame):
        """Return standardized measurement targets on the bundle device."""

        return self._targets_from_frame(frame)

    def compile_log_prob(self, *, mode=None, dynamic=True):
        """Compile the named density hot path used by catalogue inference.

        ``torch.compile(module)`` only intercepts ``module(...)``.  SBSI calls
        ``model.log_prob(...)`` directly, so compiling the module itself is a silent
        no-op for inference.  This method intentionally compiles that bound method and
        leaves sampling eager.  Repeating the same request is harmless; changing the
        compilation contract on an already-compiled bundle is rejected.
        """

        requested = (mode, bool(dynamic))
        existing = getattr(self, "_compiled_log_prob_config", None)
        if existing is not None:
            if existing != requested:
                raise RuntimeError("measurement log_prob is already compiled with a different configuration")
            return self
        if not hasattr(torch, "compile"):
            raise RuntimeError("this PyTorch build does not provide torch.compile")
        self.model.log_prob = torch.compile(
            self.model.log_prob,
            mode=mode,
            dynamic=bool(dynamic),
        )
        self._compiled_log_prob_config = requested
        return self

    def log_prob_tensor(self, target, context, *, batch_size=65536):
        """Evaluate standardized tensors without a host round trip.

        Unlike :meth:`log_prob`, this method deliberately leaves the result on
        the device and does not disable autograd.  Callers control graph
        construction with their surrounding ``torch.no_grad`` context.
        """

        if target.ndim != 2 or context.ndim != 2 or len(target) != len(context):
            raise ValueError("target and context must be aligned matrices")
        if target.device != context.device or target.device.type != self.device.type:
            raise ValueError("target and context must be on the bundle device")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        values = []
        for start in range(0, len(target), batch_size):
            stop = min(start + batch_size, len(target))
            values.append(self.model.log_prob(target[start:stop], context[start:stop]))
        if values:
            return torch.cat(values)
        return torch.empty(0, dtype=target.dtype, device=target.device)

    @torch.no_grad()
    def log_prob(self, frame, batch_size=65536):
        values = []
        for start in range(0, len(frame), batch_size):
            batch = frame.iloc[start : start + batch_size]
            context = self._context_from_frame(batch)
            target = self._targets_from_frame(batch)
            values.append(self.model.log_prob(target, context).cpu().numpy())
        return np.concatenate(values) if values else np.array([], dtype=self.target_transform.dtype)

    @torch.no_grad()
    def sample(self, condition_frame, n_samples=1, batch_size=65536, qmc=False):
        samples = []
        for start in range(0, len(condition_frame), batch_size):
            batch = condition_frame.iloc[start : start + batch_size]
            context = self._context_from_frame(batch)
            if callable(getattr(self.model, "sample_physical", None)):
                samples.append(
                    self.model.sample_physical(context, n_samples=n_samples, qmc=qmc).cpu().numpy()
                )
                continue
            draw = self.model.sample(context, n_samples=n_samples, qmc=qmc).cpu().numpy()
            flat = draw.reshape(-1, self.target_transform.dim)
            raw = self.target_transform.inverse_transform_array(flat)
            samples.append(raw.reshape(len(batch), n_samples, self.target_transform.dim))
        if not samples:
            return np.empty((0, n_samples, self.target_transform.dim), dtype=self.target_transform.dtype)
        return np.concatenate(samples, axis=0)

    def target_mean_and_gradient(
        self,
        frame,
        gradient_features: Optional[Iterable[str]] = None,
        n_samples=64,
        batch_size=2048,
    ):
        """Estimate d E[xhat] / d condition features with reparameterized draws.

        The estimate is Monte Carlo noisy but differentiable through the flow.
        Returned means are in the engineered target units, not standardized flow
        units.
        """
        if gradient_features is None:
            gradient_features = self.condition_preprocessor.feature_names
        gradient_features = list(gradient_features)
        feature_to_idx = {name: i for i, name in enumerate(self.condition_preprocessor.feature_names)}
        grad_idx = [feature_to_idx[name] for name in gradient_features]
        target_scale = torch.as_tensor(
            self.target_transform.scales,
            dtype=torch.float64 if self.target_transform.dtype == "float64" else torch.float32,
            device=self.device,
        )
        target_mean = torch.as_tensor(
            self.target_transform.means,
            dtype=torch.float64 if self.target_transform.dtype == "float64" else torch.float32,
            device=self.device,
        )

        means = []
        grads = []
        for start in range(0, len(frame), batch_size):
            batch = frame.iloc[start : start + batch_size]
            raw_context = self.condition_preprocessor.raw_tensor_from_frame(
                batch, self.device, requires_grad=True
            )
            context = self.condition_preprocessor.transform_tensor(raw_context)
            draws = self.model.sample(context, n_samples=n_samples)
            mean_standard = draws.mean(dim=1)
            mean_raw = mean_standard * target_scale + target_mean
            grad_columns = []
            for j in range(self.target_transform.dim):
                grad = torch.autograd.grad(
                    mean_raw[:, j].sum(),
                    raw_context,
                    retain_graph=j < self.target_transform.dim - 1,
                    create_graph=False,
                )[0]
                grad_columns.append(grad[:, grad_idx].detach().cpu().numpy())
            means.append(mean_raw.detach().cpu().numpy())
            grads.append(np.stack(grad_columns, axis=1))

        mean_arr = np.concatenate(means) if means else np.empty((0, self.target_transform.dim))
        grad_arr = (
            np.concatenate(grads)
            if grads
            else np.empty((0, self.target_transform.dim, len(gradient_features)))
        )
        grad_names = [
            f"dE_{target}_d{feature}"
            for target in self.target_transform.target_names
            for feature in gradient_features
        ]
        grad_df = pd.DataFrame(grad_arr.reshape(len(mean_arr), -1), columns=grad_names)
        mean_df = pd.DataFrame(mean_arr, columns=[f"E_{name}" for name in self.target_transform.target_names])
        return mean_df, grad_df


class ConditionalMeanFlow(nn.Module):
    """Affine residual density with an explicit conditional shift ``mu(c)``.

    The residual flow may be blind to selected context columns so their
    response is carried only by the mean head. ``mu`` and the residual level
    are not separately identified; callers needing a model mean must sample
    the full density.
    """

    def __init__(
        self,
        target_dim,
        context_dim,
        base_flow="affine",
        mean_hidden=0,
        mean_activation="silu",
        flow_drop_indices=None,
        **kwargs,
    ):
        super().__init__()
        self.target_dim = int(target_dim)
        self.context_dim = int(context_dim)
        # The residual flow can be made BLIND to selected context columns (e.g. the shape
        # features) so the shape->shape response is forced entirely into the explicit mean
        # head and cannot be re-absorbed/shrunk by the flow. keep_indices = the columns the
        # residual flow still sees (noise-scale features).
        drop = sorted(set(int(i) for i in (flow_drop_indices or [])))
        keep = [i for i in range(context_dim) if i not in drop]
        self.register_buffer("keep_indices", torch.as_tensor(keep, dtype=torch.long))
        flow_context_dim = len(keep)
        if base_flow != "affine":
            raise ValueError("ConditionalMeanFlow supports only an affine residual flow")
        self.flow = ConditionalAffineFlow(
            target_dim=target_dim,
            context_dim=flow_context_dim,
            **kwargs,
        )
        if mean_hidden and mean_hidden > 0:
            act = _activation(mean_activation)
            self.mean_net = nn.Sequential(
                nn.Linear(context_dim, mean_hidden), act(), nn.Linear(mean_hidden, target_dim)
            )
        else:
            self.mean_net = nn.Linear(context_dim, target_dim)

    def _mu(self, context):
        return self.mean_net(context)

    def _flow_ctx(self, context):
        return context.index_select(1, self.keep_indices)

    @torch.no_grad()
    def set_ols_mean_and_freeze(self, context_std, target_std):
        """Fit the linear mean head by OLS on standardized (context, target) and freeze it.

        Pins the conditional mean to the data's best *linear* conditional mean E[x|c],
        a pure g=0 quantity.  With the residual flow kept blind to the shape features,
        the shape->shape response d<x>/d(shape) then equals the OLS data response M_data
        *by construction* and cannot be shrunk/re-absorbed by max-likelihood training.

        This is forward-model self-calibration derived entirely from the g=0 sample; it
        never uses the sheared-sim applied-shear truth (that would be a circular,
        illegitimate empirical m-removal).  Closure already shows recovery is unbiased
        given a faithful model, so pinning M_model == M_data targets the multiplicative
        bias at its root (the flow's conditional-mean under-fit).
        """
        if not isinstance(self.mean_net, nn.Linear):
            raise ValueError("OLS mean-freeze requires a linear mean head (mean_hidden=0)")
        C = np.asarray(context_std, dtype=np.float64)
        Y = np.asarray(target_std, dtype=np.float64)
        A = np.concatenate([C, np.ones((C.shape[0], 1), dtype=np.float64)], axis=1)
        sol, *_ = np.linalg.lstsq(A, Y, rcond=None)  # (context_dim+1, target_dim)
        W = sol[:-1, :].T.astype(np.float32)  # (target_dim, context_dim)
        b = sol[-1, :].astype(np.float32)  # (target_dim,)
        self.mean_net.weight.copy_(torch.as_tensor(W))
        self.mean_net.bias.copy_(torch.as_tensor(b))
        self.mean_net.weight.requires_grad_(False)
        self.mean_net.bias.requires_grad_(False)

    def log_prob(self, x, context):
        return self.flow.log_prob(x - self._mu(context), self._flow_ctx(context))

    def sample(self, context, n_samples=1, qmc=False):
        s = self.flow.sample(self._flow_ctx(context), n_samples=n_samples, qmc=qmc)
        return s + self._mu(context)[:, None, :]


def build_flow(model_config):
    """Construct the conditional flow from a config dict (dispatch on flow_type)."""
    cfg = dict(model_config)
    flow_type = cfg.pop("flow_type", "affine")
    # Training-only response-loss metadata may be saved with newer checkpoints.
    # It does not affect the density architecture.
    cfg.pop("response_difference", None)
    cfg.pop("response_error", None)
    cfg.pop("response_components", None)
    if flow_type not in ("disk_affine", "physical_disk_affine") and any(k.startswith("disk_") for k in cfg):
        raise ValueError("disk coordinate metadata requires flow_type='disk_affine'")
    retired = sorted(k for k in cfg if k.startswith("ra"))
    if retired or flow_type.endswith("_ra"):
        raise ValueError("realisation-aware experimental flows are archived")
    if flow_type != "physical_disk_affine" and any(k.startswith("physical_") for k in cfg):
        raise ValueError("physical coordinate metadata requires physical_disk_affine")
    if flow_type == "affine":
        return ConditionalAffineFlow(**cfg)
    if flow_type == "physical_disk_affine":
        from .flow_physical_disk import ConditionalPhysicalDiskFlow

        return ConditionalPhysicalDiskFlow(**cfg)
    if flow_type == "disk_affine":
        from .flow_disk import ConditionalDiskAffineFlow

        return ConditionalDiskAffineFlow(**cfg)
    if flow_type == "mean_affine":
        cfg["base_flow"] = "affine"
        return ConditionalMeanFlow(**cfg)
    raise ValueError(f"Unknown flow_type {flow_type!r}")


def _validate_disk_contract(model, target_transform, model_config=None):
    # Local import avoids a cycle: flow_disk uses the existing affine base.
    from .flow_disk import ConditionalDiskAffineFlow

    if not isinstance(model, ConditionalDiskAffineFlow):
        if model_config is not None and model_config.get("flow_type") == "disk_affine":
            raise ValueError("disk_affine config requires a disk flow")
        return
    if target_transform is None or target_transform.dtype != "float64":
        raise ValueError("disk_affine requires original float64 targets and target metadata")
    if target_transform.target_names[:2] != ["measured_ngmix_g1", "measured_ngmix_g2"]:
        raise ValueError("disk_affine requires ngmix g1/g2 as the first two targets")
    if target_transform.dim != model.target_dim:
        raise ValueError("disk flow and target dimensions differ")
    for name, value in (
        ("disk_shape_means", target_transform.means[:2]),
        ("disk_shape_scales", target_transform.scales[:2]),
    ):
        if not np.array_equal(getattr(model, name).detach().cpu().numpy(), value):
            raise ValueError(f"disk flow {name} differs from the target transform")
    from .flow_physical_disk import ConditionalPhysicalDiskFlow

    if isinstance(model, ConditionalPhysicalDiskFlow):
        if target_transform.target_names[2:] != ["measured_flux_radius", "measured_flux_from_mag_auto"]:
            raise ValueError("physical flow requires physical radius/flux target names")
        for name, value in (
            ("physical_means", target_transform.means[2:]),
            ("physical_scales", target_transform.scales[2:]),
        ):
            if not np.array_equal(getattr(model, name).detach().cpu().numpy(), value):
                raise ValueError("physical transform and target metadata differ")
        if model_config is not None and (
            model_config.get("flow_type") != "physical_disk_affine"
            or model_config.get("physical_map") != "softplus"
        ):
            raise ValueError("physical flow requires explicit physical config")
    if model_config is not None:
        if (
            model_config.get("flow_type") not in ("disk_affine", "physical_disk_affine")
            or model_config.get("disk_map") != "radial_tanh"
        ):
            raise ValueError("disk flow requires an explicit disk_affine / radial_tanh config")
        for name in model.coordinate_buffer_names:
            if not np.array_equal(getattr(model, name).detach().cpu().numpy(), model_config.get(name)):
                raise ValueError(f"disk flow state and config disagree on {name}")


def save_measurement_model(
    path, model, condition_preprocessor, target_transform, model_config, metadata=None
):
    _validate_disk_contract(model, target_transform, model_config)
    checkpoint = {
        "state_dict": model.state_dict(),
        "condition_preprocessor": condition_preprocessor.to_state(),
        "target_transform": target_transform.to_state(),
        "model_config": dict(model_config),
        "metadata": metadata or {},
    }
    torch.save(checkpoint, path)


def load_measurement_model(path, device="cpu"):
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    condition_preprocessor = TabularPreprocessor.from_state(checkpoint["condition_preprocessor"])
    target_transform = TargetStandardizer.from_state(checkpoint["target_transform"])
    model = build_flow(checkpoint["model_config"])
    model.load_state_dict(checkpoint["state_dict"])
    _validate_disk_contract(model, target_transform, checkpoint["model_config"])
    return MeasurementModelBundle(
        model,
        condition_preprocessor,
        target_transform,
        metadata=checkpoint.get("metadata", {}),
        device=device,
    )
