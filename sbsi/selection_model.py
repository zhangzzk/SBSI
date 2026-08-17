"""Differentiable Bernoulli selection model for the SBS project."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F

from .nn_utils import activation_class as _activation


DEFAULT_SELECTION_FEATURES = [
    "Re_input_p_scaled",
    "r_input_p_scaled",
    "sersic_n_input_p",
    "e_abs_p",
    "gamma_pframe_parallel_p",
    "gamma_pframe_cross_p",
    "neighbored",
    "distance_scaled_blend",
    "pair_pframe_cos2_blend",
    "pair_pframe_sin2_blend",
    "Re_input_s_scaled_blend",
    "r_input_s_scaled_blend",
    "flux_ratio_blend",
    "sersic_n_input_s_blend",
    "e_pframe_parallel_s_blend",
    "e_pframe_cross_s_blend",
    "gamma_pframe_parallel_s_blend",
    "gamma_pframe_cross_s_blend",
]


# Shear-free conditioning for the refined g=0 forward model (SBI_shear.md §2/§4).
# Trained on the g=0 realization only: at g=0 the rendered scene shape equals the
# intrinsic shape, so the applied-shear features (gamma_pframe_*) carry no signal
# and are dropped.  The selection response dP(s=1)/dgamma is recovered afterwards
# by autograd through the intrinsic-shape features composed with the S_gamma map,
# not from a shear input feature.
SHEARFREE_G0_SELECTION_FEATURES = [
    "Re_input_p_scaled",
    "r_input_p_scaled",
    "sersic_n_input_p",
    "e_abs_p",
    "neighbored",
    "distance_scaled_blend",
    "pair_pframe_cos2_blend",
    "pair_pframe_sin2_blend",
    "Re_input_s_scaled_blend",
    "r_input_s_scaled_blend",
    "flux_ratio_blend",
    "sersic_n_input_s_blend",
    "e_pframe_parallel_s_blend",
    "e_pframe_cross_s_blend",
]


SELECTION_FEATURE_SETS = {
    "v6_primary_frame": DEFAULT_SELECTION_FEATURES,
    "g0_shearfree": SHEARFREE_G0_SELECTION_FEATURES,
}


class FocalLossWithLogits(nn.Module):
    def __init__(self, gamma=2.0, alpha=None):
        super().__init__()
        self.gamma = float(gamma)
        self.alpha = None if alpha is None else float(alpha)

    def forward(self, logits, targets):
        targets = targets.float()
        logits = logits.view_as(targets)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        prob = torch.sigmoid(logits)
        prob_t = prob * targets + (1.0 - prob) * (1.0 - targets)
        loss = (1.0 - prob_t).clamp_min(1e-8).pow(self.gamma) * bce
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
            loss = alpha_t * loss
        return loss.mean()


class SelectionMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, n_layers=4, dropout=0.0, activation="silu"):
        super().__init__()
        if n_layers < 1:
            raise ValueError("n_layers must be >= 1")
        act = _activation(activation)
        layers = []
        dim = input_dim
        for _ in range(n_layers):
            layers.append(nn.Linear(dim, hidden_dim))
            layers.append(act())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            dim = hidden_dim
        layers.append(nn.Linear(dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)




@dataclass
class TabularPreprocessor:
    feature_names: list[str]
    fill_values: np.ndarray
    means: np.ndarray
    scales: np.ndarray
    add_missing_indicators: bool = True
    # Features to log-transform BEFORE fill/standardise. Empty tuple reproduces the historical
    # behaviour exactly, and `from_state` defaults to empty, so every existing checkpoint loads
    # unchanged. The transform is stored WITH the preprocessor so inference cannot forget to apply
    # it -- a train/inference mismatch here would be silent and would corrupt every response.
    log_features: tuple[str, ...] = ()

    def _log_cols(self):
        want = set(self.log_features)
        return np.array([n in want for n in self.feature_names], dtype=bool)

    @staticmethod
    def _safe_log_np(values, mask):
        if not mask.any():
            return values
        out = values.astype(np.float32, copy=True)
        sub = out[:, mask]
        # non-positive -> NaN so the existing missing-value machinery handles it, rather than
        # -inf silently poisoning the mean/scale fit
        sub = np.where(sub > 0.0, sub, np.nan)
        with np.errstate(invalid="ignore"):
            out[:, mask] = np.log(sub)
        return out

    @classmethod
    def fit(cls, frame: pd.DataFrame, feature_names: Sequence[str], add_missing_indicators=True,
            log_features: Sequence[str] = ()):
        missing = [name for name in feature_names if name not in frame.columns]
        if missing:
            raise KeyError(f"Missing feature columns: {missing}")
        unknown = [n for n in log_features if n not in list(feature_names)]
        if unknown:
            raise KeyError(f"log_features not in feature_names: {unknown}")

        values = frame[list(feature_names)].to_numpy(dtype=np.float32, copy=True)
        _mask = np.array([n in set(log_features) for n in feature_names], dtype=bool)
        values = cls._safe_log_np(values, _mask)
        finite = np.isfinite(values)
        fill_values = np.zeros(values.shape[1], dtype=np.float32)
        for j in range(values.shape[1]):
            good = finite[:, j]
            fill_values[j] = float(np.mean(values[good, j])) if np.any(good) else 0.0

        filled = np.where(finite, values, fill_values)
        means = filled.mean(axis=0, dtype=np.float64).astype(np.float32)
        scales = filled.std(axis=0, ddof=1).astype(np.float32)
        scales[~np.isfinite(scales) | (scales < 1e-6)] = 1.0
        return cls(list(feature_names), fill_values, means, scales, add_missing_indicators,
                   tuple(log_features))

    @property
    def output_dim(self):
        dim = len(self.feature_names)
        return dim * 2 if self.add_missing_indicators else dim

    @property
    def output_names(self):
        names = list(self.feature_names)
        if self.add_missing_indicators:
            names += [f"{name}__is_missing" for name in self.feature_names]
        return names

    def transform_frame(self, frame):
        raw = frame[self.feature_names].to_numpy(dtype=np.float32, copy=True)
        raw = self._safe_log_np(raw, self._log_cols())
        finite = np.isfinite(raw)
        filled = np.where(finite, raw, self.fill_values)
        scaled = (filled - self.means) / self.scales
        if self.add_missing_indicators:
            scaled = np.concatenate([scaled, (~finite).astype(np.float32)], axis=1)
        return scaled.astype(np.float32, copy=False)

    def transform_tensor(self, raw):
        mask = self._log_cols()
        if mask.any():
            # differentiable and monotone; non-positive -> NaN, same rule as the numpy path so the
            # gradient route and the frame route cannot disagree
            m = torch.as_tensor(mask, device=raw.device)
            safe = torch.where(raw > 0, raw, torch.full_like(raw, float("nan")))
            raw = torch.where(m, torch.log(safe), raw)
        fill = torch.as_tensor(self.fill_values, dtype=raw.dtype, device=raw.device)
        means = torch.as_tensor(self.means, dtype=raw.dtype, device=raw.device)
        scales = torch.as_tensor(self.scales, dtype=raw.dtype, device=raw.device)
        finite = torch.isfinite(raw)
        filled = torch.where(finite, raw, fill)
        scaled = (filled - means) / scales
        if self.add_missing_indicators:
            scaled = torch.cat([scaled, (~finite).to(raw.dtype)], dim=1)
        return scaled

    def raw_tensor_from_frame(self, frame, device, requires_grad=False):
        raw = frame[self.feature_names].to_numpy(dtype=np.float32, copy=True)
        tensor = torch.as_tensor(raw, dtype=torch.float32, device=device)
        tensor.requires_grad_(requires_grad)
        return tensor

    def to_state(self):
        return {
            "feature_names": self.feature_names,
            "fill_values": self.fill_values,
            "means": self.means,
            "scales": self.scales,
            "add_missing_indicators": self.add_missing_indicators,
            "log_features": list(self.log_features),
        }

    @classmethod
    def from_state(cls, state):
        return cls(
            list(state["feature_names"]),
            np.asarray(state["fill_values"], dtype=np.float32),
            np.asarray(state["means"], dtype=np.float32),
            np.asarray(state["scales"], dtype=np.float32),
            bool(state.get("add_missing_indicators", True)),
            tuple(state.get("log_features", ())),
        )


class SelectionModelBundle:
    def __init__(self, model, preprocessor, temperature=1.0, metadata=None, device="cpu"):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.model.eval()
        self.preprocessor = preprocessor
        self.temperature = float(temperature)
        self.metadata = metadata or {}

    def logits_from_raw_tensor(self, raw):
        return self.model(self.preprocessor.transform_tensor(raw)) / self.temperature

    @torch.no_grad()
    def predict_proba(self, frame, batch_size=65536):
        probs = []
        for start in range(0, len(frame), batch_size):
            raw = self.preprocessor.raw_tensor_from_frame(
                frame.iloc[start:start + batch_size], self.device
            )
            probs.append(torch.sigmoid(self.logits_from_raw_tensor(raw)).cpu().numpy())
        return np.concatenate(probs) if probs else np.array([], dtype=np.float32)

    def probability_and_gradient(self, frame, gradient_features: Optional[Iterable[str]] = None, batch_size=8192):
        if gradient_features is None:
            gradient_features = self.preprocessor.feature_names
        gradient_features = list(gradient_features)
        feature_to_idx = {name: i for i, name in enumerate(self.preprocessor.feature_names)}
        grad_idx = [feature_to_idx[name] for name in gradient_features]

        probs = []
        grads = []
        for start in range(0, len(frame), batch_size):
            raw = self.preprocessor.raw_tensor_from_frame(
                frame.iloc[start:start + batch_size], self.device, requires_grad=True
            )
            prob = torch.sigmoid(self.logits_from_raw_tensor(raw))
            grad = torch.autograd.grad(prob.sum(), raw, create_graph=False)[0]
            probs.append(prob.detach().cpu().numpy())
            grads.append(grad[:, grad_idx].detach().cpu().numpy())

        prob_out = np.concatenate(probs) if probs else np.array([], dtype=np.float32)
        grad_arr = np.concatenate(grads) if grads else np.empty((0, len(gradient_features)))
        grad_df = pd.DataFrame(grad_arr, columns=[f"dPsel_d{name}" for name in gradient_features])
        return prob_out, grad_df


def save_selection_model(path, model, preprocessor, model_config, temperature, metadata=None):
    checkpoint = {
        "state_dict": model.state_dict(),
        "preprocessor": preprocessor.to_state(),
        "model_config": dict(model_config),
        "temperature": float(temperature),
        "metadata": metadata or {},
    }
    torch.save(checkpoint, path)


def load_selection_model(path, device="cpu"):
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    preprocessor = TabularPreprocessor.from_state(checkpoint["preprocessor"])
    model = SelectionMLP(**dict(checkpoint["model_config"]))
    model.load_state_dict(checkpoint["state_dict"])
    return SelectionModelBundle(
        model,
        preprocessor,
        temperature=float(checkpoint.get("temperature", 1.0)),
        metadata=checkpoint.get("metadata", {}),
        device=device,
    )


@torch.no_grad()
def collect_logits_and_targets(model, loader, device):
    model.eval()
    logits = []
    targets = []
    for xb, yb in loader:
        logits.append(model(xb.to(device)).detach().cpu())
        targets.append(yb.detach().cpu())
    return torch.cat(logits), torch.cat(targets)


def fit_temperature(logits, targets, max_iter=100):
    logits = logits.detach().float()
    targets = targets.detach().float()
    log_temperature = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.1, max_iter=max_iter)

    def closure():
        optimizer.zero_grad()
        temperature = torch.exp(log_temperature).clamp_min(1e-4)
        loss = F.binary_cross_entropy_with_logits(logits / temperature, targets)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(torch.exp(log_temperature).detach().clamp(1e-4, 1e4).item())
