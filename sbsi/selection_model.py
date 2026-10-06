"""Differentiable Bernoulli selection model for the SBS project."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn

from .nn_utils import activation_class as _activation


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
    def fit(
        cls,
        frame: pd.DataFrame,
        feature_names: Sequence[str],
        add_missing_indicators=True,
        log_features: Sequence[str] = (),
    ):
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
        return cls(
            list(feature_names), fill_values, means, scales, add_missing_indicators, tuple(log_features)
        )

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
            raw = self.preprocessor.raw_tensor_from_frame(frame.iloc[start : start + batch_size], self.device)
            probs.append(torch.sigmoid(self.logits_from_raw_tensor(raw)).cpu().numpy())
        return np.concatenate(probs) if probs else np.array([], dtype=np.float32)


def _same_preprocessor(left: TabularPreprocessor, right: TabularPreprocessor) -> bool:
    """Return whether two ensemble members accept exactly the same raw table."""

    return bool(
        left.feature_names == right.feature_names
        and left.add_missing_indicators == right.add_missing_indicators
        and left.log_features == right.log_features
        and np.array_equal(left.fill_values, right.fill_values)
        and np.array_equal(left.means, right.means)
        and np.array_equal(left.scales, right.scales)
    )


class SelectionModelEnsembleBundle:
    """Equal-weight probability ensemble with one shared feature contract."""

    def __init__(self, members: Sequence[SelectionModelBundle]):
        self.members = tuple(members)
        if not self.members:
            raise ValueError("selection-model ensemble must contain at least one member")
        reference = self.members[0].preprocessor
        if any(not _same_preprocessor(reference, member.preprocessor) for member in self.members[1:]):
            raise ValueError("selection-model ensemble members have different preprocessors")
        self.preprocessor = reference
        self.metadata = {
            "aggregation": "arithmetic_mean_probability",
            "n_members": len(self.members),
            "members": [dict(member.metadata) for member in self.members],
        }

    def predict_proba(self, frame, batch_size=65536):
        probabilities = [member.predict_proba(frame, batch_size=batch_size) for member in self.members]
        return np.mean(np.stack(probabilities, axis=0), axis=0)


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


def load_selection_model_ensemble(paths: Sequence[str], device="cpu"):
    """Load an equal-weight ensemble, rejecting incompatible input contracts."""

    return SelectionModelEnsembleBundle([load_selection_model(path, device=device) for path in paths])
