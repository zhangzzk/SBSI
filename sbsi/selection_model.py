"""The detection classifier and the input standardization it shares with the flow."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
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
    """Fill missing values, standardize, and optionally append missing-value flags."""

    feature_names: list[str]
    fill_values: np.ndarray
    means: np.ndarray
    scales: np.ndarray
    add_missing_indicators: bool = True

    def transform_frame(self, frame):
        raw = frame[self.feature_names].to_numpy(dtype=np.float32, copy=True)
        finite = np.isfinite(raw)
        filled = np.where(finite, raw, self.fill_values)
        scaled = (filled - self.means) / self.scales
        if self.add_missing_indicators:
            scaled = np.concatenate([scaled, (~finite).astype(np.float32)], axis=1)
        return scaled.astype(np.float32, copy=False)

    def transform_tensor(self, raw):
        fill = torch.as_tensor(self.fill_values, dtype=raw.dtype, device=raw.device)
        means = torch.as_tensor(self.means, dtype=raw.dtype, device=raw.device)
        scales = torch.as_tensor(self.scales, dtype=raw.dtype, device=raw.device)
        finite = torch.isfinite(raw)
        filled = torch.where(finite, raw, fill)
        scaled = (filled - means) / scales
        if self.add_missing_indicators:
            scaled = torch.cat([scaled, (~finite).to(raw.dtype)], dim=1)
        return scaled

    @classmethod
    def from_state(cls, state):
        if state.get("log_features"):
            raise ValueError("log-transformed input features are not supported")
        return cls(
            list(state["feature_names"]),
            np.asarray(state["fill_values"], dtype=np.float32),
            np.asarray(state["means"], dtype=np.float32),
            np.asarray(state["scales"], dtype=np.float32),
            bool(state.get("add_missing_indicators", True)),
        )


class SelectionModelBundle:
    def __init__(self, model, preprocessor, temperature=1.0, metadata=None, device="cpu"):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.model.eval()
        self.preprocessor = preprocessor
        self.temperature = float(temperature)
        self.metadata = metadata or {}

    @torch.no_grad()
    def predict_proba(self, frame, batch_size=65536):
        probs = []
        for start in range(0, len(frame), batch_size):
            batch = frame.iloc[start : start + batch_size][self.preprocessor.feature_names]
            raw = torch.as_tensor(batch.to_numpy(dtype=np.float32, copy=True), device=self.device)
            logits = self.model(self.preprocessor.transform_tensor(raw)) / self.temperature
            probs.append(torch.sigmoid(logits).cpu().numpy())
        return np.concatenate(probs) if probs else np.array([], dtype=np.float32)


def load_selection_model(path, device="cpu"):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
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
