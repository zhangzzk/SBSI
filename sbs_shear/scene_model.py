"""Set-conditioned models for scene-level SBSI catalogues."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn

from .measurement_model import ConditionalAffineFlow, TargetStandardizer
from .selection_model import TabularPreprocessor


SCENE_GROUP_COLUMNS = ["case", "shear_case", "input_index"]

DEFAULT_SCENE_PRIMARY_FEATURES = [
    "Re_input_p_scaled",
    "r_input_p_scaled",
    "sersic_n_input_p",
    "e_abs_p",
    "gamma_pframe_parallel_p",
    "gamma_pframe_cross_p",
    "redshift_input_p",
    "neighbor_count",
    "nearest_distance_scaled",
    "brightest_neighbor_flux_ratio",
    "log10_total_neighbor_flux_ratio",
]

DEFAULT_SCENE_NEIGHBOR_FEATURES = [
    "distance_scaled",
    "pair_pframe_cos2",
    "pair_pframe_sin2",
    "Re_input_s_scaled",
    "r_input_s_scaled",
    "flux_ratio",
    "sersic_n_input_s",
    "e_pframe_parallel_s",
    "e_pframe_cross_s",
    "gamma_pframe_parallel_s",
    "gamma_pframe_cross_s",
    "redshift_input_s",
]

RADIAL_SCENE_NEIGHBOR_FEATURES = [
    "distance_scaled",
    "Re_input_s_scaled",
    "r_input_s_scaled",
    "flux_ratio",
    "sersic_n_input_s",
    "e_abs_s",
    "redshift_input_s",
]

SCENE_GEOMETRY_NEIGHBOR_FEATURES = {
    "full": DEFAULT_SCENE_NEIGHBOR_FEATURES,
    "radial": RADIAL_SCENE_NEIGHBOR_FEATURES,
}

SCENE_SUMMARY_FEATURES = [
    "neighbor_count",
    "nearest_distance_scaled",
    "brightest_neighbor_flux_ratio",
    "log10_total_neighbor_flux_ratio",
]


@dataclass
class SetFeatureStandardizer:
    feature_names: list[str]
    fill_values: np.ndarray
    means: np.ndarray
    scales: np.ndarray

    @classmethod
    def fit(cls, neighbor_values: Sequence[np.ndarray], feature_names: Sequence[str]):
        values = [np.asarray(arr, dtype=np.float32) for arr in neighbor_values if len(arr)]
        if values:
            stacked = np.concatenate(values, axis=0)
        else:
            stacked = np.empty((0, len(feature_names)), dtype=np.float32)

        if stacked.shape[1] != len(feature_names):
            raise ValueError("neighbor array width does not match feature_names")

        finite = np.isfinite(stacked)
        fill_values = np.zeros(len(feature_names), dtype=np.float32)
        for j in range(len(feature_names)):
            good = finite[:, j] if len(stacked) else np.array([], dtype=bool)
            fill_values[j] = float(np.mean(stacked[good, j])) if np.any(good) else 0.0

        if len(stacked):
            filled = np.where(finite, stacked, fill_values)
            means = filled.mean(axis=0, dtype=np.float64).astype(np.float32)
            scales = filled.std(axis=0, ddof=1).astype(np.float32)
        else:
            means = np.zeros(len(feature_names), dtype=np.float32)
            scales = np.ones(len(feature_names), dtype=np.float32)
        scales[~np.isfinite(scales) | (scales < 1e-6)] = 1.0
        return cls(list(feature_names), fill_values, means, scales)

    @property
    def dim(self):
        return len(self.feature_names)

    def transform_array(self, values):
        values = np.asarray(values, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != self.dim:
            raise ValueError(f"expected neighbor array with shape (n, {self.dim})")
        finite = np.isfinite(values)
        filled = np.where(finite, values, self.fill_values)
        return ((filled - self.means) / self.scales).astype(np.float32, copy=False)

    def transform_padded(self, neighbor_values: Sequence[np.ndarray], max_neighbors: int):
        if max_neighbors < 1:
            raise ValueError("max_neighbors must be >= 1")
        count = len(neighbor_values)
        values = np.zeros((count, max_neighbors, self.dim), dtype=np.float32)
        mask = np.zeros((count, max_neighbors), dtype=np.float32)
        raw_counts = np.zeros(count, dtype=np.int64)
        clipped_counts = np.zeros(count, dtype=np.int64)
        for i, arr in enumerate(neighbor_values):
            arr = np.asarray(arr, dtype=np.float32)
            raw_counts[i] = len(arr)
            if len(arr) == 0:
                continue
            take = min(len(arr), max_neighbors)
            clipped_counts[i] = take
            values[i, :take, :] = self.transform_array(arr[:take])
            mask[i, :take] = 1.0
        return values, mask, raw_counts, clipped_counts

    def to_state(self):
        return {
            "feature_names": self.feature_names,
            "fill_values": self.fill_values,
            "means": self.means,
            "scales": self.scales,
        }

    @classmethod
    def from_state(cls, state):
        return cls(
            list(state["feature_names"]),
            np.asarray(state["fill_values"], dtype=np.float32),
            np.asarray(state["means"], dtype=np.float32),
            np.asarray(state["scales"], dtype=np.float32),
        )


def _activation(name):
    name = name.lower()
    if name == "silu":
        return nn.SiLU
    if name == "gelu":
        return nn.GELU
    if name == "tanh":
        return nn.Tanh
    raise ValueError(f"Unsupported activation {name!r}")


def _mlp(input_dim, output_dim, hidden_dim, n_layers, activation):
    if n_layers < 1:
        raise ValueError("n_layers must be >= 1")
    act = _activation(activation)
    layers = []
    dim = input_dim
    for _ in range(n_layers - 1):
        layers.append(nn.Linear(dim, hidden_dim))
        layers.append(act())
        dim = hidden_dim
    layers.append(nn.Linear(dim, output_dim))
    return nn.Sequential(*layers)


class DeepSetsConditioner(nn.Module):
    def __init__(
        self,
        primary_dim,
        neighbor_dim,
        context_dim=128,
        hidden_dim=128,
        neighbor_layers=2,
        context_layers=2,
        activation="silu",
        pooling="sum",
    ):
        super().__init__()
        if pooling not in {"sum", "mean"}:
            raise ValueError("pooling must be 'sum' or 'mean'")
        self.pooling = pooling
        self.neighbor_net = _mlp(neighbor_dim, hidden_dim, hidden_dim, neighbor_layers, activation)
        self.context_net = _mlp(primary_dim + hidden_dim, context_dim, hidden_dim, context_layers, activation)

    def forward(self, primary, neighbors, neighbor_mask):
        encoded = self.neighbor_net(neighbors)
        mask = neighbor_mask.unsqueeze(-1).to(encoded.dtype)
        pooled = (encoded * mask).sum(dim=1)
        if self.pooling == "mean":
            denom = mask.sum(dim=1).clamp_min(1.0)
            pooled = pooled / denom
        return self.context_net(torch.cat([primary, pooled], dim=-1))


class SetConditionedMeasurementFlow(nn.Module):
    def __init__(
        self,
        target_dim,
        primary_dim,
        neighbor_dim,
        context_dim=128,
        set_hidden_dim=128,
        set_neighbor_layers=2,
        set_context_layers=2,
        flow_hidden_dim=256,
        flow_layers=3,
        n_flows=8,
        scale_limit=3.0,
        activation="silu",
        pooling="sum",
    ):
        super().__init__()
        self.conditioner = DeepSetsConditioner(
            primary_dim=primary_dim,
            neighbor_dim=neighbor_dim,
            context_dim=context_dim,
            hidden_dim=set_hidden_dim,
            neighbor_layers=set_neighbor_layers,
            context_layers=set_context_layers,
            activation=activation,
            pooling=pooling,
        )
        self.flow = ConditionalAffineFlow(
            target_dim=target_dim,
            context_dim=context_dim,
            hidden_dim=flow_hidden_dim,
            n_layers=flow_layers,
            n_flows=n_flows,
            scale_limit=scale_limit,
            activation=activation,
        )

    def context(self, primary, neighbors, neighbor_mask):
        return self.conditioner(primary, neighbors, neighbor_mask)

    def log_prob(self, target, primary, neighbors, neighbor_mask):
        return self.flow.log_prob(target, self.context(primary, neighbors, neighbor_mask))

    def sample(self, primary, neighbors, neighbor_mask, n_samples=1):
        return self.flow.sample(self.context(primary, neighbors, neighbor_mask), n_samples=n_samples)


class SceneMeasurementModelBundle:
    def __init__(
        self,
        model,
        primary_preprocessor,
        neighbor_preprocessor,
        target_transform,
        max_neighbors,
        metadata=None,
        device="cpu",
    ):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.model.eval()
        self.primary_preprocessor = primary_preprocessor
        self.neighbor_preprocessor = neighbor_preprocessor
        self.target_transform = target_transform
        self.max_neighbors = int(max_neighbors)
        self.metadata = metadata or {}

    def _primary_from_frame(self, frame):
        arr = self.primary_preprocessor.transform_frame(frame)
        return torch.as_tensor(arr, dtype=torch.float32, device=self.device)

    def _neighbors_from_arrays(self, neighbor_values):
        values, mask, _, _ = self.neighbor_preprocessor.transform_padded(
            neighbor_values, max_neighbors=self.max_neighbors
        )
        return (
            torch.as_tensor(values, dtype=torch.float32, device=self.device),
            torch.as_tensor(mask, dtype=torch.float32, device=self.device),
        )

    def _targets_from_frame(self, frame):
        arr = self.target_transform.transform_frame(frame)
        return torch.as_tensor(arr, dtype=torch.float32, device=self.device)

    @torch.no_grad()
    def log_prob(self, primary_frame, neighbor_values, target_frame, batch_size=16384):
        values = []
        for start in range(0, len(primary_frame), batch_size):
            end = start + batch_size
            primary = self._primary_from_frame(primary_frame.iloc[start:end])
            neighbors, mask = self._neighbors_from_arrays(neighbor_values[start:end])
            target = self._targets_from_frame(target_frame.iloc[start:end])
            values.append(self.model.log_prob(target, primary, neighbors, mask).cpu().numpy())
        return np.concatenate(values) if values else np.array([], dtype=np.float32)

    @torch.no_grad()
    def sample(self, primary_frame, neighbor_values, n_samples=1, batch_size=16384):
        samples = []
        for start in range(0, len(primary_frame), batch_size):
            end = start + batch_size
            primary = self._primary_from_frame(primary_frame.iloc[start:end])
            neighbors, mask = self._neighbors_from_arrays(neighbor_values[start:end])
            draw = self.model.sample(primary, neighbors, mask, n_samples=n_samples).cpu().numpy()
            flat = draw.reshape(-1, self.target_transform.dim)
            raw = self.target_transform.inverse_transform_array(flat)
            samples.append(raw.reshape(len(primary), n_samples, self.target_transform.dim))
        if not samples:
            return np.empty((0, n_samples, self.target_transform.dim), dtype=np.float32)
        return np.concatenate(samples, axis=0)


def save_scene_measurement_model(
    path,
    model,
    primary_preprocessor,
    neighbor_preprocessor,
    target_transform,
    model_config,
    max_neighbors,
    metadata=None,
):
    checkpoint = {
        "state_dict": model.state_dict(),
        "primary_preprocessor": primary_preprocessor.to_state(),
        "neighbor_preprocessor": neighbor_preprocessor.to_state(),
        "target_transform": target_transform.to_state(),
        "model_config": dict(model_config),
        "max_neighbors": int(max_neighbors),
        "metadata": metadata or {},
    }
    torch.save(checkpoint, path)


def load_scene_measurement_model(path, device="cpu"):
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    primary_preprocessor = TabularPreprocessor.from_state(checkpoint["primary_preprocessor"])
    neighbor_preprocessor = SetFeatureStandardizer.from_state(checkpoint["neighbor_preprocessor"])
    target_transform = TargetStandardizer.from_state(checkpoint["target_transform"])
    model = SetConditionedMeasurementFlow(**dict(checkpoint["model_config"]))
    model.load_state_dict(checkpoint["state_dict"])
    return SceneMeasurementModelBundle(
        model=model,
        primary_preprocessor=primary_preprocessor,
        neighbor_preprocessor=neighbor_preprocessor,
        target_transform=target_transform,
        max_neighbors=int(checkpoint["max_neighbors"]),
        metadata=checkpoint.get("metadata", {}),
        device=device,
    )
