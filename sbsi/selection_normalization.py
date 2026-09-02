"""Reusable exact and approximate measured-selection normalizations."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping

import numpy as np
import torch


CACHE_VERSION = 1
EXACT_CACHE_VERSION = 1

_SHAPE_TARGET_PAIRS = (
    ("measured_e1_image", "measured_e2_image"),
    ("measured_ngmix_g1", "measured_ngmix_g2"),
    ("measured_galsim_g1", "measured_galsim_g2"),
)


def _key(g1: float, g2: float) -> tuple[float, float]:
    return (round(float(g1), 14), round(float(g2), 14))


def detected_selected_mass_shard(
    selection,
    flow_model,
    view,
    prior_weights: np.ndarray,
    *,
    active_indices: np.ndarray,
    random_offset_rows: int,
) -> float:
    """Reduce one active-atom shard of ``B_W`` using the global QMC stream."""

    weights = np.asarray(prior_weights, dtype=np.float64)
    active = np.asarray(active_indices, dtype=np.int64)
    if weights.shape != (len(view.flow),) or not np.isfinite(weights).all():
        raise ValueError("prior weights must be a finite atom-aligned vector")
    if (weights < 0).any():
        raise ValueError("prior weights must be non-negative")
    if active.ndim != 1 or not len(active):
        raise ValueError("normalization shard must contain active atom indices")
    if ((active < 0) | (active >= len(view.flow))).any() or (
        len(active) and (np.diff(active) <= 0).any()
    ):
        raise ValueError("active selection indices are invalid")
    if random_offset_rows < 0:
        raise ValueError("selection random offset must be non-negative")

    torch.manual_seed(selection.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(selection.seed)
    if random_offset_rows:
        # The production QMC sampler consumes one target-dimensional random
        # digital shift per active row.  Shard boundaries are aligned to the
        # same row chunks as a one-process scan, so advancing by the preceding
        # active rows reproduces that global stream.
        torch.rand(
            int(random_offset_rows),
            1,
            len(flow_model.target_transform.target_names),
            dtype=torch.float32,
            device=getattr(flow_model, "device", torch.device("cpu")),
        )

    shape_indices = None
    names = tuple(flow_model.target_transform.target_names)
    for first, second in _SHAPE_TARGET_PAIRS:
        if first in names and second in names:
            shape_indices = (names.index(first), names.index(second))
            break
    total = 0.0
    for start in range(0, len(active), selection.row_chunk):
        rows = active[start : start + selection.row_chunk]
        frame = view.flow.iloc[rows]
        samples = flow_model.sample(
            frame,
            n_samples=selection.n_samples,
            batch_size=selection.row_chunk,
            qmc=True,
        )
        shift = view.blend_shift[rows]
        if np.any(shift):
            if shape_indices is None:
                raise KeyError(f"no known two-component shape targets in {list(names)}")
            samples = np.asarray(samples).copy()
            samples[..., shape_indices[0]] += shift[:, None, 0]
            samples[..., shape_indices[1]] += shift[:, None, 1]
        passed = np.asarray(selection.output_cut(samples), dtype=bool)
        expected = (len(frame), selection.n_samples)
        if passed.shape != expected:
            raise ValueError(
                f"output cut returned shape {passed.shape}, expected {expected}"
            )
        probability = passed.mean(axis=1, dtype=np.float64)
        total += float(
            np.sum(
                weights[rows] * view.detection_probability[rows] * probability,
                dtype=np.float64,
            )
        )
    return float(total)


@dataclass(frozen=True)
class ExactPopulationNormalization:
    """Exact finite-catalogue masses at a finite set of shear points."""

    points: Mapping[tuple[float, float], float]
    finite_difference_step: float
    identity: Mapping
    source: Mapping

    def __post_init__(self):
        if not np.isfinite(self.finite_difference_step) or self.finite_difference_step <= 0:
            raise ValueError("selection normalization finite-difference step must be positive")
        validated = {}
        for point, mass in self.points.items():
            if len(point) != 2 or not np.isfinite(point).all():
                raise ValueError("selection normalization points must be finite shear pairs")
            if not np.isfinite(mass) or mass <= 0:
                raise ValueError("selection normalization masses must be finite and positive")
            key = _key(*point)
            if key in validated:
                raise ValueError(f"duplicate selection normalization point {key}")
            validated[key] = float(mass)
        if not validated:
            raise ValueError("selection normalization requires at least one point")
        object.__setattr__(self, "points", validated)

    def log_mass(self, g1: float, g2: float) -> float:
        point = _key(g1, g2)
        if point not in self.points:
            raise ValueError(
                "exact selection normalization cache has no value at shear "
                f"{point}; available points are {sorted(self.points)}"
            )
        return float(np.log(self.points[point]))

    @property
    def available_shears(self) -> tuple[tuple[float, float], ...]:
        return tuple(sorted(self.points))

    def save(self, path: str | Path) -> None:
        payload = {
            "version": EXACT_CACHE_VERSION,
            "method": "exact_distributed_detected_selected_mass",
            "finite_difference_step": float(self.finite_difference_step),
            "identity": dict(self.identity),
            "points": [
                {"g1": g1, "g2": g2, "detected_and_selected_mass": mass}
                for (g1, g2), mass in sorted(self.points.items())
            ],
            "source": dict(self.source),
        }
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(payload, indent=2) + "\n")

    @classmethod
    def from_payload(cls, payload: Mapping) -> "ExactPopulationNormalization":
        if payload.get("version") != EXACT_CACHE_VERSION:
            raise ValueError(
                f"unsupported exact selection normalization cache version "
                f"{payload.get('version')!r}"
            )
        if payload.get("method") != "exact_distributed_detected_selected_mass":
            raise ValueError("unknown exact selection normalization cache method")
        points = {
            _key(row["g1"], row["g2"]): float(row["detected_and_selected_mass"])
            for row in payload.get("points", ())
        }
        return cls(
            points=points,
            finite_difference_step=float(payload["finite_difference_step"]),
            identity=dict(payload["identity"]),
            source=dict(payload.get("source") or {}),
        )


@dataclass(frozen=True)
class QuadraticPopulationNormalization:
    """Validated local quadratic surrogate for ``log B_W(g)``."""

    center: np.ndarray
    log_mass_at_center: float
    gradient: np.ndarray
    hessian: np.ndarray
    trust_min: np.ndarray
    trust_max: np.ndarray
    finite_difference_step: float
    identity: Mapping
    validation: Mapping
    source: Mapping

    def __post_init__(self):
        for name, value, shape in (
            ("center", self.center, (2,)),
            ("gradient", self.gradient, (2,)),
            ("hessian", self.hessian, (2, 2)),
            ("trust_min", self.trust_min, (2,)),
            ("trust_max", self.trust_max, (2,)),
        ):
            array = np.asarray(value, dtype=np.float64)
            if array.shape != shape or not np.isfinite(array).all():
                raise ValueError(f"selection normalization {name} must be finite {shape}")
            object.__setattr__(self, name, array)
        if not np.isfinite(self.log_mass_at_center):
            raise ValueError("selection normalization log mass must be finite")
        if not np.isfinite(self.finite_difference_step) or self.finite_difference_step <= 0:
            raise ValueError("selection normalization finite-difference step must be positive")
        if not np.allclose(self.hessian, self.hessian.T, rtol=0, atol=1e-12):
            raise ValueError("selection normalization Hessian must be symmetric")
        if (self.trust_min > self.center).any() or (self.center > self.trust_max).any():
            raise ValueError("selection normalization trust box must contain its center")

    def log_mass(self, g1: float, g2: float) -> float:
        point = np.asarray((g1, g2), dtype=np.float64)
        if not np.isfinite(point).all():
            raise ValueError("selection normalization shear must be finite")
        if (point < self.trust_min).any() or (point > self.trust_max).any():
            raise ValueError(
                "selection normalization cache refuses extrapolation outside its "
                f"validated trust box {self.trust_min.tolist()}..{self.trust_max.tolist()}: "
                f"got {point.tolist()}"
            )
        offset = point - self.center
        return float(
            self.log_mass_at_center
            + self.gradient @ offset
            + 0.5 * offset @ self.hessian @ offset
        )

    @classmethod
    def load(cls, path: str | Path) -> "QuadraticPopulationNormalization":
        payload = json.loads(Path(path).read_text())
        if payload.get("version") != CACHE_VERSION:
            raise ValueError(
                f"unsupported selection normalization cache version {payload.get('version')!r}"
            )
        if payload.get("method") != "local_quadratic_log_detected_selected_mass":
            raise ValueError("unknown selection normalization cache method")
        return cls(
            center=payload["center"],
            log_mass_at_center=float(payload["log_mass_at_center"]),
            gradient=payload["gradient"],
            hessian=payload["hessian"],
            trust_min=payload["trust_box"]["minimum"],
            trust_max=payload["trust_box"]["maximum"],
            finite_difference_step=float(payload["finite_difference_step"]),
            identity=dict(payload["identity"]),
            validation=dict(payload["validation"]),
            source=dict(payload["source"]),
        )


def load_population_normalization(
    path: str | Path,
) -> ExactPopulationNormalization | QuadraticPopulationNormalization:
    """Load a normalization cache by its declared method."""

    payload = json.loads(Path(path).read_text())
    method = payload.get("method")
    if method == "exact_distributed_detected_selected_mass":
        return ExactPopulationNormalization.from_payload(payload)
    if method == "local_quadratic_log_detected_selected_mass":
        return QuadraticPopulationNormalization.load(path)
    raise ValueError(f"unknown selection normalization cache method {method!r}")


def log_mass_derivatives(
    normalization: ExactPopulationNormalization | QuadraticPopulationNormalization,
    center: np.ndarray | tuple[float, float] | list[float],
) -> tuple[float, np.ndarray, np.ndarray]:
    """Evaluate ``log B``, its score, and Hessian on the cache stencil.

    The inference estimator obtains these same derivatives from its nine-view
    finite-difference stencil.  Exposing the calculation here lets a hybrid
    observation carry update the common population-normalization term for all
    rows, including rows whose observation-specific numerator is not rerun.
    """

    point = np.asarray(center, dtype=np.float64)
    if point.shape != (2,) or not np.isfinite(point).all():
        raise ValueError("selection normalization center must be a finite two-vector")
    h = float(normalization.finite_difference_step)

    def value(d1: float, d2: float) -> float:
        return normalization.log_mass(float(point[0] + d1), float(point[1] + d2))

    zero = value(0.0, 0.0)
    plus1, minus1 = value(h, 0.0), value(-h, 0.0)
    plus2, minus2 = value(0.0, h), value(0.0, -h)
    gradient = np.array(
        [(plus1 - minus1) / (2.0 * h), (plus2 - minus2) / (2.0 * h)],
        dtype=np.float64,
    )
    hessian = np.array(
        [
            [(plus1 - 2.0 * zero + minus1) / h**2, 0.0],
            [0.0, (plus2 - 2.0 * zero + minus2) / h**2],
        ],
        dtype=np.float64,
    )
    mixed = (
        value(h, h)
        - value(h, -h)
        - value(-h, h)
        + value(-h, -h)
    ) / (4.0 * h**2)
    hessian[0, 1] = hessian[1, 0] = mixed
    return zero, gradient, hessian


__all__ = [
    "detected_selected_mass_shard",
    "ExactPopulationNormalization",
    "QuadraticPopulationNormalization",
    "load_population_normalization",
    "log_mass_derivatives",
]
