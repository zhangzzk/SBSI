"""Frozen proposal covariance fitted to independent particle pilot runs.

This is a proposal heuristic, not a posterior covariance estimator. Runs have
equal weight, and only their within-run covariance contributes. Freeze the
artifact before drawing production particles; production seeds must differ from
every pilot seed. The geometry changes neither the target nor its support.

The Gaussian increment is symmetric in unconstrained physical coordinates.
Wrapping the phase after adding it preserves symmetry on the circle, including
cross-coordinate correlations: opposite increments have the same density and
the wrapped density sums over every integer phase winding. Fixed multiscale
mixtures retain this property because their probabilities are state independent.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import numpy as np


PHYSICAL_COLUMNS = (1, 6, 7, 8, 9)
ORIGINAL_WIDTHS = np.array([0.08, 0.08, 0.2, 0.2, 0.2])
BROAD_WIDTHS = 4 * ORIGINAL_WIDTHS
PILOT_SCALE = 2.38**2 / len(PHYSICAL_COLUMNS)
EIGENVALUE_FLOOR = 1e-4


def phase_residual(phase: np.ndarray, centre: np.ndarray) -> np.ndarray:
    """Return phase residuals in radians on [-pi, pi)."""
    return (np.asarray(phase) - np.asarray(centre) + np.pi) % (2 * np.pi) - np.pi


class FrozenParticleGeometry:
    """Object-specific symmetric increments from a fixed pilot artifact."""

    def __init__(
        self, pilot_covariance: np.ndarray, phase_centres: np.ndarray, metadata: dict,
        *, scale_factors=(1.0,), scale_probabilities=(1.0,),
    ):
        covariance = np.array(pilot_covariance, dtype=np.float64, copy=True)
        centres = np.array(phase_centres, dtype=np.float64, copy=True)
        if (covariance.ndim != 3 or covariance.shape[1:] != (5, 5)
                or not len(covariance) or not np.isfinite(covariance).all()
                or not np.allclose(covariance, covariance.swapaxes(1, 2), atol=1e-12, rtol=1e-10)):
            raise ValueError("pilot covariance must contain finite symmetric 5x5 matrices")
        if centres.shape != (len(covariance),) or not np.isfinite(centres).all():
            raise ValueError("phase centres must match the object count")
        factors = np.asarray(scale_factors, dtype=np.float64)
        probabilities = np.asarray(scale_probabilities, dtype=np.float64)
        if (factors.ndim != 1 or not len(factors) or not np.isfinite(factors).all()
                or np.any(factors <= 0) or probabilities.shape != factors.shape
                or not np.isfinite(probabilities).all() or np.any(probabilities <= 0)
                or not np.isclose(probabilities.sum(), 1.0, atol=1e-12, rtol=0)):
            raise ValueError("scales must be positive with matching normalized positive probabilities")
        try:
            np.linalg.cholesky(covariance)
        except np.linalg.LinAlgError as exc:
            raise ValueError("pilot covariance must be positive definite") from exc
        self._metadata = deepcopy(metadata)
        observations = self._metadata.get("observations")
        seeds = self._metadata.get("pilot_seeds")
        if (not isinstance(observations, list) or len(observations) != len(covariance)
                or any(type(v) is not int or v < 0 for v in observations)
                or len(set(observations)) != len(observations)):
            raise ValueError("metadata requires unique ordered integer observations")
        if (not isinstance(seeds, list) or not seeds
                or any(type(v) is not int or v < 0 for v in seeds)
                or len(set(seeds)) != len(seeds)):
            raise ValueError("metadata requires distinct integer pilot seeds")
        self._pilot_covariance = covariance
        self._phase_centres = centres % (2 * np.pi)
        self._precision = np.linalg.inv(covariance)
        self._broad_precision = np.diag(1 / BROAD_WIDTHS**2)
        self._scale_factors = factors.copy()
        self._scale_probabilities = probabilities.copy() / probabilities.sum()
        self._cholesky_cache = {}
        for value in (self._pilot_covariance, self._phase_centres, self._precision,
                      self._broad_precision, self._scale_factors, self._scale_probabilities):
            value.flags.writeable = False

    @property
    def metadata(self) -> dict:
        """Return a copy so callers cannot alter the frozen provenance."""
        return deepcopy(self._metadata)

    def validate_production(self, seed: int, observations: list[int]) -> None:
        """Reject reuse of a pilot RNG seed or a changed observation ordering."""
        if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
            raise ValueError("production seed must be a nonnegative integer")
        if int(seed) in self._metadata["pilot_seeds"]:
            raise ValueError("production seed reuses an independent pilot seed")
        if list(observations) != self._metadata["observations"]:
            raise ValueError("production observations or their order differ from the pilot")

    def covariance(self, beta: float) -> np.ndarray:
        """Base covariance, excluding the optional mixture's squared scale."""
        if not np.isscalar(beta) or not np.isfinite(beta) or not 0 <= beta <= 1:
            raise ValueError("beta must be a finite scalar in [0, 1]")
        precision = float(beta) * self._precision + (1 - float(beta)) * self._broad_precision
        covariance = np.linalg.inv(precision)
        return (covariance + covariance.swapaxes(1, 2)) / 2

    def increments(self, owner: np.ndarray, beta: float, rng: np.random.Generator) -> np.ndarray:
        """Draw (N,5) increments; owner indexes the frozen observation ordering."""
        owner = np.asarray(owner)
        if (owner.ndim != 1 or owner.dtype.kind not in "iu" or np.any(owner < 0)
                or np.any(owner >= len(self._pilot_covariance))):
            raise ValueError("owner must be a vector of valid integer object indices")
        if not np.isscalar(beta) or not np.isfinite(beta) or not 0 <= beta <= 1:
            raise ValueError("beta must be a finite scalar in [0, 1]")
        key = float(beta)
        if key not in self._cholesky_cache:
            self._cholesky_cache[key] = np.linalg.cholesky(self.covariance(key))
        normal = rng.normal(size=(len(owner), 5))
        increments = np.einsum("nij,nj->ni", self._cholesky_cache[key][owner], normal)
        if len(self._scale_factors) > 1:
            factors = rng.choice(self._scale_factors, size=len(owner), p=self._scale_probabilities)
            increments *= factors[:, None]
        else:
            increments *= self._scale_factors[0]
        return increments

    def save(self, path: str | Path) -> None:
        """Create a new artifact; refuse to overwrite an already frozen one."""
        payload = {
            "schema": "sbsi-frozen-particle-geometry-v1",
            "physical_columns": list(PHYSICAL_COLUMNS),
            "original_widths": ORIGINAL_WIDTHS.tolist(),
            "broad_widths": BROAD_WIDTHS.tolist(),
            "pilot_covariance": self._pilot_covariance.tolist(),
            "phase_centres": self._phase_centres.tolist(),
            "scale_factors": self._scale_factors.tolist(),
            "scale_probabilities": self._scale_probabilities.tolist(),
            "metadata": self.metadata,
        }
        with Path(path).open("x") as handle:
            json.dump(payload, handle, indent=2, allow_nan=False)
            handle.write("\n")

    @classmethod
    def load(cls, path: str | Path) -> "FrozenParticleGeometry":
        payload = json.loads(Path(path).read_text())
        if (payload.get("schema") != "sbsi-frozen-particle-geometry-v1"
                or payload.get("physical_columns") != list(PHYSICAL_COLUMNS)
                or payload.get("original_widths") != ORIGINAL_WIDTHS.tolist()
                or payload.get("broad_widths") != BROAD_WIDTHS.tolist()):
            raise ValueError("incompatible frozen particle geometry schema or coordinates")
        return cls(payload["pilot_covariance"], payload["phase_centres"], payload["metadata"],
                   scale_factors=payload["scale_factors"],
                   scale_probabilities=payload["scale_probabilities"])


def fit_particle_geometry(
    runs: list[np.ndarray], metadata: dict, *, multiscale: bool = False
) -> FrozenParticleGeometry:
    """Fit equal-run within-run covariance, retaining no between-run spread.

    Each array has shape (objects, particles, 10). Phases are first unwrapped
    around a single fixed circular centre per object, then each run is centred
    separately for its ordinary sample covariance (ddof=1). The mean covariance
    is scaled by original proposal widths, eigenvalues are floored at 1e-4,
    and the restored covariance is multiplied by 2.38**2/5. No score, particle
    weight, normalizer, or between-run mean enters this proposal heuristic.
    """
    if not runs:
        raise ValueError("at least one pilot run is required")
    physical = []
    objects = None
    for run in runs:
        run = np.asarray(run)
        if (run.ndim != 3 or run.shape[2] != 10 or run.shape[1] < 2
                or not run.shape[0] or (objects is not None and run.shape[0] != objects)):
            raise ValueError("pilot arrays must have shape (same objects, >=2 particles, 10)")
        objects = run.shape[0]
        selected = np.array(run[..., PHYSICAL_COLUMNS], dtype=np.float64, copy=True)
        if not np.isfinite(selected).all():
            raise ValueError("pilot physical coordinates must be finite")
        physical.append(selected)
    circular = np.mean([np.exp(1j * run[..., -1]).mean(axis=1) for run in physical], axis=0)
    resultant = np.abs(circular)
    ambiguous = resultant <= 64 * np.finfo(float).eps
    centres = np.where(ambiguous, 0.0, np.angle(circular)) % (2 * np.pi)
    covariances = []
    for run in physical:
        run[..., -1] = phase_residual(run[..., -1], centres[:, None])
        residual = run - run.mean(axis=1, keepdims=True)
        covariances.append(np.einsum("oni,onj->oij", residual, residual) / (run.shape[1] - 1))
    pooled = np.mean(covariances, axis=0)
    relative = pooled / ORIGINAL_WIDTHS[None, :, None] / ORIGINAL_WIDTHS[None, None, :]
    eigenvalues, eigenvectors = np.linalg.eigh(relative)
    regularized = np.maximum(eigenvalues, EIGENVALUE_FLOOR)
    restored = np.einsum("oik,ok,ojk->oij", eigenvectors, regularized, eigenvectors)
    restored *= ORIGINAL_WIDTHS[None, :, None] * ORIGINAL_WIDTHS[None, None, :]
    details = deepcopy(metadata)
    details.update({
        "fit": "equal-run mean of within-run ddof=1 physical covariance; proposal heuristic only",
        "pilot_particle_counts": [run.shape[1] for run in physical],
        "physical_columns": list(PHYSICAL_COLUMNS),
        "phase_residual": "radians in [-pi,pi) about fixed equal-run circular centre",
        "phase_centres": centres.tolist(),
        "phase_resultants": resultant.tolist(),
        "phase_ambiguous_zero_centre": ambiguous.tolist(),
        "raw_covariance": pooled.tolist(),
        "relative_eigenvalues_before_floor": eigenvalues.tolist(),
        "relative_eigenvalue_floor": EIGENVALUE_FLOOR,
        "pilot_scale_after_floor": PILOT_SCALE,
        "temperature_rule": "inverse(beta*inverse(pilot_scaled)+(1-beta)*inverse(broad))",
        "scale_semantics": "fixed state-independent multipliers of standard deviation",
        "uses_scores_or_posterior_weights": False,
    })
    return FrozenParticleGeometry(
        PILOT_SCALE * restored, centres, details,
        scale_factors=(0.25, 1, 4) if multiscale else (1,),
        scale_probabilities=(0.2, 0.6, 0.2) if multiscale else (1,),
    )
