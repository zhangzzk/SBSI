"""Bayesian inference API for a trained SBSI measurement likelihood.

This module performs grid-based latent-shape inference and an empirical-Bayes
fixed-point shear estimate.  It conditions on the detected/selected population used
to train the flow.  A future scene-prior implementation can add the full selection
normalization derived in ``doc/INFERENCE.md`` without changing this public interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

from .catalogue import Catalogue, load_catalogue
from .domain import Domain
from .measurement_model import add_measurement_target_features, load_measurement_model
from .posterior_shape import PosteriorShapeEstimator, RadialShapePrior, make_e_grid
from .preprocessing import rescale


@dataclass(frozen=True)
class PosteriorResult:
    index: np.ndarray
    mean_shape: np.ndarray
    log_evidence: np.ndarray


@dataclass(frozen=True)
class ShearInferenceResult:
    shear: np.ndarray
    posterior: PosteriorResult
    iterations: int
    converged: bool


class BayesianInference:
    """Latent-shape and catalogue-shear inference for an e-blind mean flow."""

    def __init__(
        self,
        checkpoint: Union[str, Path],
        *,
        device: Optional[str] = None,
        grid_size: int = 41,
        grid_extent: float = 0.96,
        grid_radius: float = 0.95,
        domain: Optional[Domain] = None,
    ):
        self.checkpoint = Path(checkpoint)
        if not self.checkpoint.is_file():
            raise FileNotFoundError(self.checkpoint)
        self.bundle = load_measurement_model(str(self.checkpoint), device=device or "cpu")
        self.grid, self.cell_area = make_e_grid(grid_size, grid_extent, grid_radius)
        self.estimator = PosteriorShapeEstimator(self.bundle, self.grid, device=device)
        self.domain = domain or Domain.from_flow_metadata(self.bundle.metadata)

    @classmethod
    def load(
        cls,
        checkpoint: Union[str, Path],
        *,
        device: Optional[str] = None,
        **grid_options,
    ) -> "BayesianInference":
        """Load one explicit flow checkpoint as a measurement likelihood."""

        return cls(checkpoint, device=device, **grid_options)

    @staticmethod
    def fit_prior(
        catalogue: Catalogue,
        *,
        e1_column: str = "e1_input_rot0_p",
        e2_column: str = "e2_input_rot0_p",
        bins: int = 60,
        min_samples: int = 10_000,
    ) -> RadialShapePrior:
        frame = load_catalogue(catalogue)
        missing = sorted({e1_column, e2_column} - set(frame.columns))
        if missing:
            raise KeyError(f"prior frame lacks intrinsic-shape columns: {missing}")
        return RadialShapePrior(
            frame[e1_column].to_numpy(float),
            frame[e2_column].to_numpy(float),
            n_bins=bins,
            min_samples=min_samples,
        )

    def _prepare(self, frame: pd.DataFrame, rescale_inputs: bool) -> pd.DataFrame:
        prepared = frame.copy()
        missing = sorted({"r_input_p", "Re_input_p"} - set(prepared.columns))
        if missing:
            raise KeyError(f"inference frame lacks domain columns: {missing}")
        inside = self.domain.mask(
            prepared["r_input_p"].to_numpy(float),
            prepared["Re_input_p"].to_numpy(float),
        )
        if not inside.all():
            raise ValueError(
                f"{int((~inside).sum()):,} rows are outside the trained flow domain"
            )
        if rescale_inputs:
            prepared = rescale(prepared)
        prepared = add_measurement_target_features(prepared)
        required = set(self.bundle.condition_preprocessor.feature_names)
        required.update(self.bundle.target_transform.target_names)
        missing = sorted(required - set(prepared.columns))
        if missing:
            raise KeyError(f"inference frame lacks model inputs: {missing}")
        return prepared

    def _log_likelihood(self, prepared: pd.DataFrame, chunk: int):
        observed = prepared[self.bundle.target_transform.target_names].to_numpy(float)
        return self.estimator.log_likelihood(prepared, observed, chunk=chunk)

    def posterior(
        self,
        catalogue: Catalogue,
        prior: RadialShapePrior,
        *,
        log_prior: Optional[np.ndarray] = None,
        chunk: int = 1024,
        rescale_inputs: bool = True,
    ) -> PosteriorResult:
        prepared = self._prepare(load_catalogue(catalogue), rescale_inputs)
        likelihood = self._log_likelihood(prepared, chunk)
        if log_prior is None:
            log_prior = prior.log_prob(self.grid[:, 0], self.grid[:, 1])
        mean, evidence = self.estimator.posterior_mean(likelihood, log_prior)
        return PosteriorResult(
            index=prepared.index.to_numpy(copy=True),
            mean_shape=mean,
            log_evidence=evidence,
        )

    def infer_shear(
        self,
        catalogue: Catalogue,
        prior: RadialShapePrior,
        *,
        initial=(0.0, 0.0),
        tolerance: float = 2.0e-5,
        max_iterations: int = 50,
        damping: float = 1.0,
        chunk: int = 1024,
        rescale_inputs: bool = True,
    ) -> ShearInferenceResult:
        """Estimate catalogue shear by re-shearing the prior to a fixed point."""

        if not (0.0 < damping <= 1.0):
            raise ValueError("damping must lie in (0, 1]")
        if tolerance <= 0:
            raise ValueError("tolerance must be positive")
        if max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        prepared = self._prepare(load_catalogue(catalogue), rescale_inputs)
        likelihood = self._log_likelihood(prepared, chunk)
        shear = np.asarray(initial, dtype=float)
        if shear.shape != (2,) or not np.isfinite(shear).all():
            raise ValueError("initial shear must be two finite components")
        if np.hypot(*shear) >= 1.0:
            raise ValueError("initial reduced shear must have magnitude below one")
        converged = False
        mean = evidence = None
        iteration = 0
        for iteration in range(1, max_iterations + 1):
            log_prior = prior.sheared_log_prob(
                self.grid[:, 0], self.grid[:, 1], shear[0], shear[1]
            )
            mean, evidence = self.estimator.posterior_mean(likelihood, log_prior)
            update = mean.mean(axis=0)
            candidate = (1.0 - damping) * shear + damping * update
            if not np.isfinite(candidate).all() or np.hypot(*candidate) >= 1.0:
                raise RuntimeError("empirical-Bayes shear iteration diverged")
            if np.max(np.abs(candidate - shear)) <= tolerance:
                shear = candidate
                converged = True
                break
            shear = candidate
        final_log_prior = prior.sheared_log_prob(
            self.grid[:, 0], self.grid[:, 1], shear[0], shear[1]
        )
        mean, evidence = self.estimator.posterior_mean(likelihood, final_log_prior)
        posterior = PosteriorResult(
            index=prepared.index.to_numpy(copy=True),
            mean_shape=mean,
            log_evidence=evidence,
        )
        return ShearInferenceResult(
            shear=shear,
            posterior=posterior,
            iterations=iteration,
            converged=converged,
        )


__all__ = [
    "BayesianInference",
    "PosteriorResult",
    "RadialShapePrior",
    "ShearInferenceResult",
]
