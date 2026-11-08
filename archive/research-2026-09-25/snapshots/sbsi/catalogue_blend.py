"""Reusable external blend-response values for a finite scene prior.

The BlendEMU regression target is a response at zero shear, not a new latent
variable to sample for every likelihood evaluation.  This module evaluates the
response-pair view once, sums pair predictions onto each catalogue atom, and
persists the aligned vector.  At trial shear ``g`` the external-response model
then shifts the measured shape by

``R_blend(z) * (e(S_g z) - e(z))``.

Keeping the response fixed while the exact ellipticity displacement changes is
the finite-shear implementation of MATH.md equation (M.2).  It also avoids
re-running the emulator across every likelihood-profile node.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping, Optional

import numpy as np

from .coordinates import ellipticity_from_axis_ratio_angle
from .forward_catalogue import EmulatorPairingConfig
from .response import predict_blend_response
from .scene_prior import ScenePrior, ShearedScenePrior
from .shear_map import apply_shear_to_ellipticity


CATALOGUE_BLEND_CACHE_VERSION = 1


@dataclass(frozen=True)
class CatalogueBlendResponse:
    """One fixed ``R_blend`` value aligned to every scene-prior row."""

    values: np.ndarray
    metadata: Mapping
    report: Mapping

    def __post_init__(self):
        values = np.asarray(self.values, dtype=np.float64)
        if values.ndim != 1 or not len(values):
            raise ValueError("blend response must be a non-empty one-dimensional array")
        if not np.isfinite(values).all():
            raise ValueError("blend response contains missing or non-finite values")
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "metadata", dict(self.metadata))
        object.__setattr__(self, "report", dict(self.report))

    @classmethod
    def from_emulator(
        cls,
        prior: ScenePrior,
        emulator,
        *,
        config: Optional[EmulatorPairingConfig] = None,
        active_indices: Optional[np.ndarray] = None,
        metadata: Optional[Mapping] = None,
    ) -> "CatalogueBlendResponse":
        """Evaluate BlendEMU once on zero-shear pairs for positive-mass atoms.

        Rows with no training-supported neighbour receive zero response.  That
        is a physical isolated/no-eligible-pair branch, not a failed catalogue
        join: every requested pair prediction is required to be finite and is
        aligned explicitly by ``primary_row``.
        """

        pairing = config or EmulatorPairingConfig.from_emulator(
            emulator, task="regression"
        )
        if active_indices is None:
            active = np.flatnonzero(prior.weights > 0)
        else:
            active = np.asarray(active_indices, dtype=np.int64)
            if active.ndim != 1 or ((active < 0) | (active >= len(prior.galaxies))).any():
                raise ValueError("active blend-response indices are invalid")
            if len(active) and (np.diff(active) <= 0).any():
                raise ValueError(
                    "active blend-response indices must be strictly increasing"
                )
        if not len(active):
            raise ValueError("blend response has no active catalogue atoms")

        try:
            pairs = prior.shear(0.0, 0.0).response_pairs(
                config=pairing,
                primary_indices=active,
            )
        except ValueError as error:
            if str(error) not in {
                "no neighbours lie inside the response aperture",
                "no response pairs remain after emulator training cuts",
            }:
                raise
            pairs = None

        values = np.zeros(len(prior.galaxies), dtype=np.float64)
        n_pairs = 0
        if pairs is not None:
            active_mask = np.zeros(len(prior.galaxies), dtype=bool)
            active_mask[active] = True
            n_pairs = len(pairs)
            if n_pairs:
                predicted = predict_blend_response(emulator, pairs)
                rows = predicted.index.to_numpy(dtype=np.int64)
                if ((rows < 0) | (rows >= len(values))).any():
                    raise RuntimeError("BlendEMU returned an out-of-range primary row")
                if not active_mask[rows].all():
                    raise RuntimeError("BlendEMU returned a response for an inactive atom")
                values[rows] = predicted.to_numpy(dtype=np.float64)

        active_values = values[active]
        report = {
            "n_scene_rows": int(len(values)),
            "n_active_atoms": int(len(active)),
            "n_response_pairs": int(n_pairs),
            "n_nonzero_atoms": int(np.count_nonzero(active_values)),
            "active_mean": float(active_values.mean()),
            "prior_weighted_mean": float(np.sum(prior.weights * values)),
            "active_min": float(active_values.min()),
            "active_max": float(active_values.max()),
        }
        return cls(values, dict(metadata or {}), report)

    def shape_shift_at(
        self,
        prior: ScenePrior,
        g1: float,
        g2: float,
    ) -> np.ndarray:
        """Return the aligned shift without materializing a sheared scene."""

        if len(self.values) != len(prior.galaxies):
            raise ValueError("blend response and scene prior row counts do not match")
        e1, e2 = ellipticity_from_axis_ratio_angle(
            prior.galaxies["axis_ratio"].to_numpy(dtype=float),
            prior.galaxies["position_angle"].to_numpy(dtype=float),
        )
        # Use the analytic map directly instead of converting the sheared
        # axis-ratio/angle representation back to components.  Besides being
        # the declared scene map, this makes the external shift bit-exactly
        # zero at zero shear rather than leaving round-trip noise at ~1e-18.
        se1, se2 = apply_shear_to_ellipticity(e1, e2, float(g1), float(g2))
        return self.values[:, None] * np.column_stack((se1 - e1, se2 - e2))

    def shape_shift(self, sheared: ShearedScenePrior) -> np.ndarray:
        """Return the atom-aligned shift for an existing sheared scene."""

        return self.shape_shift_at(sheared.prior, sheared.g1, sheared.g2)

    def save(self, path: str | Path) -> None:
        """Persist the aligned response and its complete identity/report."""

        root = Path(path)
        root.mkdir(parents=True, exist_ok=True)
        np.save(root / "r_blend.npy", self.values)
        manifest = {
            "version": CATALOGUE_BLEND_CACHE_VERSION,
            "values": "r_blend.npy",
            "n_rows": int(len(self.values)),
            "metadata": self.metadata,
            "report": self.report,
        }
        (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> "CatalogueBlendResponse":
        """Load a persisted response vector without invoking BlendEMU."""

        root = Path(path)
        manifest = json.loads((root / "manifest.json").read_text())
        if manifest.get("version") != CATALOGUE_BLEND_CACHE_VERSION:
            raise ValueError(
                f"unsupported blend-response cache version {manifest.get('version')!r}"
            )
        values = np.asarray(np.load(root / manifest["values"]), dtype=np.float64)
        if values.shape != (int(manifest["n_rows"]),):
            raise ValueError("blend-response cache row count does not match its manifest")
        return cls(values, manifest.get("metadata") or {}, manifest.get("report") or {})


__all__ = ["CATALOGUE_BLEND_CACHE_VERSION", "CatalogueBlendResponse"]
