"""Finite catalogue-prior likelihood with flow, detection, and measured cuts.

For prior atoms ``z_j`` with masses ``pi_j`` this module evaluates

``A_i(g) = sum_j pi_j P_det(S_g z_j) p_flow(xhat_i | S_g z_j)``

and, for a catalogue conditioned on detection and a measured-output cut ``W``,

``p(xhat_i | detected, W, g) = A_i(g) / B_W(g)``,
``B_W(g) = sum_j pi_j P_det(S_g z_j) P_pass(S_g z_j)``.

For every retained observation ``W(xhat_i)=1``, so the cut cancels from the
per-object numerator.  It does not cancel from the population normalization,
where ``P_pass(z) = Integral W(xhat) p_flow(xhat | z) dxhat`` is estimated with
common-random-number flow draws.

The measurement flow is normalized conditional on detection and therefore
must not absorb the Bernoulli factor.  The same catalogue atoms can be reused
at every finite-difference point, giving the common-random-number coupling
needed by the Lagrangian score.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.special import logsumexp
import torch

from .catalogue_blend import CatalogueBlendResponse
from .lagrangian_score import curve_derivatives
from .scene_prior import ScenePrior
from .shear_map import apply_shear_to_ellipticity


MODEL_CACHE_VERSION = 4
SUPPORTED_MODEL_CACHE_VERSIONS = {2, 3, MODEL_CACHE_VERSION}
SELECTION_CACHE_VERSION = 3

_SHAPE_TARGET_PAIRS = (
    ("measured_e1_image", "measured_e2_image"),
    ("measured_ngmix_g1", "measured_ngmix_g2"),
    ("measured_galsim_g1", "measured_galsim_g2"),
)

# These are the only classifier inputs in the declared BlendEMU detection
# checkpoint.  They are spin-0 scene properties under the SBSI shape-only
# shear convention.  Keeping the allow-list explicit makes an accidental
# orientation/ellipticity input a hard error instead of silently reusing a
# zero-shear classifier value.
SHEAR_INVARIANT_DETECTION_FEATURES = frozenset(
    {
        "Re_input_p_scaled",
        "Re_input_s_scaled",
        "r_input_p_scaled",
        "r_input_s_scaled",
        "sersic_n_input_p",
        "sersic_n_input_s",
        "distance_scaled",
    }
)

# The current flow changes only the primary intrinsic shape.  Every other
# condition below is computed once at zero shear and reused by the numerical
# stencil.  Unknown feature sets take the conservative full-view path.
ZERO_SHEAR_REUSABLE_FLOW_FEATURES = frozenset(
    {
        "e1_input_p",
        "e2_input_p",
        "sersic_n_input_p",
        "r_input_p",
        "Re_input_p",
        "nbr_flux_near",
        "nbr_flux_far",
        "nbr_flux_max",
    }
)


def _shape_target_indices(names: Sequence[str]) -> tuple[int, int]:
    """Locate the measured two-component shape in a flow output vector."""

    names = tuple(names)
    for first, second in _SHAPE_TARGET_PAIRS:
        if first in names and second in names:
            return names.index(first), names.index(second)
    raise KeyError(f"no known two-component shape target pair in {list(names)}")


class CatalogueSelection:
    """Measured-output selection probabilities for catalogue-prior atoms.

    ``output_cut`` is normally :class:`sbsi.score_inference.OutputCut`.  The
    class only relies on its callable predicate, ``target_names``, and ``key``
    interface so the exact same cut object can be used by the older score path
    and this finite-catalogue likelihood.

    Re-seeding to the same state at every trial shear holds the flow latents
    fixed across the likelihood curve.  Rows are processed in fixed chunks and
    only their pass fractions are retained, avoiding an ``N_atom x N_draw``
    sample tensor in memory.
    """

    def __init__(
        self,
        output_cut,
        *,
        n_samples: int = 64,
        seed: int = 8101,
        row_chunk: int = 8192,
    ):
        if n_samples <= 0 or n_samples & (n_samples - 1):
            raise ValueError("selection n_samples must be a positive power of two")
        if row_chunk <= 0:
            raise ValueError("selection row_chunk must be positive")
        if output_cut is None or not callable(output_cut):
            raise TypeError("output_cut must be callable")
        self.output_cut = output_cut
        self.n_samples = int(n_samples)
        self.seed = int(seed)
        self.row_chunk = int(row_chunk)
        self._probabilities: dict[tuple[float, float], np.ndarray] = {}
        self.metadata: dict = {}

    @staticmethod
    def _key(g1: float, g2: float) -> tuple[float, float]:
        return (round(float(g1), 14), round(float(g2), 14))

    @property
    def available_shears(self) -> tuple[tuple[float, float], ...]:
        return tuple(sorted(self._probabilities))

    @staticmethod
    def _validated_probability(
        probability: np.ndarray,
        *,
        n_rows: Optional[int] = None,
    ) -> np.ndarray:
        """Return one validated atom-aligned pass-probability vector."""

        values = np.asarray(probability, dtype=np.float64)
        expected = None if n_rows is None else (int(n_rows),)
        if values.ndim != 1 or not len(values) or (
            expected is not None and values.shape != expected
        ):
            suffix = "a non-empty vector" if expected is None else str(expected)
            raise ValueError(
                "selection probability must be an atom-aligned array with shape "
                f"{suffix}, got {values.shape}"
            )
        if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
            raise ValueError("selection probabilities must be finite and in [0, 1]")
        return values

    def cached_probability(self, g1: float, g2: float) -> np.ndarray:
        """Return an already-computed array without rebuilding a model view."""

        key = self._key(g1, g2)
        if key not in self._probabilities:
            raise KeyError(f"no selection probability is cached at shear {key}")
        return self._validated_probability(self._probabilities[key])

    def probability(
        self,
        flow_model,
        view: "CatalogueModelView",
        *,
        active_indices: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Return ``P_pass`` for every atom in one sheared model view."""

        key = self._key(view.g1, view.g2)
        if key in self._probabilities:
            probability = self._validated_probability(
                self._probabilities[key], n_rows=len(view.flow)
            )
            return probability

        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        if active_indices is None:
            active = np.arange(len(view.flow), dtype=np.int64)
        else:
            active = np.asarray(active_indices, dtype=np.int64)
            if active.ndim != 1 or ((active < 0) | (active >= len(view.flow))).any():
                raise ValueError("active selection indices are invalid")
            if len(active) and (np.diff(active) <= 0).any():
                raise ValueError("active selection indices must be strictly increasing")
        probability = np.zeros(len(view.flow), dtype=np.float64)
        for start in range(0, len(active), self.row_chunk):
            rows = active[start : start + self.row_chunk]
            frame = view.flow.iloc[rows]
            samples = flow_model.sample(
                frame,
                n_samples=self.n_samples,
                batch_size=self.row_chunk,
                qmc=True,
            )
            shift = view.blend_shift[rows]
            if np.any(shift):
                first, second = _shape_target_indices(
                    flow_model.target_transform.target_names
                )
                samples = np.asarray(samples).copy()
                samples[..., first] += shift[:, None, 0]
                samples[..., second] += shift[:, None, 1]
            passed = np.asarray(self.output_cut(samples), dtype=bool)
            expected = (len(frame), self.n_samples)
            if passed.shape != expected:
                raise ValueError(
                    f"output cut returned shape {passed.shape}, expected {expected}"
                )
            probability[rows] = passed.mean(axis=1, dtype=np.float64)
        probability = self._validated_probability(probability, n_rows=len(view.flow))
        self._probabilities[key] = probability
        return probability

    def save(self, path: str | Path, *, metadata: Optional[Mapping] = None) -> None:
        """Persist computed probabilities with their cut and sampling identity."""

        if not self._probabilities:
            raise ValueError("no selection probabilities have been computed")
        root = Path(path)
        root.mkdir(parents=True, exist_ok=True)
        entries = []
        ordered = sorted(self._probabilities.items())
        n_rows = len(self._validated_probability(ordered[0][1]))
        for key, probability in ordered:
            probability = self._validated_probability(probability, n_rows=n_rows)
            token = sha256(
                np.asarray(key, dtype="<f8").tobytes()
            ).hexdigest()[:20]
            name = f"selection_{token}.npy"
            temporary = root / f".{name}.{os.getpid()}.tmp"
            with temporary.open("wb") as handle:
                np.save(handle, probability)
            temporary.replace(root / name)
            entries.append({"g1": key[0], "g2": key[1], "probability": name})
        saved_metadata = dict(self.metadata)
        saved_metadata.update(dict(metadata or {}))
        manifest = {
            "version": SELECTION_CACHE_VERSION,
            "cut_key": self.output_cut.key(),
            "target_names": list(self.output_cut.target_names),
            "n_samples": self.n_samples,
            "seed": self.seed,
            "row_chunk": self.row_chunk,
            "n_rows": int(n_rows),
            "entries": entries,
            "metadata": saved_metadata,
        }
        manifest_path = root / "manifest.json"
        temporary_manifest = root / f".manifest.json.{os.getpid()}.tmp"
        temporary_manifest.write_text(json.dumps(manifest, indent=2) + "\n")
        temporary_manifest.replace(manifest_path)

    @classmethod
    def load(cls, path: str | Path, *, output_cut) -> "CatalogueSelection":
        """Load a cache, refusing a different measured selection predicate."""

        root = Path(path)
        manifest = json.loads((root / "manifest.json").read_text())
        if manifest.get("version") != SELECTION_CACHE_VERSION:
            raise ValueError(
                f"unsupported selection cache version {manifest.get('version')!r}"
            )
        if manifest.get("cut_key") != output_cut.key() or tuple(
            manifest.get("target_names", ())
        ) != tuple(output_cut.target_names):
            raise ValueError("selection cache cut does not match the requested output cut")
        selection = cls(
            output_cut,
            n_samples=int(manifest["n_samples"]),
            seed=int(manifest["seed"]),
            row_chunk=int(manifest["row_chunk"]),
        )
        selection.metadata = dict(manifest.get("metadata") or {})
        manifest_entries = manifest.get("entries")
        if not manifest_entries:
            raise ValueError("selection cache manifest contains no probability arrays")
        expected_rows = manifest.get("n_rows")
        if expected_rows is not None:
            expected_rows = int(expected_rows)
            if expected_rows <= 0:
                raise ValueError("selection cache n_rows must be positive")
        else:
            first_probability = selection._validated_probability(
                np.load(root / manifest_entries[0]["probability"])
            )
            expected_rows = len(first_probability)
        for entry in manifest_entries:
            key = selection._key(entry["g1"], entry["g2"])
            probability = selection._validated_probability(
                np.load(root / entry["probability"]), n_rows=expected_rows
            )
            selection._probabilities[key] = probability
        return selection


@dataclass(frozen=True)
class CatalogueModelView:
    """Model-specific tables for one sheared view of the prior atoms."""

    g1: float
    g2: float
    flow: pd.DataFrame
    detection: pd.DataFrame
    detection_probability: np.ndarray
    blend_shift: Optional[np.ndarray] = None

    def __post_init__(self):
        n = len(self.flow)
        probability = np.asarray(self.detection_probability, dtype=np.float64)
        if len(self.detection) != n or probability.shape != (n,):
            raise ValueError("flow, detection, and probability rows are not aligned")
        if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
            raise ValueError("detection probabilities must be finite and in [0, 1]")
        if not self.flow.index.equals(self.detection.index):
            raise ValueError("flow and detection views must have identical primary-row indices")
        if self.blend_shift is None:
            blend_shift = np.zeros((n, 2), dtype=np.float64)
        else:
            blend_shift = np.asarray(self.blend_shift, dtype=np.float64)
        if blend_shift.shape != (n, 2) or not np.isfinite(blend_shift).all():
            raise ValueError("blend shift must be a finite array with shape (n_rows, 2)")
        object.__setattr__(self, "detection_probability", probability)
        object.__setattr__(self, "blend_shift", blend_shift)


def _feature_names(model, *, flow: bool) -> tuple[str, ...]:
    if flow and hasattr(model, "condition_preprocessor"):
        return tuple(model.condition_preprocessor.feature_names)
    if not flow and hasattr(model, "preprocessor"):
        return tuple(model.preprocessor.feature_names)
    if not flow and hasattr(model, "bst_cla") and hasattr(model.bst_cla, "feature_names"):
        return tuple(model.bst_cla.feature_names)
    return ()


def predict_detection_probability(detector, frame: pd.DataFrame) -> np.ndarray:
    """Call either an SBSI selection bundle or BlendEMU classifier adapter."""

    if hasattr(detector, "predict_proba"):
        probability = detector.predict_proba(frame)
    elif hasattr(detector, "predict_on_pairs"):
        predicted = detector.predict_on_pairs(
            frame, task="detection", rescaled=True, warn_extrapolation=False
        )
        if "detection_prob" not in predicted:
            raise KeyError("detector output lacks detection_prob")
        probability = predicted["detection_prob"].to_numpy(dtype=float)
    elif callable(detector):
        probability = detector(frame)
    else:
        raise TypeError(
            "detector must expose predict_proba, BlendEMU predict_on_pairs, or be callable"
        )
    probability = np.asarray(probability, dtype=np.float64).reshape(-1)
    if probability.shape != (len(frame),):
        raise ValueError("detector returned the wrong number of probabilities")
    return probability


class CatalogueModelCache:
    """Lazy in-memory cache of flow and classifier views at trial shears.

    The expensive catalogue neighbour search lives in :class:`ScenePrior` and
    is never repeated.  This layer only transforms the stored graph, rebuilds
    the three model-specific feature conventions, and evaluates detection.
    """

    def __init__(
        self,
        prior: ScenePrior,
        *,
        detector,
        conditions: Mapping[str, float],
        detection_radius_arcsec: float,
        flow_neighbour_radius_arcsec: float,
        crowding_radii_arcsec: Sequence[float],
        detection_neighbour_selection: str = "nearest",
        detection_impact_exponent: float = 2.0,
        blend_response: Optional[CatalogueBlendResponse] = None,
        flow_features: Optional[Sequence[str]] = None,
        detection_features: Optional[Sequence[str]] = None,
    ):
        self.prior = prior
        self.detector = detector
        self.conditions = {name: float(value) for name, value in conditions.items()}
        self.detection_radius_arcsec = float(detection_radius_arcsec)
        if detection_neighbour_selection not in {"nearest", "impact"}:
            raise ValueError(
                "detection_neighbour_selection must be 'nearest' or 'impact'"
            )
        if not np.isfinite(detection_impact_exponent) or detection_impact_exponent < 0:
            raise ValueError("detection_impact_exponent must be finite and non-negative")
        self.detection_neighbour_selection = detection_neighbour_selection
        self.detection_impact_exponent = float(detection_impact_exponent)
        self.flow_neighbour_radius_arcsec = float(flow_neighbour_radius_arcsec)
        self.crowding_radii_arcsec = tuple(float(v) for v in crowding_radii_arcsec)
        if blend_response is not None and len(blend_response.values) != len(prior.galaxies):
            raise ValueError("blend response and scene prior row counts do not match")
        self.blend_response = blend_response
        self.flow_features = (
            None if flow_features is None else tuple(dict.fromkeys(flow_features))
        )
        inferred_detection = _feature_names(detector, flow=False)
        self.detection_features = tuple(
            dict.fromkeys(detection_features or inferred_detection)
        )
        self._views: dict[tuple[float, float], CatalogueModelView] = {}
        self.metadata: dict = {}

    @staticmethod
    def _key(g1: float, g2: float) -> tuple[float, float]:
        return (round(float(g1), 14), round(float(g2), 14))

    def get(self, g1: float, g2: float) -> CatalogueModelView:
        key = self._key(g1, g2)
        if key not in self._views:
            if self.detector is None:
                raise KeyError(
                    f"model cache has no view at shear {key} and no detector is loaded "
                    "to build it"
                )
            reusable = (
                key != (0.0, 0.0)
                and self.flow_features is not None
                and set(self.flow_features) <= ZERO_SHEAR_REUSABLE_FLOW_FEATURES
                and {"e1_input_p", "e2_input_p"} <= set(self.flow_features)
                and set(self.detection_features)
                <= SHEAR_INVARIANT_DETECTION_FEATURES
            )
            if reusable:
                zero = self.get(0.0, 0.0)
                # Shallow-copy the frame so invariant spin-0 blocks remain
                # shared; assigning the two shape columns materializes only
                # the shear-dependent data.
                flow = zero.flow.copy(deep=False)
                e1, e2 = apply_shear_to_ellipticity(
                    zero.flow["e1_input_p"].to_numpy(dtype=float),
                    zero.flow["e2_input_p"].to_numpy(dtype=float),
                    key[0],
                    key[1],
                )
                flow["e1_input_p"] = e1
                flow["e2_input_p"] = e2
                detection = zero.detection
                probability = zero.detection_probability
                blend_shift = (
                    np.zeros((len(flow), 2), dtype=np.float64)
                    if self.blend_response is None
                    else self.blend_response.shape_shift_at(
                        self.prior, key[0], key[1]
                    )
                )
            else:
                sheared = self.prior.shear(*key)
                flow = sheared.flow_view(
                    conditions=self.conditions,
                    neighbour_radius_arcsec=self.flow_neighbour_radius_arcsec,
                    crowding_radii_arcsec=self.crowding_radii_arcsec,
                )
                detection = sheared.detection_view(
                    conditions=self.conditions,
                    radius_arcsec=self.detection_radius_arcsec,
                    neighbour_selection=self.detection_neighbour_selection,
                    impact_exponent=self.detection_impact_exponent,
                )
                probability = predict_detection_probability(self.detector, detection)
                if self.flow_features is not None:
                    missing = sorted(set(self.flow_features) - set(flow))
                    if missing:
                        raise KeyError(f"scene cache lacks flow features: {missing}")
                    flow = flow.loc[:, self.flow_features].copy()
                if self.detection_features:
                    missing = sorted(set(self.detection_features) - set(detection))
                    if missing:
                        raise KeyError(
                            f"scene cache lacks detection features: {missing}"
                        )
                    detection = detection.loc[:, self.detection_features].copy()
                blend_shift = (
                    np.zeros((len(flow), 2), dtype=np.float64)
                    if self.blend_response is None
                    else self.blend_response.shape_shift(sheared)
                )
            self._views[key] = CatalogueModelView(
                key[0], key[1], flow, detection, probability, blend_shift
            )
        return self._views[key]

    def validate_model_features(self, flow_model) -> None:
        """Fail early if a checkpoint asks for a feature this cache cannot build."""

        view = self.get(0.0, 0.0)
        missing_flow = sorted(set(_feature_names(flow_model, flow=True)) - set(view.flow))
        missing_detection = sorted(
            set(_feature_names(self.detector, flow=False)) - set(view.detection)
        )
        messages = []
        if missing_flow:
            messages.append(f"flow features: {missing_flow}")
        if missing_detection:
            messages.append(f"detection features: {missing_detection}")
        if messages:
            raise KeyError("scene cache cannot supply " + "; ".join(messages))

    def validate_detection_shear_invariance(self) -> tuple[str, ...]:
        """Verify that checkpoint-declared detection inputs are spin-0.

        The shape-only closure reuses one classifier evaluation at every
        stencil point.  This is valid only for the explicit metadata feature
        set above.  An empty/unknown feature list is rejected for production
        rather than guessed from column names.
        """

        if not self.detection_features:
            raise ValueError(
                "detection checkpoint does not expose an auditable feature list"
            )
        varying = sorted(
            set(self.detection_features) - SHEAR_INVARIANT_DETECTION_FEATURES
        )
        if varying:
            raise ValueError(
                "detection checkpoint has shape-sensitive or unknown inputs: "
                f"{varying}"
            )
        return self.detection_features

    def with_prior(self, prior: ScenePrior) -> "CatalogueModelCache":
        """Return a row-aligned cache with different empirical prior masses."""

        if len(prior.galaxies) != len(self.prior.galaxies):
            raise ValueError("reweighted prior must preserve every scene row")
        cache = CatalogueModelCache(
            prior,
            detector=self.detector,
            conditions=self.conditions,
            detection_radius_arcsec=self.detection_radius_arcsec,
            flow_neighbour_radius_arcsec=self.flow_neighbour_radius_arcsec,
            crowding_radii_arcsec=self.crowding_radii_arcsec,
            detection_neighbour_selection=self.detection_neighbour_selection,
            detection_impact_exponent=self.detection_impact_exponent,
            blend_response=self.blend_response,
            flow_features=self.flow_features,
            detection_features=self.detection_features,
        )
        cache._views = self._views
        cache.metadata = dict(self.metadata)
        return cache

    def precompute(self, shears: Sequence[Sequence[float]]) -> None:
        for g1, g2 in shears:
            self.get(float(g1), float(g2))

    @property
    def available_shears(self) -> tuple[tuple[float, float], ...]:
        """Shear views currently resident in memory."""

        return tuple(sorted(self._views))

    def attach_detector(self, detector) -> None:
        """Attach a classifier so a cache loaded from disk can grow lazily."""

        if detector is None:
            raise ValueError("detector must not be None")
        inferred = _feature_names(detector, flow=False)
        if self.detection_features and inferred and inferred != self.detection_features:
            raise ValueError(
                "attached detector feature metadata does not match the cached feature set"
            )
        if not self.detection_features:
            self.detection_features = inferred
        self.detector = detector

    def discard_views(
        self, shears: Sequence[Sequence[float]], *, missing_ok: bool = True
    ) -> None:
        """Release selected in-memory views after a bounded-memory scan."""

        for g1, g2 in shears:
            key = self._key(g1, g2)
            if key not in self._views and not missing_ok:
                raise KeyError(f"model cache has no view at shear {key}")
            self._views.pop(key, None)

    def save(self, path: str | Path, *, metadata: Optional[Mapping] = None) -> None:
        """Persist all currently cached model views and their manifest."""

        if not self._views:
            raise ValueError("no model views have been computed")
        root = Path(path)
        root.mkdir(parents=True, exist_ok=True)
        compact = (
            self.blend_response is None
            and self.flow_features is not None
            and set(self.flow_features) <= ZERO_SHEAR_REUSABLE_FLOW_FEATURES
            and {"e1_input_p", "e2_input_p"} <= set(self.flow_features)
            and set(self.detection_features)
            <= SHEAR_INVARIANT_DETECTION_FEATURES
        )
        entries = []
        if compact:
            zero = self.get(0.0, 0.0)
            flow_name = "flow_zero.parquet"
            detection_name = "detection_zero.parquet"
            zero.flow.reset_index().to_parquet(root / flow_name, index=False)
            detection = zero.detection.copy()
            detection["detection_probability"] = zero.detection_probability
            detection.reset_index().to_parquet(root / detection_name, index=False)
            entries.append(
                {
                    "g1": 0.0,
                    "g2": 0.0,
                    "flow": flow_name,
                    "detection": detection_name,
                }
            )
        else:
            for index, (key, view) in enumerate(sorted(self._views.items())):
                token = f"s{index:03d}"
                flow_name = f"flow_{token}.parquet"
                detection_name = f"detection_{token}.parquet"
                view.flow.reset_index().to_parquet(root / flow_name, index=False)
                detection = view.detection.copy()
                detection["detection_probability"] = view.detection_probability
                detection.reset_index().to_parquet(root / detection_name, index=False)
                entries.append(
                    {
                        "g1": key[0],
                        "g2": key[1],
                        "flow": flow_name,
                        "detection": detection_name,
                    }
                )
        saved_metadata = dict(self.metadata)
        saved_metadata.update(dict(metadata or {}))
        manifest = {
            "version": MODEL_CACHE_VERSION,
            "conditions": self.conditions,
            "detection_radius_arcsec": self.detection_radius_arcsec,
            "detection_neighbour_selection": self.detection_neighbour_selection,
            "detection_impact_exponent": self.detection_impact_exponent,
            "flow_neighbour_radius_arcsec": self.flow_neighbour_radius_arcsec,
            "crowding_radii_arcsec": list(self.crowding_radii_arcsec),
            "flow_features": (
                None if self.flow_features is None else list(self.flow_features)
            ),
            "detection_features": list(self.detection_features),
            "storage": "shape_only_zero_base" if compact else "full_views",
            "views": entries,
            "metadata": saved_metadata,
        }
        (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        prior: ScenePrior,
        blend_response: Optional[CatalogueBlendResponse] = None,
    ) -> "CatalogueModelCache":
        """Load persisted stencil views without loading the detection model."""

        root = Path(path)
        manifest = json.loads((root / "manifest.json").read_text())
        if manifest.get("version") not in SUPPORTED_MODEL_CACHE_VERSIONS:
            raise ValueError(
                f"unsupported model-cache version {manifest.get('version')!r}; "
                f"expected one of {sorted(SUPPORTED_MODEL_CACHE_VERSIONS)}"
            )
        cache = cls(
            prior,
            detector=None,
            conditions=manifest["conditions"],
            detection_radius_arcsec=manifest["detection_radius_arcsec"],
            flow_neighbour_radius_arcsec=manifest["flow_neighbour_radius_arcsec"],
            crowding_radii_arcsec=manifest["crowding_radii_arcsec"],
            detection_neighbour_selection=manifest.get(
                "detection_neighbour_selection", "nearest"
            ),
            detection_impact_exponent=manifest.get("detection_impact_exponent", 2.0),
            blend_response=blend_response,
            flow_features=manifest.get("flow_features"),
            detection_features=manifest.get("detection_features"),
        )
        cache.metadata = dict(manifest.get("metadata") or {})
        expected_index = pd.Index(
            np.arange(len(prior.galaxies), dtype=np.int64), name="primary_row"
        )
        for entry in manifest["views"]:
            flow = pd.read_parquet(root / entry["flow"]).set_index("primary_row")
            detection = pd.read_parquet(root / entry["detection"]).set_index("primary_row")
            if len(flow) != len(prior.galaxies) or not flow.index.equals(expected_index):
                raise ValueError(
                    "model-cache flow rows are not aligned with scene primary rows"
                )
            if len(detection) != len(prior.galaxies) or not detection.index.equals(
                expected_index
            ):
                raise ValueError(
                    "model-cache detection rows are not aligned with scene primary rows"
                )
            probability = detection.pop("detection_probability").to_numpy(dtype=float)
            key = cache._key(entry["g1"], entry["g2"])
            blend_shift = (
                np.zeros((len(flow), 2), dtype=np.float64)
                if blend_response is None
                else blend_response.shape_shift_at(prior, *key)
            )
            cache._views[key] = CatalogueModelView(
                key[0], key[1], flow, detection, probability, blend_shift
            )
        return cache


@dataclass(frozen=True)
class CatalogueScore:
    """Directional score and observed information for measured objects."""

    log_likelihood: np.ndarray
    score: np.ndarray
    information: np.ndarray


@dataclass(frozen=True)
class CatalogueTensorView:
    """Accelerator-resident representation of one catalogue model view."""

    context: torch.Tensor
    log_detected_mass: torch.Tensor
    blend_shift_standardized: torch.Tensor


class CatalogueLikelihood:
    """Exact or importance-sampled finite catalogue likelihood.

    When ``selection`` is supplied, every observation must pass its measured
    output cut and the catalogue normalization includes the corresponding
    per-atom ``P_pass``.  The numerator is unchanged because the deterministic
    cut is one for every retained observation.
    """

    def __init__(
        self,
        flow_model,
        cache: CatalogueModelCache,
        *,
        selection: Optional[CatalogueSelection] = None,
    ):
        self.flow_model = flow_model
        self.cache = cache
        self.selection = selection
        target_transform = getattr(flow_model, "target_transform", None)
        if target_transform is None or not hasattr(target_transform, "target_names"):
            raise TypeError("flow model must be a MeasurementModelBundle-like object")
        self.target_names = tuple(target_transform.target_names)
        self.shape_target_indices: Optional[tuple[int, int]] = None
        if cache.blend_response is not None:
            self.shape_target_indices = _shape_target_indices(self.target_names)
        if selection is not None and tuple(selection.output_cut.target_names) != self.target_names:
            raise ValueError(
                "selection target names do not match the measurement flow outputs"
            )
        self.cache.validate_model_features(flow_model)
        self._tensor_views: dict[tuple[float, float], CatalogueTensorView] = {}
        self._raw_zero_context: Optional[torch.Tensor] = None

    @property
    def tensor_native_available(self) -> bool:
        """Whether the flow exposes the tensor-native catalogue interface."""

        required = ("context_tensor", "target_tensor", "log_prob_tensor", "device")
        return all(hasattr(self.flow_model, name) for name in required)

    @property
    def tensor_shape_only_available(self) -> bool:
        """Whether nonzero views can be formed from the zero tensor base."""

        if not self.tensor_native_available or self.cache.blend_response is not None:
            return False
        features = tuple(self.flow_model.condition_preprocessor.feature_names)
        preprocessor = self.flow_model.condition_preprocessor
        return bool(
            "e1_input_p" in features
            and "e2_input_p" in features
            and not ({"e1_input_p", "e2_input_p"} & set(preprocessor.log_features))
            and self.cache.flow_features is not None
            and set(self.cache.flow_features) <= ZERO_SHEAR_REUSABLE_FLOW_FEATURES
            and set(self.cache.detection_features) <= SHEAR_INVARIANT_DETECTION_FEATURES
        )

    def _shear_standardized_context(
        self,
        context: torch.Tensor,
        flat_indices: torch.Tensor,
        g1: float,
        g2: float,
    ) -> torch.Tensor:
        """Replace only standardized intrinsic-shape columns on the device."""

        if g1 == 0.0 and g2 == 0.0:
            return context
        preprocessor = self.flow_model.condition_preprocessor
        features = tuple(preprocessor.feature_names)
        first = features.index("e1_input_p")
        second = features.index("e2_input_p")
        raw = self.raw_zero_context_tensor()
        e1 = raw[:, first].index_select(0, flat_indices)
        e2 = raw[:, second].index_select(0, flat_indices)
        g1_tensor = torch.as_tensor(g1, dtype=e1.dtype, device=e1.device)
        g2_tensor = torch.as_tensor(g2, dtype=e1.dtype, device=e1.device)
        denominator_real = 1.0 + g1_tensor * e1 + g2_tensor * e2
        denominator_imag = g1_tensor * e2 - g2_tensor * e1
        numerator_real = e1 + g1_tensor
        numerator_imag = e2 + g2_tensor
        denominator = denominator_real.square() + denominator_imag.square()
        sheared_e1 = (
            numerator_real * denominator_real
            + numerator_imag * denominator_imag
        ) / denominator
        sheared_e2 = (
            numerator_imag * denominator_real
            - numerator_real * denominator_imag
        ) / denominator
        means = torch.as_tensor(
            preprocessor.means, dtype=context.dtype, device=context.device
        )
        scales = torch.as_tensor(
            preprocessor.scales, dtype=context.dtype, device=context.device
        )
        context = context.clone()
        context[:, first] = (sheared_e1 - means[first]) / scales[first]
        context[:, second] = (sheared_e2 - means[second]) / scales[second]
        return context

    def _tensor_view(self, g1: float, g2: float) -> CatalogueTensorView:
        """Build one standardized device view and cache it for later blocks."""

        if not self.tensor_native_available:
            raise TypeError("flow model does not expose tensor-native inference")
        key = self.cache._key(g1, g2)
        cached = self._tensor_views.get(key)
        if cached is not None:
            return cached
        view = self.cache.get(*key)
        with torch.no_grad():
            context = self.flow_model.context_tensor(view.flow)
            mass = self.cache.prior.weights * view.detection_probability
            log_mass = np.full(len(mass), -np.inf, dtype=np.float32)
            positive = mass > 0
            log_mass[positive] = np.log(mass[positive]).astype(np.float32)
            log_detected_mass = torch.as_tensor(
                log_mass, dtype=context.dtype, device=context.device
            )
            if self.shape_target_indices is not None:
                scales = np.asarray(self.flow_model.target_transform.scales, dtype=np.float32)
                shift = np.zeros((len(view.flow), len(self.target_names)), dtype=np.float32)
                first, second = self.shape_target_indices
                shift[:, first] = view.blend_shift[:, 0] / scales[first]
                shift[:, second] = view.blend_shift[:, 1] / scales[second]
                blend_shift = torch.as_tensor(
                    shift, dtype=context.dtype, device=context.device
                )
            else:
                blend_shift = torch.empty(
                    0, dtype=context.dtype, device=context.device
                )
        cached = CatalogueTensorView(context, log_detected_mass, blend_shift)
        self._tensor_views[key] = cached
        return cached

    def discard_tensor_views(
        self, shears: Optional[Sequence[Sequence[float]]] = None
    ) -> None:
        """Release standardized accelerator views after a bounded-memory scan."""

        if shears is None:
            self._tensor_views.clear()
            self._raw_zero_context = None
            return
        for g1, g2 in shears:
            self._tensor_views.pop(self.cache._key(g1, g2), None)

    def raw_zero_context_tensor(self) -> torch.Tensor:
        """Return raw zero-shear flow features cached on the bundle device."""

        if not self.tensor_native_available:
            raise TypeError("flow model does not expose tensor-native inference")
        if self._raw_zero_context is None:
            preprocessor = self.flow_model.condition_preprocessor
            if not hasattr(preprocessor, "raw_tensor_from_frame"):
                raise TypeError("flow preprocessor lacks a raw tensor interface")
            zero = self.cache.get(0.0, 0.0).flow
            self._raw_zero_context = preprocessor.raw_tensor_from_frame(
                zero, self.flow_model.device
            )
        return self._raw_zero_context

    def observed_target_tensor(self, observed) -> torch.Tensor:
        """Standardize observed targets once and retain them on the device."""

        if not self.tensor_native_available:
            raise TypeError("flow model does not expose tensor-native inference")
        return self.flow_model.target_tensor(self._observed_frame(observed))

    def log_importance_weights_tensor(
        self,
        observed,
        g1: float,
        g2: float,
        *,
        atom_indices: np.ndarray,
        proposal_probability: np.ndarray,
        observed_targets: Optional[torch.Tensor] = None,
        object_chunk: int = 16,
        atom_chunk: int = 4096,
    ) -> torch.Tensor:
        """Tensor-native ``log[pi Pdet L / q]`` with device-side weights.

        Catalogue conditions are standardized once per shear view.  Each hot
        block then consists only of device gathers, flow evaluation, and
        device-side additions; no pandas object or host transfer is created in
        the atom loop.
        """

        frame = self._observed_frame(observed)
        indices = np.asarray(atom_indices, dtype=np.int64)
        proposal = np.asarray(proposal_probability, dtype=np.float64)
        if indices.ndim == 1:
            indices = np.broadcast_to(indices, (len(frame), len(indices)))
            proposal = np.broadcast_to(proposal, indices.shape)
        if indices.ndim != 2 or indices.shape[0] != len(frame):
            raise ValueError("atom_indices must have shape (M,) or (N, M)")
        if proposal.shape != indices.shape or not np.isfinite(proposal).all() or (proposal <= 0).any():
            raise ValueError("proposal probabilities must be positive and align with atoms")
        if ((indices < 0) | (indices >= len(self.cache.prior.galaxies))).any():
            raise ValueError("atom_indices contains an out-of-range row")
        if object_chunk <= 0 or atom_chunk <= 0:
            raise ValueError("chunk sizes must be positive")

        tensor_shear = self.tensor_shape_only_available
        tensor_view = self._tensor_view(0.0, 0.0) if tensor_shear else self._tensor_view(g1, g2)
        if observed_targets is None:
            observed_targets = self.flow_model.target_tensor(frame)
        if observed_targets.shape != (len(frame), len(self.target_names)):
            raise ValueError("observed target tensor has the wrong shape")
        if observed_targets.device != tensor_view.context.device:
            raise ValueError("observed targets are on the wrong device")

        device = tensor_view.context.device
        output = torch.empty(
            indices.shape, dtype=tensor_view.context.dtype, device=device
        )
        with torch.no_grad():
            for start in range(0, len(frame), object_chunk):
                stop = min(start + object_chunk, len(frame))
                selected_all = torch.as_tensor(
                    np.ascontiguousarray(indices[start:stop]),
                    dtype=torch.long,
                    device=device,
                )
                proposal_all = torch.as_tensor(
                    np.ascontiguousarray(proposal[start:stop]),
                    dtype=tensor_view.context.dtype,
                    device=device,
                )
                targets = observed_targets[start:stop]
                for atom_start in range(0, indices.shape[1], atom_chunk):
                    atom_stop = min(atom_start + atom_chunk, indices.shape[1])
                    selected = selected_all[:, atom_start:atom_stop]
                    width = atom_stop - atom_start
                    flat = selected.reshape(-1)
                    context = tensor_view.context.index_select(0, flat)
                    if tensor_shear:
                        context = self._shear_standardized_context(
                            context, flat, float(g1), float(g2)
                        )
                    target = (
                        targets[:, None, :]
                        .expand(stop - start, width, len(self.target_names))
                        .reshape(-1, len(self.target_names))
                    )
                    if self.shape_target_indices is not None:
                        target = target - tensor_view.blend_shift_standardized.index_select(
                            0, flat
                        )
                    log_flow = self.flow_model.log_prob_tensor(
                        target, context, batch_size=65536
                    ).reshape(stop - start, width)
                    block = (
                        log_flow
                        + tensor_view.log_detected_mass.index_select(0, flat).reshape(
                            stop - start, width
                        )
                        - torch.log(proposal_all[:, atom_start:atom_stop])
                    )
                    output[start:stop, atom_start:atom_stop] = block
        return output

    def _observed_frame(self, observed) -> pd.DataFrame:
        if isinstance(observed, pd.DataFrame):
            missing = sorted(set(self.target_names) - set(observed))
            if missing:
                raise KeyError(f"observations lack flow targets: {missing}")
            frame = observed.reset_index(drop=True).copy()
        else:
            values = np.asarray(observed, dtype=float)
            if values.ndim != 2 or values.shape[1] != len(self.target_names):
                raise ValueError(
                    f"observed array must have shape (n, {len(self.target_names)})"
                )
            frame = pd.DataFrame(values, columns=self.target_names)
        if self.selection is not None:
            values = frame.loc[:, self.target_names].to_numpy(dtype=float)
            passed = np.asarray(self.selection.output_cut(values), dtype=bool)
            if passed.shape != (len(frame),):
                raise ValueError("selection cut returned the wrong observed-data shape")
            if not passed.all():
                raise ValueError(
                    f"{int((~passed).sum())} observations fail the declared measured cut"
                )
        return frame

    def selection_probability(self, g1: float, g2: float) -> np.ndarray:
        """Per-atom ``P_pass``; identically one for an uncut likelihood."""

        view = self.cache.get(g1, g2)
        if self.selection is None:
            return np.ones(len(view.flow), dtype=np.float64)
        active = np.flatnonzero(self.cache.prior.weights > 0).astype(np.int64)
        return self.selection.probability(
            self.flow_model,
            view,
            active_indices=active,
        )

    def log_population_normalization(self, g1: float, g2: float) -> float:
        """Return ``log B_W(g)`` for the detected-and-selected population.

        ``B_W`` includes the empirical prior mass, the detection probability,
        and, when a measured-output cut is declared, the integrated pass
        probability of that cut.  Keeping this calculation in one method
        prevents exact, importance, and recentered likelihood paths from
        silently conditioning on different populations.
        """

        view = self.cache.get(g1, g2)
        selected_mass = (
            self.cache.prior.weights
            * view.detection_probability
            * self.selection_probability(g1, g2)
        )
        total = float(selected_mass.sum())
        if total <= 0:
            raise RuntimeError(
                "the catalogue prior has zero detected-and-selected probability"
            )
        return float(np.log(total))

    def _log_numerator(
        self,
        observed: pd.DataFrame,
        view: CatalogueModelView,
        *,
        atom_indices: Optional[np.ndarray],
        proposal_probability: Optional[np.ndarray],
        object_chunk: int,
        atom_chunk: int,
    ) -> np.ndarray:
        n_atoms = len(self.cache.prior.galaxies)
        if atom_indices is None:
            indices = np.flatnonzero(self.cache.prior.weights > 0).astype(np.int64)
            proposal = None
            common = True
            normalizer = 0.0
        else:
            indices = np.asarray(atom_indices, dtype=np.int64)
            if indices.ndim not in {1, 2} or indices.size == 0:
                raise ValueError("atom_indices is empty")
            if indices.ndim == 2 and indices.shape[0] != len(observed):
                raise ValueError("per-object atom_indices must have one row per observation")
            if ((indices < 0) | (indices >= n_atoms)).any():
                raise ValueError("atom_indices contains an out-of-range row")
            if proposal_probability is None:
                raise ValueError("sampled atoms require proposal_probability")
            proposal = np.asarray(proposal_probability, dtype=np.float64)
            if proposal.shape != indices.shape or not np.isfinite(proposal).all() or (proposal <= 0).any():
                raise ValueError("proposal probabilities must be positive and align with atoms")
            common = indices.ndim == 1
            normalizer = np.log(indices.shape[-1])

        result = np.full(len(observed), -np.inf, dtype=np.float64)
        for object_start in range(0, len(observed), object_chunk):
            object_stop = min(object_start + object_chunk, len(observed))
            obs = observed.iloc[object_start:object_stop]
            accumulator = np.full(len(obs), -np.inf, dtype=np.float64)
            m = indices.shape[-1]
            for atom_start in range(0, m, atom_chunk):
                atom_stop = min(atom_start + atom_chunk, m)
                if common:
                    selected = np.broadcast_to(
                        indices[atom_start:atom_stop],
                        (len(obs), atom_stop - atom_start),
                    )
                    selected_proposal = None if proposal is None else np.broadcast_to(
                        proposal[atom_start:atom_stop], selected.shape
                    )
                else:
                    selected = indices[object_start:object_stop, atom_start:atom_stop]
                    selected_proposal = proposal[
                        object_start:object_stop, atom_start:atom_stop
                    ]
                k = selected.shape[1]
                context = view.flow.iloc[selected.reshape(-1)].reset_index(drop=True)
                for target_index, name in enumerate(self.target_names):
                    values = np.repeat(obs[name].to_numpy(copy=False), k)
                    if self.shape_target_indices is not None:
                        if target_index == self.shape_target_indices[0]:
                            values = values - view.blend_shift[selected, 0].reshape(-1)
                        elif target_index == self.shape_target_indices[1]:
                            values = values - view.blend_shift[selected, 1].reshape(-1)
                    context[name] = values
                log_flow = self.flow_model.log_prob(context).reshape(len(obs), k)
                probability = view.detection_probability[selected]
                log_det = np.full(selected.shape, -np.inf, dtype=np.float64)
                positive = probability > 0
                log_det[positive] = np.log(probability[positive])
                log_mass = np.log(self.cache.prior.weights[selected])
                if selected_proposal is not None:
                    log_mass = log_mass - np.log(selected_proposal)
                terms = log_flow + log_det + log_mass
                accumulator = np.logaddexp(accumulator, logsumexp(terms, axis=1))
            result[object_start:object_stop] = accumulator - normalizer
        return result

    def log_likelihood(
        self,
        observed,
        g1: float,
        g2: float,
        *,
        atom_indices: Optional[np.ndarray] = None,
        proposal_probability: Optional[np.ndarray] = None,
        object_chunk: int = 64,
        atom_chunk: int = 4096,
    ) -> np.ndarray:
        """Log likelihood conditional on detection and the declared output cut.

        With no ``atom_indices`` this is the exact finite-catalogue sum.  With
        sampled indices it is the ordinary (not self-normalized) importance
        estimate of the flow numerator, including the required ``pi/q``
        correction.  Sample arrays may be shared ``(M,)`` or target-specific
        ``(N, M)``.  Detection normalization is exact; measured-cut
        normalization uses the cached common-random-number flow integral.
        """

        observed_frame = self._observed_frame(observed)
        view = self.cache.get(g1, g2)
        log_num = self._log_numerator(
            observed_frame,
            view,
            atom_indices=atom_indices,
            proposal_probability=proposal_probability,
            object_chunk=object_chunk,
            atom_chunk=atom_chunk,
        )
        # The deterministic measured cut cancels from the numerator for every
        # retained observation, but its integrated probability remains in the
        # selected-population normalization.  Detection and measured selection
        # are distinct: the flow is conditional on detection, and P_pass is an
        # integral over that conditional flow.
        return log_num - self.log_population_normalization(g1, g2)

    def log_importance_weights(
        self,
        observed,
        g1: float,
        g2: float,
        *,
        atom_indices: np.ndarray,
        proposal_probability: np.ndarray,
        object_chunk: int = 64,
        atom_chunk: int = 4096,
    ) -> np.ndarray:
        """Return ``log[pi_j Pdet_j L_ij / q_i(j)]`` for diagnostics."""

        observed = self._observed_frame(observed)
        indices = np.asarray(atom_indices, dtype=np.int64)
        proposal = np.asarray(proposal_probability, dtype=np.float64)
        if indices.ndim == 1:
            indices = np.broadcast_to(indices, (len(observed), len(indices)))
            proposal = np.broadcast_to(proposal, indices.shape)
        if indices.ndim != 2 or indices.shape[0] != len(observed):
            raise ValueError("atom_indices must have shape (M,) or (N, M)")
        if proposal.shape != indices.shape or not np.isfinite(proposal).all() or (proposal <= 0).any():
            raise ValueError("proposal probabilities must be positive and align with atoms")
        if ((indices < 0) | (indices >= len(self.cache.prior.galaxies))).any():
            raise ValueError("atom_indices contains an out-of-range row")

        view = self.cache.get(g1, g2)
        output = np.full(indices.shape, -np.inf, dtype=np.float64)
        m = indices.shape[1]
        for start in range(0, len(observed), object_chunk):
            stop = min(start + object_chunk, len(observed))
            obs = observed.iloc[start:stop]
            for atom_start in range(0, m, atom_chunk):
                atom_stop = min(atom_start + atom_chunk, m)
                selected = indices[start:stop, atom_start:atom_stop]
                k = atom_stop - atom_start
                context = view.flow.iloc[selected.reshape(-1)].reset_index(drop=True)
                for target_index, name in enumerate(self.target_names):
                    values = np.repeat(obs[name].to_numpy(copy=False), k)
                    if self.shape_target_indices is not None:
                        if target_index == self.shape_target_indices[0]:
                            values = values - view.blend_shift[selected, 0].reshape(-1)
                        elif target_index == self.shape_target_indices[1]:
                            values = values - view.blend_shift[selected, 1].reshape(-1)
                    context[name] = values
                log_flow = self.flow_model.log_prob(context).reshape(stop - start, k)
                probability = view.detection_probability[selected]
                positive = probability > 0
                block = output[start:stop, atom_start:atom_stop]
                block[positive] = (
                    log_flow[positive]
                    + np.log(self.cache.prior.weights[selected][positive])
                    - np.log(
                        proposal[start:stop, atom_start:atom_stop][positive]
                    )
                    + np.log(probability[positive])
                )
        return output

    def conditional_log_likelihood(
        self,
        observed,
        g1: float,
        g2: float,
        *,
        atom_indices: np.ndarray,
        object_chunk: int = 4096,
    ) -> np.ndarray:
        """Return ``log p_flow(x_i | S_g z_{j_i})`` for aligned atom rows.

        This deliberately excludes prior mass, detection probability, and the
        detected-population normalization.  Under the shape-only convention
        those factors are shear-invariant, so this curve isolates the
        conditional measurement-flow contribution to shear curvature.
        """

        observed = self._observed_frame(observed)
        indices = np.asarray(atom_indices, dtype=np.int64)
        if indices.shape != (len(observed),):
            raise ValueError("atom_indices must contain one row per observation")
        if ((indices < 0) | (indices >= len(self.cache.prior.galaxies))).any():
            raise ValueError("atom_indices contains an out-of-range row")
        if object_chunk <= 0:
            raise ValueError("object_chunk must be positive")

        view = self.cache.get(g1, g2)
        output = np.empty(len(observed), dtype=np.float64)
        for start in range(0, len(observed), object_chunk):
            stop = min(start + object_chunk, len(observed))
            rows = indices[start:stop]
            context = view.flow.iloc[rows].reset_index(drop=True)
            obs = observed.iloc[start:stop]
            for target_index, name in enumerate(self.target_names):
                values = obs[name].to_numpy(copy=False)
                if self.shape_target_indices is not None:
                    if target_index == self.shape_target_indices[0]:
                        values = values - view.blend_shift[rows, 0]
                    elif target_index == self.shape_target_indices[1]:
                        values = values - view.blend_shift[rows, 1]
                context[name] = values
            output[start:stop] = self.flow_model.log_prob(context)
        return output

    def score_and_information(
        self,
        observed,
        *,
        center: Sequence[float] = (0.0, 0.0),
        direction: Sequence[float] = (1.0, 0.0),
        delta: float = 0.01,
        richardson: bool = True,
        atom_indices: Optional[np.ndarray] = None,
        proposal_probability: Optional[np.ndarray] = None,
        object_chunk: int = 64,
        atom_chunk: int = 4096,
    ) -> CatalogueScore:
        """Differentiate one conditional-likelihood curve about ``center``."""

        center = np.asarray(center, dtype=float)
        direction = np.asarray(direction, dtype=float)
        if center.shape != (2,) or not np.isfinite(center).all():
            raise ValueError("center must be a finite two-vector")
        if direction.shape != (2,) or not np.isfinite(direction).all():
            raise ValueError("direction must be a finite two-vector")
        norm = float(np.linalg.norm(direction))
        if norm == 0:
            raise ValueError("direction must be nonzero")
        direction = direction / norm

        def curve(t):
            return self.log_likelihood(
                observed,
                center[0] + t * direction[0],
                center[1] + t * direction[1],
                atom_indices=atom_indices,
                proposal_probability=proposal_probability,
                object_chunk=object_chunk,
                atom_chunk=atom_chunk,
            )

        log_likelihood, score, curvature = curve_derivatives(
            curve, delta=delta, richardson=richardson
        )
        return CatalogueScore(log_likelihood, score, -curvature)


__all__ = [
    "CatalogueLikelihood",
    "CatalogueModelCache",
    "CatalogueModelView",
    "CatalogueSelection",
    "CatalogueScore",
    "SHEAR_INVARIANT_DETECTION_FEATURES",
    "ZERO_SHEAR_REUSABLE_FLOW_FEATURES",
    "predict_detection_probability",
]
