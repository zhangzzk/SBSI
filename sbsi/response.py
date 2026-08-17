"""Response prediction from a measurement-flow ensemble and emulator output.

This module provides the induced first-moment projection used to form the flow self-response
``R_flow`` that enters the certified, parameter-free decomposition

    m = R_sim / (R_flow + R_blend) - 1.

Public API
----------
load_sheared_sample(...)   Stream + select + reservoir-sample a sheared catalogue.
model_mean_proj(...)       ``< E[e_hat | S_{s*ghat}(intrinsic)] . ghat >`` -- the induced
                           first moment projected on the per-object applied-shear direction.
flow_response(...)         The antithetic +/-g secant ``R_flow = (m_+g - m_-g)/(2g)``,
                           optionally per-object, with optional Common-Random-Numbers reseeding.
sample_measurement(...)    Pooled conditional draws of the flow's measured targets
                           at each row's intrinsic input properties (the direct
                           flow output, before any response reduction).

Most users should call :meth:`ResponsePredictor.load` with their flow paths and
then pass their inference catalogue and external emulator responses to
:meth:`ResponsePredictor.predict`.  The lower-level functions remain public for
controlled response diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc

from .catalogue import Catalogue, load_catalogue
from .domain import Domain
from .measurement_model import raw_columns_for_measurement_targets
from .measurement_model import load_measurement_model
from .models import ModelPaths
from .blend_lookup import join_blend
from .preprocessing import (
    DEFAULT_SELECTION_CUTS,
    raw_columns_for_selection_features,
    rescale,
    source_select_selection,
)
from .shear_map import apply_shear_to_ellipticity


def load_sheared_sample(catalogue, bundle, max_rows, shear_threshold, seed, max_read_batches=None,
                        snr_min=None):
    """Stream a sheared catalogue, apply the standard cuts + detected, and reservoir-
    sample.  Keep intrinsic shape, applied-shear truth, and the measured targets.
    snr_min applies a measured-quality cut (measured_flux_auto/fluxerr_auto > snr_min)
    on the SHEARED measured quantity -- a shear-dependent selection."""
    rng = np.random.default_rng(seed)
    condition_features = bundle.condition_preprocessor.feature_names
    target_features = bundle.target_transform.target_names

    with ipc.open_file(catalogue) as reader:
        available = set(reader.schema.names)
        needed = set()
        needed |= raw_columns_for_selection_features(condition_features, available_columns=available)
        needed |= raw_columns_for_measurement_targets(target_features)
        needed |= {"detected", "gamma1_input_p", "gamma2_input_p"}
        needed |= {"e1_input_rot0_p", "e2_input_rot0_p"}
        needed |= {"r_input_p", "Re_input_p", "distance", "neighbored"}
        needed |= {"measured_flux_auto", "measured_fluxerr_auto", "measured_mag_auto"}
        read_columns = sorted(c for c in needed if c in available)

        reservoir = None
        raw_rows = 0
        for bi in range(reader.num_record_batches):
            if max_read_batches is not None and bi >= max_read_batches:
                break
            batch = pa.Table.from_batches([reader.get_batch(bi)]).select(read_columns).to_pandas()
            raw_rows += len(batch)
            batch = source_select_selection(batch, cuts=DEFAULT_SELECTION_CUTS)
            if len(batch) == 0:
                continue
            batch = batch[batch["detected"].astype(bool)].reset_index(drop=True)
            if len(batch) == 0:
                continue
            if snr_min is not None and "measured_flux_auto" in batch.columns:
                snr = batch["measured_flux_auto"].to_numpy(float) / batch["measured_fluxerr_auto"].to_numpy(float)
                batch = batch[np.isfinite(snr) & (snr > snr_min)].reset_index(drop=True)
                if len(batch) == 0:
                    continue
            gmag = np.hypot(batch["gamma1_input_p"].to_numpy(float), batch["gamma2_input_p"].to_numpy(float))
            batch = batch[gmag > shear_threshold].reset_index(drop=True)
            if len(batch) == 0:
                continue
            batch = batch.copy()
            batch["__key"] = rng.random(len(batch))
            reservoir = batch if reservoir is None else pd.concat([reservoir, batch], ignore_index=True)
            if len(reservoir) > 2 * max_rows:
                reservoir = reservoir.nlargest(max_rows, "__key").reset_index(drop=True)

    if reservoir is None:
        raise SystemExit(f"No sheared rows selected from {catalogue}")
    if len(reservoir) > max_rows:
        reservoir = reservoir.nlargest(max_rows, "__key").reset_index(drop=True)
    reservoir = reservoir.drop(columns="__key").reset_index(drop=True)
    print(f"  raw scanned={raw_rows:,}  kept (sheared)={len(reservoir):,}")
    return reservoir


def _shape_target_indices(names):
    """Locate the (e1,e2)-like shape target pair among the flow's target names,
    supporting both SExtractor (measured_e1_image/e2_image) and ngmix
    (measured_ngmix_g1/g2) conventions."""
    for c1, c2 in (("measured_e1_image", "measured_e2_image"),
                   ("measured_ngmix_g1", "measured_ngmix_g2"),
                   ("measured_galsim_g1", "measured_galsim_g2")):
        if c1 in names and c2 in names:
            return names.index(c1), names.index(c2)
    raise KeyError(f"No known shape target pair in {names}")


def model_mean_proj(bundle, base, s, ghat1, ghat2, intrinsic, rescale_kwargs,
                    n_samples, batch_size, return_proj=False):
    """< E[e_hat | S_{s*ghat}(intrinsic)] . ghat >  -- induced flow first moment
    projected onto the per-object applied-shear direction.

    Returns (global_mean, sem).  With return_proj=True also returns the per-object
    projection array `proj` (shape N,), letting callers form a per-object response
    (proj_{+g} - proj_{-g})/(2g).  Because the global mean is exactly np.mean(proj),
    the scalar response is identical whether taken from the two means or from the
    per-object array -- so exposing proj never changes the certified global R_flow."""
    frame = base.copy()
    e1p, e2p = apply_shear_to_ellipticity(intrinsic[0], intrinsic[1], s * ghat1, s * ghat2)
    frame["e1_input_rot0_p"] = e1p
    frame["e2_input_rot0_p"] = e2p
    frame = rescale(frame, **rescale_kwargs)
    draws = bundle.sample(frame, n_samples=n_samples, batch_size=batch_size)  # (N, n_samples, dim)
    mean = draws.mean(axis=1)  # (N, dim) in engineered target units
    i1, i2 = _shape_target_indices(bundle.target_transform.target_names)
    proj = mean[:, i1] * ghat1 + mean[:, i2] * ghat2
    gmean, sem = float(np.mean(proj)), float(np.std(proj) / np.sqrt(len(proj)))
    if return_proj:
        return gmean, sem, proj
    return gmean, sem


def flow_response(bundle, base, g, ghat1, ghat2, intrinsic, rescale_kwargs,
                  n_samples, batch_size, reseed=None, return_perobj=False):
    """Antithetic +/-g secant of the induced first moment -- the flow self-response

        R_flow = ( <proj(+g)> - <proj(-g)> ) / (2 g).

    This is the canonical way to read the trained flow's shape-response; it folds the
    two ``model_mean_proj`` legs the certified harvest evaluates inline.  ``reseed``, if
    given, is called with no arguments immediately before each leg so both legs share
    Common Random Numbers (flow-sampling noise cancels in the difference) -- variance
    reduction only, it does not bias R_flow.  With return_perobj=True also returns the
    per-object secant ``(proj_{+g} - proj_{-g})/(2 g)``; its mean is identically the
    scalar R_flow, so exposing it never changes the certified number.
    """
    if reseed is not None:
        reseed()
    mp, _, projp = model_mean_proj(bundle, base, +g, ghat1, ghat2, intrinsic,
                                   rescale_kwargs, n_samples, batch_size, return_proj=True)
    if reseed is not None:
        reseed()
    mm, _, projm = model_mean_proj(bundle, base, -g, ghat1, ghat2, intrinsic,
                                   rescale_kwargs, n_samples, batch_size, return_proj=True)
    R = (mp - mm) / (2 * g)
    if return_perobj:
        return R, (projp - projm) / (2 * g)
    return R


def predict_blend_response(
    emulator,
    pairs: Catalogue,
) -> pd.Series:
    """Predict pair responses and sum them on the pair table's primary index.

    Pair preparation is deliberately separate and explicit. Pass the
    ``emulator_pairs`` view returned by :func:`prepare_forward_catalogue` (or a
    compatible custom pair table). The returned Series is indexed by
    ``primary_row`` so the one-row-per-primary flow view can align by key rather
    than relying on incidental row order.
    """

    pairs = load_catalogue(pairs)
    required = {"primary_row", "distance_scaled"}
    missing = sorted(required - set(pairs.columns))
    if missing:
        raise KeyError(f"prepared pair catalogue is missing columns: {missing}")
    if pairs.empty:
        raise ValueError("prepared pair catalogue is empty")
    if not hasattr(emulator, "predict_on_pairs"):
        raise TypeError("emulator must provide BlendEMU's predict_on_pairs API")
    predicted = emulator.predict_on_pairs(pairs, task="response", rescaled=True)
    missing = sorted({"primary_row", "response"} - set(predicted.columns))
    if missing:
        raise RuntimeError(f"BlendEMU pair prediction lacks columns: {missing}")
    if not np.isfinite(predicted["response"].to_numpy(dtype=float)).all():
        raise ValueError("BlendEMU returned a missing or non-finite pair response")

    summed = predicted.groupby("primary_row", sort=False)["response"].sum()
    indices = summed.index.to_numpy(dtype=np.int64)
    if np.any(indices < 0):
        raise RuntimeError("prepared pair table contains a negative primary row")
    return pd.Series(
        summed.to_numpy(dtype=np.float64),
        index=pd.Index(indices, name="primary_row"),
        name="R_blend",
    )


@dataclass(frozen=True)
class ResponsePrediction:
    """Per-object response from a flow ensemble plus emulator output."""

    model: str
    index: np.ndarray
    flow_by_seed: np.ndarray
    blend: np.ndarray

    def __post_init__(self):
        flow = np.asarray(self.flow_by_seed)
        blend = np.asarray(self.blend)
        index = np.asarray(self.index)
        if flow.ndim != 2 or flow.shape[0] == 0 or flow.shape[1] == 0:
            raise ValueError("flow_by_seed must have shape (n_seeds, n_objects)")
        if blend.shape != (flow.shape[1],) or index.shape != (flow.shape[1],):
            raise ValueError("index and blend must match the flow object axis")
        if not np.isfinite(flow).all() or not np.isfinite(blend).all():
            raise ValueError("response arrays must be finite")

    @property
    def flow(self):
        return self.flow_by_seed.mean(axis=0)

    @property
    def total(self):
        return self.flow + self.blend

    @property
    def seed_means(self):
        return self.flow_by_seed.mean(axis=1)

    @property
    def flow_mean(self):
        return float(self.flow.mean())

    @property
    def blend_mean(self):
        return float(self.blend.mean())

    @property
    def total_mean(self):
        return float(self.total.mean())

    @property
    def flow_seed_sem(self):
        values = self.seed_means
        if len(values) < 2:
            return float("nan")
        return float(values.std(ddof=1) / np.sqrt(len(values)))

    def multiplicative_bias(self, simulation_response):
        """Return ``mean(R_sim) / mean(R_model) - 1`` using SBSI's sign convention."""

        response = np.asarray(simulation_response, dtype=float)
        sim_mean = float(response) if response.ndim == 0 else float(response.mean())
        return sim_mean / self.total_mean - 1.0

    def summary(self):
        return {
            "model": self.model,
            "n_objects": int(len(self.index)),
            "n_seeds": int(self.flow_by_seed.shape[0]),
            "R_flow": self.flow_mean,
            "R_blend": self.blend_mean,
            "R_total": self.total_mean,
            "R_flow_seed_sem": self.flow_seed_sem,
        }


@dataclass(frozen=True)
class MeasurementSample:
    """Pooled conditional draws of the measured parameters from the flow ensemble.

    ``samples`` has shape ``(n_objects, n_seeds * n_samples, n_targets)`` in
    physical units (magnitudes, log FLUX_RADIUS in pixels, ellipticity
    components), column order following ``target_names``.  Draws from every
    flow seed are pooled along the draw axis, so the array represents the
    ensemble predictive distribution of the measurement model.
    """

    model: str
    index: np.ndarray
    target_names: Tuple[str, ...]
    samples: np.ndarray

    def __post_init__(self):
        samples = np.asarray(self.samples)
        index = np.asarray(self.index)
        target_names = tuple(self.target_names)
        if samples.ndim != 3 or samples.shape[0] != len(index):
            raise ValueError(
                "samples must have shape (n_objects, n_draws, n_targets) matching index"
            )
        if samples.shape[1] == 0 or samples.shape[2] != len(target_names):
            raise ValueError("samples carry no draws or do not match target_names")
        object.__setattr__(self, "samples", samples)
        object.__setattr__(self, "index", index)
        object.__setattr__(self, "target_names", target_names)

    def column(self, name: str) -> np.ndarray:
        """Return one measured parameter as an ``(n_objects, n_draws)`` array."""
        if name not in self.target_names:
            raise KeyError(
                f"unknown target {name!r}; available: {list(self.target_names)}"
            )
        return self.samples[:, :, self.target_names.index(name)]


class ResponsePredictor:
    """Predict ``R_flow + R_blend`` from user-selected model artifacts.

    The flow is evaluated with a central antithetic secant and common random
    numbers.  Blend response is supplied in the inference catalogue, as an
    aligned array, or in a separate user-supplied catalogue joined on
    ``(case, input_index)``.  Missing rows are dropped; they are never
    interpreted as zero response.  The support domain is read from flow
    checkpoint metadata unless the caller supplies it explicitly.
    """

    def __init__(
        self,
        flow_checkpoints: Union[Sequence[Union[str, Path]], ModelPaths],
        device: Optional[str] = None,
        *,
        domain: Optional[Domain] = None,
        label: Optional[str] = None,
    ):
        if isinstance(flow_checkpoints, ModelPaths):
            paths = flow_checkpoints.flow_checkpoints
            label = label or flow_checkpoints.name
        else:
            paths = flow_checkpoints
        self.checkpoints = tuple(Path(path) for path in paths)
        if not self.checkpoints:
            raise ValueError("at least one flow checkpoint is required")
        missing = [path for path in self.checkpoints if not path.is_file()]
        if missing:
            rendered = "\n  ".join(str(path) for path in missing)
            raise FileNotFoundError(f"missing flow checkpoints:\n  {rendered}")

        import torch

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.label = label or "custom"
        self._first_bundle = load_measurement_model(
            str(self.checkpoints[0]), device=self.device
        )
        self._checkpoint_domain = Domain.from_flow_metadata(self._first_bundle.metadata)
        self.domain = domain or self._checkpoint_domain

    @classmethod
    def load(
        cls,
        flow_checkpoints: Union[Sequence[Union[str, Path]], ModelPaths],
        *,
        device: Optional[str] = None,
        domain: Optional[Domain] = None,
        label: Optional[str] = None,
    ) -> "ResponsePredictor":
        """Load a flow ensemble from explicit paths or a path-only preset."""

        return cls(
            flow_checkpoints,
            device=device,
            domain=domain,
            label=label,
        )

    @property
    def condition_features(self):
        return tuple(self._first_bundle.condition_preprocessor.feature_names)

    @property
    def target_features(self):
        return tuple(self._first_bundle.target_transform.target_names)

    def _domain_mask(self, frame):
        needed = {"r_input_p", "Re_input_p"}
        missing = sorted(needed - set(frame.columns))
        if missing:
            raise KeyError(f"response frame lacks domain columns: {missing}")
        return self.domain.mask(
            frame["r_input_p"].to_numpy(float),
            frame["Re_input_p"].to_numpy(float),
        )

    def predict(
        self,
        catalogue: Catalogue,
        *,
        shear: float = 0.02,
        shear_direction: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        blend_response: Union[str, pd.Series, np.ndarray, Sequence[float]] = "R_blend",
        blend_catalogue: Optional[Catalogue] = None,
        n_samples: int = 64,
        batch_size: int = 16384,
        random_seed: int = 12345,
        rescale_kwargs: Optional[dict] = None,
        strict_domain: bool = True,
        minimum_lookup_match: float = 0.9999,
    ) -> ResponsePrediction:
        frame = load_catalogue(catalogue)
        if shear <= 0:
            raise ValueError("shear must be positive")
        if n_samples <= 0 or batch_size <= 0:
            raise ValueError("n_samples and batch_size must be positive")
        if not 0.0 <= minimum_lookup_match <= 1.0:
            raise ValueError("minimum_lookup_match must lie in [0, 1]")
        if len(frame) == 0:
            raise ValueError("response frame is empty")
        frame = frame.copy()
        original_size = len(frame)
        directions = None
        if shear_direction is not None:
            directions = (
                np.asarray(shear_direction[0], dtype=float),
                np.asarray(shear_direction[1], dtype=float),
            )
            if any(values.shape != (original_size,) for values in directions):
                raise ValueError("shear_direction arrays must match the input frame")
        domain = self._domain_mask(frame)
        if strict_domain and not domain.all():
            raise ValueError(
                f"{int((~domain).sum()):,} of {len(frame):,} rows are outside "
                "the trained flow domain"
            )
        if not strict_domain:
            frame = frame.loc[domain].copy()
            if directions is not None:
                directions = tuple(values[domain] for values in directions)
            if frame.empty:
                raise ValueError("no rows remain inside the trained flow domain")

        if blend_catalogue is not None:
            if not isinstance(blend_response, str):
                raise TypeError(
                    "blend_response must name a column when blend_catalogue is supplied"
                )
            missing = sorted({"case", "input_index"} - set(frame.columns))
            if missing:
                raise KeyError(
                    f"joining the emulator catalogue needs key columns {missing}"
                )
            lookup = load_catalogue(blend_catalogue)
            blend, keep, _, _ = join_blend(
                frame,
                lookup,
                blend_response,
                min_match=minimum_lookup_match,
                label="emulator response catalogue",
            )
            frame = frame.loc[keep].copy()
            blend = blend[keep]
            if directions is not None:
                directions = tuple(values[keep] for values in directions)
            if frame.empty:
                raise ValueError("no rows remain after the blend lookup join")
        elif isinstance(blend_response, str):
            if blend_response not in frame.columns:
                raise KeyError(
                    f"inference catalogue lacks emulator column {blend_response!r}; "
                    "supply blend_catalogue or an aligned response array"
                )
            blend = frame[blend_response].to_numpy(dtype=float)
            if not np.isfinite(blend).all():
                raise ValueError("emulator response column contains missing or non-finite values")
        elif isinstance(blend_response, pd.Series):
            if not blend_response.index.is_unique:
                raise ValueError("blend_response Series index must be unique")
            aligned = blend_response.reindex(frame.index)
            if aligned.isna().any():
                raise ValueError(
                    "blend_response Series does not cover every retained flow primary"
                )
            blend = aligned.to_numpy(dtype=float)
            if not np.isfinite(blend).all():
                raise ValueError("blend_response contains non-finite values")
        else:
            blend = np.asarray(blend_response, dtype=float)
            if not strict_domain and blend.shape == (original_size,):
                blend = blend[domain]
            if blend.shape != (len(frame),):
                raise ValueError(f"blend_response shape {blend.shape} != ({len(frame)},)")
            if not np.isfinite(blend).all():
                raise ValueError("blend_response contains missing or non-finite values")

        required = {
            "e1_input_rot0_p",
            "e2_input_rot0_p",
            "r_input_s",
            "Re_input_s",
            "distance",
        }
        missing = sorted(required - set(frame.columns))
        if missing:
            raise KeyError(f"response frame lacks flow/rescaling columns: {missing}")
        intrinsic = (
            frame["e1_input_rot0_p"].to_numpy(float),
            frame["e2_input_rot0_p"].to_numpy(float),
        )
        if directions is None:
            ghat1 = np.ones(len(frame), dtype=float)
            ghat2 = np.zeros(len(frame), dtype=float)
        else:
            ghat1, ghat2 = directions
            norm = np.hypot(ghat1, ghat2)
            if np.any(norm <= 0) or not np.isfinite(norm).all():
                raise ValueError("shear directions must be finite and non-zero")
            ghat1, ghat2 = ghat1 / norm, ghat2 / norm

        import torch

        device = self.device
        optics = dict(
            pixel_rms=0.312,
            pixel_size=0.2,
            zero_mag=30.0,
            psf_fwhm=0.73,
            moffat_beta=2.224,
        )
        optics.update(rescale_kwargs or {})
        seed_rows = []
        for checkpoint_index, checkpoint in enumerate(self.checkpoints):
            bundle = (
                self._first_bundle
                if checkpoint_index == 0
                else load_measurement_model(str(checkpoint), device=device)
            )
            checkpoint_domain = Domain.from_flow_metadata(bundle.metadata)
            if checkpoint_domain != self._checkpoint_domain:
                raise ValueError(
                    f"flow checkpoint domain differs from the ensemble domain: {checkpoint}"
                )

            def reseed():
                torch.manual_seed(random_seed)

            _, per_object = flow_response(
                bundle,
                frame,
                shear,
                ghat1,
                ghat2,
                intrinsic,
                optics,
                n_samples,
                batch_size,
                reseed=reseed,
                return_perobj=True,
            )
            seed_rows.append(np.asarray(per_object, dtype=np.float64))
            if checkpoint_index != 0:
                del bundle
            if str(device).startswith("cuda"):
                torch.cuda.empty_cache()

        return ResponsePrediction(
            model=self.label,
            index=frame.index.to_numpy(copy=True),
            flow_by_seed=np.stack(seed_rows, axis=0),
            blend=np.asarray(blend, dtype=np.float64),
        )


def sample_measurement(
    flow_checkpoints: Union[Sequence[Union[str, Path]], ModelPaths],
    catalogue: Catalogue,
    *,
    n_samples: int = 256,
    batch_size: int = 16384,
    random_seed: int = 12345,
    rescale_kwargs: Optional[dict] = None,
    device: Optional[str] = None,
) -> MeasurementSample:
    """Draw the flow ensemble's measured-parameter distribution per object.

    For every row of ``catalogue`` -- the same one-row-per-primary frame
    :meth:`ResponsePredictor.predict` consumes -- draw ``n_samples`` vectors
    of the flow's measured targets at the row's intrinsic input properties.
    No shear is applied: the draws are the direct conditional distribution
    ``p(measured | input galaxy, neighbours)`` the flow models.  Draws from
    every checkpoint are pooled (``n_seeds * n_samples`` per object), giving
    the ensemble predictive; the same ``random_seed`` makes repeat calls
    reproducible.
    """

    if isinstance(flow_checkpoints, ModelPaths):
        paths = flow_checkpoints.flow_checkpoints
        label = flow_checkpoints.name
    else:
        paths = list(flow_checkpoints)
        label = "custom"
    checkpoints = tuple(Path(path) for path in paths)
    if not checkpoints:
        raise ValueError("at least one flow checkpoint is required")
    if n_samples <= 0 or batch_size <= 0:
        raise ValueError("n_samples and batch_size must be positive")
    missing = [path for path in checkpoints if not path.is_file()]
    if missing:
        rendered = "\n  ".join(str(path) for path in missing)
        raise FileNotFoundError(f"missing flow checkpoints:\n  {rendered}")

    import torch

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    frame = load_catalogue(catalogue)
    if len(frame) == 0:
        raise ValueError("measurement frame is empty")
    required = {
        "e1_input_rot0_p",
        "e2_input_rot0_p",
        "r_input_p",
        "Re_input_p",
        "r_input_s",
        "Re_input_s",
        "distance",
    }
    missing_columns = sorted(required - set(frame.columns))
    if missing_columns:
        raise KeyError(f"measurement frame lacks flow/rescaling columns: {missing_columns}")

    first_bundle = load_measurement_model(str(checkpoints[0]), device=device)
    ensemble_domain = Domain.from_flow_metadata(first_bundle.metadata)
    target_names = tuple(first_bundle.target_transform.target_names)

    domain_mask = ensemble_domain.mask(
        frame["r_input_p"].to_numpy(float),
        frame["Re_input_p"].to_numpy(float),
    )
    if not domain_mask.all():
        raise ValueError(
            f"{int((~domain_mask).sum()):,} of {len(frame):,} rows are outside "
            "the trained flow domain"
        )

    optics = dict(
        pixel_rms=0.312,
        pixel_size=0.2,
        zero_mag=30.0,
        psf_fwhm=0.73,
        moffat_beta=2.224,
    )
    optics.update(rescale_kwargs or {})
    conditions = rescale(frame.copy(), **optics)

    draws = []
    for checkpoint_index, checkpoint in enumerate(checkpoints):
        bundle = (
            first_bundle
            if checkpoint_index == 0
            else load_measurement_model(str(checkpoint), device=device)
        )
        if Domain.from_flow_metadata(bundle.metadata) != ensemble_domain:
            raise ValueError(
                f"flow checkpoint domain differs from the ensemble domain: {checkpoint}"
            )
        if tuple(bundle.target_transform.target_names) != target_names:
            raise ValueError(
                f"flow checkpoint targets differ from the ensemble targets: {checkpoint}"
            )
        torch.manual_seed(random_seed)
        draws.append(
            bundle.sample(conditions, n_samples=n_samples, batch_size=batch_size)
        )
        if checkpoint_index != 0:
            del bundle
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()
    samples = np.concatenate(draws, axis=1)

    return MeasurementSample(
        model=label,
        index=frame.index.to_numpy(copy=True),
        target_names=target_names,
        samples=samples,
    )


__all__ = [
    "MeasurementSample",
    "ResponsePrediction",
    "ResponsePredictor",
    "flow_response",
    "load_sheared_sample",
    "model_mean_proj",
    "predict_blend_response",
    "sample_measurement",
]
