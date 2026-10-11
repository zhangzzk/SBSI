"""Response prediction from the measurement flow and the blending emulator.

The model response splits into two parts,

    R_model = R_flow + R_blend,   m = R_sim / R_model - 1.

``R_flow`` is how an isolated galaxy's measured shape follows an applied shear;
``R_blend`` is the extra shift its neighbours cause.  Both come from the same
kind of flow draws: each galaxy is measured many times at ``+shear`` and
``-shear`` along g1 with shared (antithetic) random draws, and the response is
the difference of the mean measured g1 divided by ``2 * shear``.

Most users call :meth:`ResponsePredictor.load` and
:func:`predict_blend_response` on a :func:`~sbsi.prepare_forward_catalogue`
result.  These per-galaxy responses use every draw, with no cut on the
measurement; selection-aware responses come from the emulator's
:meth:`~sbsi.disk_predictor.DiskResponsePredictor.predict`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from .catalogue import Catalogue, load_catalogue
from .crowding import V39_FEATURES
from .disk_predictor import RANDOM_SEED, draw_moments, load_flow
from .domain import Domain
from .models import ModelPaths


def _flow_checkpoint(flow_checkpoints: Union[Sequence[Union[str, Path]], ModelPaths]):
    if isinstance(flow_checkpoints, ModelPaths):
        paths, label = flow_checkpoints.flow_checkpoints, flow_checkpoints.name
    else:
        paths, label = tuple(flow_checkpoints), None
    if len(paths) != 1:
        raise ValueError("exactly one flow checkpoint is required")
    path = Path(paths[0])
    if not path.is_file():
        raise FileNotFoundError(f"missing flow checkpoint: {path}")
    return path, label


def predict_blend_response(
    emulator,
    pairs: Catalogue,
    *,
    shear: float = 0.02,
    n_samples: int = 64,
    batch_size: int = 1024,
    random_seed: int = RANDOM_SEED,
) -> pd.Series:
    """Per-galaxy blending response, indexed by ``primary_row``.

    Pass the ``emulator_pairs`` view returned by
    :func:`prepare_forward_catalogue`; each pair row carries its primary's flow
    inputs.  The emulator draws ``n_samples`` measurements of every primary at
    ``+shear`` and ``-shear`` and moves each draw's shape by its neighbours'
    predicted response; ``R_blend`` is the mean move per unit shear.  With the
    same seed, batch size and device as :meth:`ResponsePredictor.predict`,
    both parts use the same draws.
    """

    pairs = load_catalogue(pairs)
    required = {"primary_row", "secondary_row", *V39_FEATURES}
    missing = sorted(required - set(pairs.columns))
    if missing:
        raise KeyError(f"prepared pair catalogue is missing columns: {missing}")
    if pairs.empty:
        raise ValueError("prepared pair catalogue is empty")
    if not hasattr(emulator, "predict"):
        raise TypeError("emulator must be the model returned by load_emulator")
    inputs = (
        pairs.drop_duplicates("primary_row")
        .set_index("primary_row")
        .sort_index()[list(V39_FEATURES)]
    )
    prediction = emulator.predict(
        inputs,
        pairs,
        shear=shear,
        n_draws=n_samples,
        batch_size=batch_size,
        random_seed=random_seed,
    )
    blend = prediction.per_galaxy()["R_blend"]
    return pd.Series(
        blend.to_numpy(dtype=np.float64),
        index=pd.Index(prediction.index, name="primary_row"),
        name="R_blend",
    )


@dataclass(frozen=True)
class ResponsePrediction:
    """Per-object flow response plus emulator output."""

    model: str
    index: np.ndarray
    flow: np.ndarray
    blend: np.ndarray

    def __post_init__(self):
        flow = np.asarray(self.flow)
        if flow.ndim != 1 or flow.size == 0:
            raise ValueError("flow must be a non-empty per-object array")
        if np.shape(self.blend) != flow.shape or np.shape(self.index) != flow.shape:
            raise ValueError("index and blend must match the flow object axis")
        if not np.isfinite(flow).all() or not np.isfinite(self.blend).all():
            raise ValueError("response arrays must be finite")

    @property
    def total(self):
        return self.flow + self.blend

    @property
    def flow_mean(self):
        return float(self.flow.mean())

    @property
    def blend_mean(self):
        return float(self.blend.mean())

    @property
    def total_mean(self):
        return float(self.total.mean())

    def multiplicative_bias(self, simulation_response):
        """Return ``mean(R_sim) / mean(R_model) - 1`` using SBSI's sign convention."""

        response = np.asarray(simulation_response, dtype=float)
        sim_mean = float(response) if response.ndim == 0 else float(response.mean())
        return sim_mean / self.total_mean - 1.0

    def summary(self):
        """Catalogue means with 1-sigma errors (object-to-object scatter / sqrt(n)).

        The errors include the draw noise of every object.
        """

        n = len(self.index)
        out = {"model": self.model, "n_objects": int(n)}
        for name, values in (("R_flow", self.flow), ("R_blend", self.blend), ("R_total", self.total)):
            out[name] = float(values.mean())
            out[f"{name}_err"] = float(values.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
        return out


# The flow draws FLUX_RADIUS in pixels and the flux at zero point 30 that
# MAG_AUTO was converted from; these give the conventional coordinates.
_DERIVED_TARGETS = {
    "measured_log_flux_radius": ("measured_flux_radius", np.log),
    "measured_mag_auto": ("measured_flux_from_mag_auto", lambda flux: 30.0 - 2.5 * np.log10(flux)),
}


@dataclass(frozen=True)
class MeasurementSample:
    """Conditional draws of the measured parameters from the flow.

    ``samples`` has shape ``(n_objects, n_samples, n_targets)`` in the flow's
    physical units, column order following ``target_names``: measured g1, g2,
    FLUX_RADIUS in pixels, and flux at zero point 30.  :meth:`column` also
    returns ``measured_log_flux_radius`` and ``measured_mag_auto``.
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
        """Return one measured parameter as an ``(n_objects, n_draws)`` array.

        A draw with zero size or flux gives an infinite log radius or magnitude.
        """
        if name in _DERIVED_TARGETS and name not in self.target_names:
            source, transform = _DERIVED_TARGETS[name]
            with np.errstate(divide="ignore"):
                return transform(self.column(source))
        if name not in self.target_names:
            raise KeyError(
                f"unknown target {name!r}; available: "
                f"{[*self.target_names, *_DERIVED_TARGETS]}"
            )
        return self.samples[:, :, self.target_names.index(name)]


class ResponsePredictor:
    """Predict per-object ``R_flow`` and attach the emulator's ``R_blend``.

    The flow is evaluated at ``+shear`` and ``-shear`` with shared antithetic
    draws.  Blend response is supplied as a catalogue column, an aligned array
    or a Series keyed like the catalogue (``predict_blend_response``).  The
    support domain is read from the flow checkpoint unless supplied.
    """

    def __init__(
        self,
        flow_checkpoints: Union[Sequence[Union[str, Path]], ModelPaths],
        device: Optional[str] = None,
        *,
        domain: Optional[Domain] = None,
        label: Optional[str] = None,
    ):
        import torch

        checkpoint, model_label = _flow_checkpoint(flow_checkpoints)
        self.checkpoints = (checkpoint,)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.label = label or model_label or "custom"
        self._flow = load_flow(checkpoint, self.device)
        self.domain = domain or Domain.from_flow_metadata(self._flow.metadata)

    @classmethod
    def load(
        cls,
        flow_checkpoints: Union[Sequence[Union[str, Path]], ModelPaths],
        *,
        device: Optional[str] = None,
        domain: Optional[Domain] = None,
        label: Optional[str] = None,
    ) -> "ResponsePredictor":
        """Load the flow from explicit paths or a path-only preset."""

        return cls(flow_checkpoints, device=device, domain=domain, label=label)

    @property
    def condition_features(self):
        return tuple(self._flow.condition_preprocessor.feature_names)

    @property
    def target_features(self):
        return tuple(self._flow.target_transform.target_names)

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
        blend_response: Union[str, pd.Series, np.ndarray, Sequence[float]] = "R_blend",
        n_samples: int = 64,
        batch_size: int = 1024,
        random_seed: int = RANDOM_SEED,
        strict_domain: bool = True,
    ) -> ResponsePrediction:
        frame = load_catalogue(catalogue)
        if n_samples <= 0 or batch_size <= 0:
            raise ValueError("n_samples and batch_size must be positive")
        if len(frame) == 0:
            raise ValueError("response frame is empty")
        frame = frame.copy()
        original_size = len(frame)
        domain = self._domain_mask(frame)
        if strict_domain and not domain.all():
            raise ValueError(
                f"{int((~domain).sum()):,} of {len(frame):,} rows are outside "
                "the trained flow domain"
            )
        if not strict_domain:
            frame = frame.loc[domain].copy()
            if frame.empty:
                raise ValueError("no rows remain inside the trained flow domain")

        if isinstance(blend_response, str):
            if blend_response not in frame.columns:
                raise KeyError(
                    f"inference catalogue lacks emulator column {blend_response!r}; "
                    "supply an aligned response array or Series"
                )
            blend = frame[blend_response].to_numpy(dtype=float)
        elif isinstance(blend_response, pd.Series):
            if not blend_response.index.is_unique:
                raise ValueError("blend_response Series index must be unique")
            aligned = blend_response.reindex(frame.index)
            if aligned.isna().any():
                raise ValueError(
                    "blend_response Series does not cover every retained flow primary"
                )
            blend = aligned.to_numpy(dtype=float)
        else:
            blend = np.asarray(blend_response, dtype=float)
            if not strict_domain and blend.shape == (original_size,):
                blend = blend[domain]
            if blend.shape != (len(frame),):
                raise ValueError(f"blend_response shape {blend.shape} != ({len(frame)},)")
        if not np.isfinite(blend).all():
            raise ValueError("blend_response contains missing or non-finite values")

        missing = sorted(set(V39_FEATURES) - set(frame.columns))
        if missing:
            raise KeyError(f"response frame lacks flow input columns: {missing}")
        _, kept, _ = draw_moments(
            self._flow,
            frame,
            shear=shear,
            n_draws=n_samples,
            batch_size=batch_size,
            random_seed=random_seed,
        )
        flow = (kept[0, :, 0, 0] - kept[1, :, 0, 0]) / (2 * shear)
        return ResponsePrediction(
            model=self.label,
            index=frame.index.to_numpy(copy=True),
            flow=flow,
            blend=np.asarray(blend, dtype=np.float64),
        )


def sample_measurement(
    flow_checkpoints: Union[Sequence[Union[str, Path]], ModelPaths],
    catalogue: Catalogue,
    *,
    n_samples: int = 256,
    batch_size: int = 16384,
    random_seed: int = 12345,
    device: Optional[str] = None,
) -> MeasurementSample:
    """Draw the flow's measured-parameter distribution per object.

    For every row of ``catalogue`` -- the same one-row-per-primary frame
    :meth:`ResponsePredictor.predict` consumes -- draw ``n_samples`` vectors
    of the flow's measured targets at the row's input properties.  No shear
    and no blending shift are applied: the draws are the flow's direct
    distribution ``p(measured | input galaxy, neighbours)``.  The same
    ``random_seed`` makes repeat calls reproducible.
    """

    checkpoint, label = _flow_checkpoint(flow_checkpoints)
    if n_samples <= 0 or batch_size <= 0:
        raise ValueError("n_samples and batch_size must be positive")

    import torch

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    frame = load_catalogue(catalogue)
    if len(frame) == 0:
        raise ValueError("measurement frame is empty")
    required = {"r_input_p", "Re_input_p", *V39_FEATURES}
    missing_columns = sorted(required - set(frame.columns))
    if missing_columns:
        raise KeyError(f"measurement frame lacks flow input columns: {missing_columns}")

    flow = load_flow(checkpoint, device)
    domain_mask = Domain.from_flow_metadata(flow.metadata).mask(
        frame["r_input_p"].to_numpy(float),
        frame["Re_input_p"].to_numpy(float),
    )
    if not domain_mask.all():
        raise ValueError(
            f"{int((~domain_mask).sum()):,} of {len(frame):,} rows are outside "
            "the trained flow domain"
        )
    torch.manual_seed(random_seed)
    samples = flow.sample(frame[list(V39_FEATURES)], n_samples=n_samples, batch_size=batch_size)
    return MeasurementSample(
        model=label or "custom",
        index=frame.index.to_numpy(copy=True),
        target_names=tuple(flow.target_transform.target_names),
        samples=samples,
    )


__all__ = [
    "MeasurementSample",
    "ResponsePrediction",
    "ResponsePredictor",
    "predict_blend_response",
    "sample_measurement",
]
