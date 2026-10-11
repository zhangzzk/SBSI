"""The V3.9-like blending model: flow draws moved by a per-draw disk emulator.

:func:`sbsi.load_emulator` returns a :class:`DiskResponsePredictor`.  It holds
the measurement flow, the usable-detection classifier and the
output-conditioned blending emulator together, because the blending response
of a galaxy depends on each simulated measurement of it (its measured size and
flux), not only on its true properties.

``predict`` reproduces the development constant-shear evaluation behind the
published V3.9-like bias: the same sheared inputs, antithetic common random draws, per-draw Mobius
blending shift, output cuts and per-block seeds.  Calling it with
``random_seed = 7301 + 10000019 * case`` on the same GPU model replays the saved
per-galaxy rows exactly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional

import numpy as np
import pandas as pd

from .crowding import V39_FEATURES
from .models import ModelPaths, load_detection_classifier, load_disk_response
from .shear_map import apply_shear_to_ellipticity

# Cuts on each generated measurement, in the flow's output units (pixels of
# 0.2 arcsec, flux at zero point 30): "main" is FLUX_RADIUS > 0.6 arcsec and
# MAG_AUTO < 25.8, "r075" is FLUX_RADIUS >= 0.75 arcsec and MAG_AUTO < 25.8.
CUTS = ("uncut", "main", "r075")
FLUX_MINIMUM = 10 ** (-0.4 * (25.8 - 30))
BLOCK_SEED_STRIDE = 1000003
RANDOM_SEED = 7301
# Pair search of the emulator's training tables: every neighbour within
# 10 arcsec, at most k - 1 = 19 per primary.
PAIR_RADIUS_ARCSEC = 10.0
PAIR_NEIGHBOURS = 20


def load_flow(checkpoint, device):
    """Load the V3.9-like measurement flow and check its inputs."""

    from .measurement_model import load_measurement_model

    flow = load_measurement_model(str(checkpoint), device=device)
    flow.model.eval()
    if tuple(flow.condition_preprocessor.feature_names) != V39_FEATURES:
        raise ValueError("the flow does not take the V3.9 inputs")
    return flow


def sheared_inputs(inputs: pd.DataFrame, g1: float, g2: float = 0.0) -> pd.DataFrame:
    """Return ``inputs`` with the intrinsic shapes sheared by ``(g1, g2)``."""

    out = inputs.copy()
    out["e1_input_p"], out["e2_input_p"] = apply_shear_to_ellipticity(
        inputs["e1_input_p"].to_numpy(), inputs["e2_input_p"].to_numpy(), g1, g2)
    return out


def output_cuts(values):
    """Boolean ``(..., len(CUTS))`` mask of draws passing each cut."""

    import torch

    radius, flux = values[..., 2], values[..., 3]
    return torch.stack((
        torch.ones_like(radius, dtype=torch.bool),
        (radius > 3.0) & (flux > FLUX_MINIMUM),
        (radius >= 3.75) & (flux > FLUX_MINIMUM),
    ), dim=-1)


def paired_draws(model, contexts, draws, seed):
    """Physical flow draws sharing antithetic latents ``z, -z`` across contexts."""

    import torch

    if draws < 2 or draws % 2:
        raise ValueError("an even draw count >= 2 is required")
    batch = len(contexts[0])
    generator = torch.Generator(device=contexts[0].device).manual_seed(int(seed))
    half = torch.randn(batch, draws // 2, model.target_dim, dtype=contexts[0].dtype,
                       device=contexts[0].device, generator=generator)
    base = torch.cat((half, -half), dim=1).reshape(batch * draws, model.target_dim)
    out = []
    for context in contexts:
        expanded = context[:, None, :].expand(batch, draws, model.context_dim)
        values, _ = model.forward_physical(base, expanded.reshape(batch * draws, model.context_dim))
        out.append(values.reshape(batch, draws, model.target_dim))
    return out


def selected_draw_moments(values, blend_shift):
    """Per galaxy and cut: passing fraction, mean kept shape, mean kept blend shift."""

    keep = output_cuts(values).to(values.dtype)
    mass = keep.mean(1)
    flow = (keep[..., None] * values[..., None, :2]).mean(1)
    blend = (keep[..., None] * blend_shift[:, :, None, :]).mean(1)
    return mass, flow, blend


def draw_moments(flow, inputs, *, shear, n_draws, batch_size, random_seed, disk=None, pairs=None):
    """Per-galaxy draw averages ``(mass, flow, blend)`` in the ``+shear`` and ``-shear`` legs.

    Both legs use the same antithetic draws (common random numbers).  With a
    ``disk`` emulator, each draw's shape is moved by the emulator's response to
    the applied shear; the move is the blend part, the unmoved shape the flow
    part.  Without one the blend part is zero.  Block ``b`` of ``batch_size``
    galaxies uses seed ``random_seed + 1000003 * b``.  ``pairs`` is keyed by
    ``primary_row`` and ``secondary_row``, matching the ``inputs`` index.
    """

    import torch

    from .disk_response_transport import disk_response_transport
    from .output_conditioned_response import TRUTH_FEATURES, predict_scene_draw_responses

    ids = inputs.index.to_numpy(np.int64)
    if len(ids) == 0 or np.any(np.diff(ids) <= 0):
        raise ValueError("inputs must be indexed by unique increasing ids")
    if not 0 < shear < 1:
        raise ValueError("shear must lie in (0, 1)")
    if disk is not None:
        pairs = pairs.sort_values("primary_row", kind="stable")
        pair_ids = pairs["primary_row"].to_numpy(np.int64)
        keyed = pairs[list(TRUTH_FEATURES)].assign(index_input_p=pair_ids,
                                                    index_input_s=pairs["secondary_row"].to_numpy(np.int64))
    legs = [sheared_inputs(inputs[list(V39_FEATURES)], sign * shear) for sign in (1, -1)]
    contexts = [flow.context_tensor(leg) for leg in legs]
    n = len(ids)
    mass = np.zeros((2, n, len(CUTS)))
    kept = np.zeros((2, n, len(CUTS), 2))
    blend = np.zeros((2, n, len(CUTS), 2))
    with torch.no_grad():
        for block, start in enumerate(range(0, n, batch_size)):
            stop = min(n, start + batch_size)
            draws = paired_draws(flow.model, [c[start:stop] for c in contexts], n_draws,
                                 random_seed + BLOCK_SEED_STRIDE * block)
            for leg, (sign, values) in enumerate(zip((1, -1), draws)):
                shift = torch.zeros_like(values[..., :2])
                if disk is not None:
                    lo = np.searchsorted(pair_ids, ids[start], side="left")
                    hi = np.searchsorted(pair_ids, ids[stop - 1], side="right")
                    response, _ = predict_scene_draw_responses(
                        disk, keyed.iloc[lo:hi], ids[start:stop], values.cpu().numpy())
                    vector = torch.tensor([sign * shear, 0.0], dtype=torch.float64, device=values.device)
                    velocity = torch.as_tensor(response, dtype=torch.float64, device=values.device)[..., None] * vector
                    moved, _ = disk_response_transport(values[..., :2], velocity)
                    shift = moved - values[..., :2]
                moments = selected_draw_moments(values, shift)
                for out, x in zip((mass, kept, blend), moments):
                    out[leg, start:stop] = x.cpu().numpy()
    return mass, kept, blend


@dataclass(frozen=True)
class DiskResponsePrediction:
    """Per-galaxy draw averages in the ``+shear`` and ``-shear`` legs.

    ``mass`` has shape ``(2, n, len(CUTS))``: the fraction of draws passing each
    cut.  ``flow`` and ``blend`` have shape ``(2, n, len(CUTS), 2)``: the draw
    mean of the kept measured shape and of the kept blending shift (zero for
    failing draws).  ``p_usable`` ``(2, n)`` is the classifier probability.
    Leg 0 is ``+shear`` along g1, leg 1 is ``-shear``.
    """

    model: str
    index: np.ndarray
    shear: float
    mass: np.ndarray
    flow: np.ndarray
    blend: np.ndarray
    p_usable: np.ndarray

    def _weights(self, weight):
        if weight is None:
            return np.ones_like(self.p_usable)
        if isinstance(weight, str):
            if weight != "usable":
                raise ValueError("weight must be None, 'usable', or an array")
            return self.p_usable
        weight = np.broadcast_to(np.asarray(weight, dtype=float), self.p_usable.shape)
        return weight

    def response(self, cut="main", weight=None, counts=None):
        """Population ``(R_flow, R_blend)`` along g1, as ratios of sums per leg.

        ``R = (N+/D+ - N-/D-) / 2h`` with ``N = sum_i w_i * mean kept e1`` and
        ``D = sum_i w_i * passing fraction``.  ``weight`` is None (all 1),
        ``"usable"`` (classifier probability in each leg), or per-galaxy
        weights; ``counts`` optionally multiplies them (bootstrap resampling).
        """

        column = CUTS.index(cut)
        w = self._weights(weight)
        if counts is not None:
            w = w * counts
        denominator = (w * self.mass[:, :, column]).sum(1)
        parts = []
        for part in (self.flow, self.blend):
            ratio = (w * part[:, :, column, 0]).sum(1) / denominator
            parts.append(float((ratio[0] - ratio[1]) / (2 * self.shear)))
        return tuple(parts)

    def _bootstrap(self, cut, weight, n_bootstrap, random_seed):
        rng = np.random.default_rng(random_seed)
        n = len(self.index)
        return np.array([
            self.response(cut, weight, np.bincount(rng.integers(0, n, n), minlength=n))
            for _ in range(n_bootstrap)
        ])

    def summary(self, settings=None, *, n_bootstrap=500, random_seed=20261006):
        """Table of ``R_flow``, ``R_blend`` and ``R_model`` with 1-sigma errors.

        ``settings`` maps a row label to ``(cut, weight)``; the default rows are
        uncut, the main cut, and the main cut weighted by usable probability.
        Errors come from resampling galaxies and include the draw noise.
        """

        settings = settings or {
            "uncut": ("uncut", None),
            "main cut": ("main", None),
            "main cut x usable probability": ("main", "usable"),
        }
        rows = {}
        for label, (cut, weight) in settings.items():
            flow, blend = self.response(cut, weight)
            boot = self._bootstrap(cut, weight, n_bootstrap, random_seed)
            rows[label] = {
                "R_flow": flow, "R_flow_err": boot[:, 0].std(ddof=1),
                "R_blend": blend, "R_blend_err": boot[:, 1].std(ddof=1),
                "R_model": flow + blend, "R_model_err": boot.sum(1).std(ddof=1),
            }
        return pd.DataFrame.from_dict(rows, orient="index")

    def per_galaxy(self) -> pd.DataFrame:
        """Uncut per-galaxy responses (noisy with few draws) and mean usable probability."""

        flow = (self.flow[0, :, 0, 0] - self.flow[1, :, 0, 0]) / (2 * self.shear)
        blend = (self.blend[0, :, 0, 0] - self.blend[1, :, 0, 0]) / (2 * self.shear)
        return pd.DataFrame({
            "R_flow": flow, "R_blend": blend, "R_total": flow + blend,
            "P_usable": self.p_usable.mean(0),
        }, index=pd.Index(self.index, name="index"))

    def multiplicative_bias(self, simulation_response, cut="main", weight="usable",
                            *, n_bootstrap=500, random_seed=20261006):
        """Return ``(m, sigma_m)`` with ``m = R_sim / R_model - 1``.

        ``sigma_m`` propagates only the model's bootstrap error; add the
        simulation's own error separately.
        """

        model = sum(self.response(cut, weight))
        sigma = self._bootstrap(cut, weight, n_bootstrap, random_seed).sum(1).std(ddof=1)
        ratio = float(simulation_response) / model
        return ratio - 1.0, ratio * sigma / model


class DiskResponsePredictor:
    """Flow, usable classifier and per-draw blending emulator of the V3.9-like model."""

    r_max_arcsec = PAIR_RADIUS_ARCSEC
    k = PAIR_NEIGHBOURS

    def __init__(self, models: ModelPaths, device: Optional[str] = None, *,
                 conditions: Mapping[str, float], nthread: Optional[int] = None):
        import os

        import torch

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.label = models.name
        self.conditions = dict(conditions)
        self.flow = load_flow(models.flow_checkpoint, self.device)
        self.classifier = load_detection_classifier(models, device=self.device)
        nthread = nthread or (len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count())
        self.disk = load_disk_response(models, device=self.device, nthread=nthread)

    def predict(self, inputs: pd.DataFrame, pairs: pd.DataFrame, *, shear: float = 0.02,
                n_draws: int = 64, batch_size: int = 1024,
                random_seed: int = RANDOM_SEED) -> DiskResponsePrediction:
        """Draw measurements at ``+shear`` and ``-shear`` along g1 and average them.

        ``inputs`` holds the ``V39_FEATURES`` indexed by ``primary_row``
        (``prepare_forward_catalogue(...).flow_inputs``); ``pairs`` is the
        matching ``emulator_pairs``.  See :func:`draw_moments`.
        """

        mass, flow, blend = draw_moments(self.flow, inputs, shear=shear, n_draws=n_draws,
                                         batch_size=batch_size, random_seed=random_seed,
                                         disk=self.disk, pairs=pairs)
        legs = [sheared_inputs(inputs[list(V39_FEATURES)], sign * shear) for sign in (1, -1)]
        p_usable = np.stack([self.classifier.predict_proba(leg) for leg in legs]).astype(np.float64)
        return DiskResponsePrediction(self.label, inputs.index.to_numpy(np.int64), float(shear),
                                      mass, flow, blend, p_usable)

    def predict_detection(self, catalogue) -> pd.DataFrame:
        """Probability that each galaxy is detected with a usable measurement, at zero shear.

        Returns a ``detection_prob`` column indexed by catalogue row.
        """

        from .forward_catalogue import prepare_flow_inputs

        inputs = prepare_flow_inputs(catalogue)
        p = self.classifier.predict_proba(inputs[list(V39_FEATURES)]).astype(np.float64)
        return pd.DataFrame({"detection_prob": p}, index=inputs.index)


__all__ = [
    "CUTS",
    "DiskResponsePrediction",
    "DiskResponsePredictor",
    "draw_moments",
    "load_flow",
    "sheared_inputs",
]
