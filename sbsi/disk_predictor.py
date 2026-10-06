"""Response prediction with the V3.9-like disk models.

The V3.9-like preset has one physical-disk measurement flow, one usable-detection
classifier, and an output-conditioned blending emulator that acts on every flow
draw.  This module wraps the three into the same few calls as the V3 API:

    inputs, pairs = prepare_disk_inputs(catalogue, conditions)
    predictor = DiskResponsePredictor(get_model("V3.9-like"))
    prediction = predictor.predict(inputs, pairs)
    prediction.summary()

``predict`` reproduces the ConstGold evaluation of the published V3.9-like bias
(``scripts/evaluate_smooth_classifier_joint.py`` in the frozen F8 checkout):
the same sheared inputs, antithetic common random draws, per-draw Mobius
blending shift, output cuts and per-block seeds.  Calling it with
``random_seed = 7301 + 10000019 * case`` on the same GPU model replays the saved
per-galaxy rows exactly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional

import numpy as np
import pandas as pd

from .catalogue import load_catalogue
from .coordinates import ellipticity_from_axis_ratio_angle
from .crowding import V39_FEATURES, physical_close_crowding
from .forward_catalogue import make_pair_catalogue, rescale_emulator_pairs
from .models import ModelPaths, load_detection_classifier, load_disk_response
from .response import MeasurementSample
from .shear_map import apply_shear_to_ellipticity

LEGS = ("plus", "minus")
# Cuts on each generated measurement, in the flow's output units (pixels of
# 0.2 arcsec, flux at zero point 30): "main" is FLUX_RADIUS > 0.6 arcsec and
# MAG_AUTO < 25.8, "r075" is FLUX_RADIUS >= 0.75 arcsec and MAG_AUTO < 25.8.
CUTS = ("uncut", "main", "r075")
FLUX_MINIMUM = 10 ** (-0.4 * (25.8 - 30))
BLOCK_SEED_STRIDE = 1000003


def prepare_disk_inputs(
    catalogue,
    conditions: Mapping[str, float],
    *,
    r_max_arcsec: float = 10.0,
    k: int = 20,
    workers: int = 1,
):
    """Build the eight flow/classifier inputs and the blending pair table.

    ``catalogue`` is one independent scene with the input-catalogue columns
    (``index``, ``RA``, ``DEC``, ``r``, ``Re``, ``axis_ratio``,
    ``position_angle``, ``sersic_n``).  Returns ``(inputs, pairs)``: ``inputs``
    has the ``V39_FEATURES`` columns, indexed by the catalogue ``index`` in
    increasing order, and stored at float32 precision as in training; ``pairs``
    lists up to ``k - 1`` neighbours within ``r_max_arcsec`` per galaxy, scaled
    by ``conditions`` and sorted by primary.  Every galaxy is a primary; one
    without neighbours gets zero blending response.
    """

    cat = load_catalogue(catalogue).sort_values("index", kind="stable").reset_index(drop=True)
    ids = cat["index"].to_numpy(np.int64)
    if np.any(np.diff(ids) == 0):
        raise ValueError("catalogue index values must be unique")
    q = cat["axis_ratio"].to_numpy(float)
    e1, e2 = ellipticity_from_axis_ratio_angle(q, cat["position_angle"].to_numpy(float))
    inputs = pd.DataFrame({
        "e1_input_p": e1,
        "e2_input_p": e2,
        "sersic_n_input_p": cat["sersic_n"].to_numpy(float),
        "r_input_p": cat["r"].to_numpy(float),
        "circularized_Re_input_p": cat["Re"].to_numpy(float) * np.sqrt(q),
    }, index=pd.Index(ids, name="index"))
    inputs[["nbr_close_050", "nbr_close_100", "nbr_close_200"]] = physical_close_crowding(cat, workers=workers)
    inputs = inputs[list(V39_FEATURES)].astype(np.float32).astype(np.float64)
    pairs = rescale_emulator_pairs(make_pair_catalogue(cat, r_max_arcsec=r_max_arcsec, k=k), conditions)
    pairs = pairs.sort_values("index_input_p", kind="stable").reset_index(drop=True)
    return inputs, pairs


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
    """Flow, usable classifier and per-draw blending of a disk model preset."""

    def __init__(self, models: ModelPaths, device: Optional[str] = None, *, nthread: int = 1):
        import torch

        from .measurement_model import load_measurement_model

        if models.response_backend != "disk_response_moment" or len(models.flow_checkpoints) != 1:
            raise ValueError("a disk model preset with one flow checkpoint is required")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.label = models.name
        self.flow = load_measurement_model(str(models.flow_checkpoints[0]), device=self.device)
        self.flow.model.eval()
        if tuple(self.flow.condition_preprocessor.feature_names) != V39_FEATURES:
            raise ValueError("the flow does not take the V3.9 inputs")
        self.classifier = load_detection_classifier(models, device=self.device)
        self.disk = load_disk_response(models, device=self.device, nthread=nthread)

    def predict(self, inputs: pd.DataFrame, pairs: pd.DataFrame, *, shear: float = 0.02,
                n_draws: int = 64, batch_size: int = 1024, random_seed: int = 7301) -> DiskResponsePrediction:
        """Draw measurements at ``+shear`` and ``-shear`` along g1 and average them.

        Both legs use the same antithetic draws (common random numbers).  Each
        draw's shape is moved by the blending emulator's response to the
        applied shear; the move is the blend part, the unmoved shape the flow
        part.  Block ``b`` of ``batch_size`` galaxies uses seed
        ``random_seed + 1000003 * b``.
        """

        import torch

        from .disk_response_transport import disk_response_transport
        from .output_conditioned_response import predict_scene_draw_responses

        ids = inputs.index.to_numpy(np.int64)
        pair_ids = pairs["index_input_p"].to_numpy(np.int64)
        if len(ids) == 0 or np.any(np.diff(ids) <= 0):
            raise ValueError("inputs must be indexed by unique increasing ids")
        if np.any(np.diff(pair_ids) < 0):
            raise ValueError("pairs must be sorted by index_input_p")
        if not 0 < shear < 1:
            raise ValueError("shear must lie in (0, 1)")
        legs = [sheared_inputs(inputs, sign * shear) for sign in (1, -1)]
        contexts = [self.flow.context_tensor(leg) for leg in legs]
        p_usable = np.stack([self.classifier.predict_proba(leg) for leg in legs]).astype(np.float64)
        n = len(ids)
        mass = np.zeros((2, n, len(CUTS)))
        flow = np.zeros((2, n, len(CUTS), 2))
        blend = np.zeros((2, n, len(CUTS), 2))
        with torch.no_grad():
            for block, start in enumerate(range(0, n, batch_size)):
                stop = min(n, start + batch_size)
                draws = paired_draws(self.flow.model, [c[start:stop] for c in contexts], n_draws,
                                     random_seed + BLOCK_SEED_STRIDE * block)
                lo = np.searchsorted(pair_ids, ids[start], side="left")
                hi = np.searchsorted(pair_ids, ids[stop - 1], side="right")
                for leg, (sign, values) in enumerate(zip((1, -1), draws)):
                    response, _ = predict_scene_draw_responses(
                        self.disk, pairs.iloc[lo:hi], ids[start:stop], values.cpu().numpy())
                    vector = torch.tensor([sign * shear, 0.0], dtype=torch.float64, device=values.device)
                    velocity = torch.as_tensor(response, dtype=torch.float64, device=values.device)[..., None] * vector
                    moved, _ = disk_response_transport(values[..., :2], velocity)
                    moments = selected_draw_moments(values, moved - values[..., :2])
                    for out, x in zip((mass, flow, blend), moments):
                        out[leg, start:stop] = x.cpu().numpy()
        return DiskResponsePrediction(self.label, ids, float(shear), mass, flow, blend, p_usable)

    def sample(self, inputs: pd.DataFrame, *, n_samples: int = 256, random_seed: int = 12345) -> MeasurementSample:
        """Flow draws of the measured ``(g1, g2, FLUX_RADIUS px, flux at ZP 30)``.

        No shear and no blending shift are applied: this is the flow's direct
        distribution of measurements given each galaxy's inputs.
        """

        import torch

        torch.manual_seed(random_seed)
        samples = self.flow.sample(inputs, n_samples=n_samples)
        return MeasurementSample(
            model=self.label,
            index=inputs.index.to_numpy(copy=True),
            target_names=tuple(self.flow.target_transform.target_names),
            samples=samples,
        )


__all__ = [
    "CUTS",
    "DiskResponsePrediction",
    "DiskResponsePredictor",
    "prepare_disk_inputs",
    "sheared_inputs",
]
