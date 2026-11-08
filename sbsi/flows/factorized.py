"""Retained factorized photometry/shape transport; original parameter names preserved."""

import torch
from torch import nn
from sbsi.measurement_model import build_flow, ConditionalAffineFlow

FORMAT = "sbsi_experimental_factorized_frame_v1"


class PhotometryShapeFlow(nn.Module):
    """Triangular transport: photometry first, then conditional shape.

    The four coordinates keep their original ordering [shape2, photometry2].
    Both flows are on the original standardized unbounded coordinates.
    """

    def __init__(self, config):
        super().__init__()
        assert config["target_dim"] == 4 and config["n_flows"] == 10
        kw = {k: config[k] for k in ["hidden_dim", "n_layers", "scale_limit", "activation"]}
        self.photometry = ConditionalAffineFlow(2, config["context_dim"], n_flows=5, **kw)
        self.shape = ConditionalAffineFlow(2, config["context_dim"] + 2, n_flows=5, **kw)

    def forward(self, z, context):
        phot, lp = self.photometry.forward(z[:, 2:], context)
        shape, ls = self.shape.forward(z[:, :2], torch.cat([context, phot], -1))
        return torch.cat([shape, phot], -1), lp + ls

    def inverse(self, y, context):
        zp, lp = self.photometry.inverse(y[:, 2:], context)
        zs, ls = self.shape.inverse(y[:, :2], torch.cat([context, y[:, 2:]], -1))
        return torch.cat([zs, zp], -1), lp + ls


class FrameFlow(nn.Module):
    def __init__(self, config, condition, kind):
        super().__init__()
        if kind not in ["cartesian", "frame"]:
            raise ValueError(kind)
        self.kind = kind
        self.base = build_flow(config)
        self.base.base = PhotometryShapeFlow(config)
        self.target_dim = self.base.target_dim
        self.context_dim = self.base.context_dim
        self.register_buffer("input_means", torch.as_tensor(condition.means[:2], dtype=torch.float64))
        self.register_buffer("input_scales", torch.as_tensor(condition.scales[:2], dtype=torch.float64))

    def rotation(self, context):
        e = context[:, :2].double() * self.input_scales + self.input_means
        norm = torch.linalg.vector_norm(e, dim=-1)
        safe = norm.clamp_min(1e-8)
        c = torch.where(norm >= 1e-8, e[:, 0] / safe, torch.ones_like(norm))
        s = torch.where(norm >= 1e-8, e[:, 1] / safe, torch.zeros_like(norm))
        return c, s

    def rotate(self, e, context, inverse=False):
        c, s = self.rotation(context)
        if inverse:
            s = -s
        return torch.stack([c * e[:, 0] + s * e[:, 1], -s * e[:, 0] + c * e[:, 1]], -1)

    def log_prob(self, y, context):
        if self.kind == "cartesian":
            return self.base.log_prob(y, context)
        mean, scale = self.base.disk_shape_means, self.base.disk_shape_scales
        shape = self.rotate(y[:, :2] * scale + mean, context)
        local = torch.cat([(shape - mean) / scale, y[:, 2:]], -1)
        return self.base.log_prob(local, context)

    def forward_physical(self, z, context):
        physical, ld = self.base.forward_physical(z, context)
        if self.kind == "frame":
            physical = torch.cat([self.rotate(physical[:, :2], context, inverse=True), physical[:, 2:]], -1)
        return physical, ld


class ExtraInputLinear(nn.Module):
    """Keep the exact extra-column arithmetic and checkpoint parameter names."""

    def __init__(self, original, index):
        super().__init__()
        self.original = original
        self.index = int(index)
        self.extra_weight = nn.Parameter(torch.zeros(original.out_features))

    def forward(self, x):
        old = torch.cat([x[:, : self.index], x[:, self.index + 1 :]], -1)
        return self.original(old) + x[:, self.index : self.index + 1] * self.extra_weight


def augment(model):
    old_width = model.context_dim
    for flow in [model.base.base.photometry, model.base.base.shape]:
        for layer in flow.layers:
            layer.net[0] = ExtraInputLinear(layer.net[0], 2 + old_width)
        flow.context_dim += 1
    model.context_dim = old_width + 1
    return model


class FactorizedCondition:
    """Original base preprocessing plus the explicitly enabled reference score."""

    score_feature = "neighbour_reference_coefficient"

    def __init__(self, state):
        import numpy as np
        from sbsi.selection_model import TabularPreprocessor

        self.base = TabularPreprocessor.from_state(state["base"])
        self.mean, self.scale = float(state["mean"]), float(state["scale"])
        self.use_score = bool(state["use_score"])
        if not np.isfinite([self.mean, self.scale]).all() or self.scale <= 0:
            raise ValueError("finite reference-score normalization required")
        self.feature_names = list(self.base.feature_names) + ([self.score_feature] if self.use_score else [])

    def transform_frame(self, frame):
        import numpy as np

        old = self.base.transform_frame(frame)
        if self.use_score:
            if self.score_feature not in frame:
                raise ValueError("multistage flow requires explicit reference neighbour coefficient")
            score = frame[self.score_feature].to_numpy(float)
            if not np.isfinite(score).all():
                raise ValueError("nonfinite reference neighbour coefficient")
            extra = (score - self.mean) / self.scale
        else:
            extra = np.zeros(len(old))
        return np.column_stack([old, extra]).astype(np.float32)

    def raw_tensor_from_frame(self, frame, **kwargs):
        raw = self.base.raw_tensor_from_frame(frame, **kwargs)
        if not self.use_score:
            return raw
        score = torch.as_tensor(frame[self.score_feature].to_numpy(float), device=raw.device, dtype=raw.dtype)
        if not torch.isfinite(score).all():
            raise ValueError("nonfinite reference neighbour coefficient")
        return torch.cat([raw, score[:, None]], -1)

    def transform_tensor(self, raw):
        old = self.base.transform_tensor(raw[:, : len(self.base.feature_names)])
        extra = (raw[:, -1:] - self.mean) / self.scale if self.use_score else torch.zeros_like(raw[:, :1])
        return torch.cat([old, extra.to(old.dtype)], -1)


def load_factorized_checkpoint(checkpoint, device="cpu"):
    """Load the unchanged Cartesian research state into the standard bundle API."""
    from sbsi.measurement_model import MeasurementModelBundle, TargetStandardizer, _validate_disk_contract
    from sbsi.selection_model import TabularPreprocessor

    kind = checkpoint["format"]
    if kind == "sbsi_experimental_neighbour_factorized_v1":
        condition = FactorizedCondition(checkpoint["condition"])
        frame = augment(FrameFlow(checkpoint["config"], condition.base, "cartesian"))
    elif kind == FORMAT and checkpoint["kind"] == "cartesian":
        condition = TabularPreprocessor.from_state(checkpoint["condition"])
        frame = FrameFlow(checkpoint["config"], condition, "cartesian")
    else:
        raise ValueError("unsupported factorized checkpoint format or shape frame")
    frame.load_state_dict(checkpoint["state_dict"], strict=True)
    target = TargetStandardizer.from_state(checkpoint["target"])
    # The Cartesian wrapper only delegates density/physical draws to this base.
    # Correct its declared width for sample_physical's context expansion.
    physical = frame.base
    physical.context_dim = frame.context_dim
    _validate_disk_contract(physical, target, checkpoint["config"])
    return MeasurementModelBundle(
        physical, condition, target, metadata=checkpoint.get("metadata", {}), device=device
    )
