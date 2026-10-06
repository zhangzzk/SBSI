"""Explicit experimental response prediction conditioned on joint flow outputs.

This loader does not change any configured likelihood. When used with a flow,
radius and flux must be taken from the same generated draw, so selection and
response remain correlated. Baseline measured values are allowed for isolated
simulation validation only. Model training belongs to BlendEMU.
"""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import numpy as np

TRUTH_FEATURES = ("Re_input_p_scaled", "Re_input_s_scaled", "r_input_p_scaled", "r_input_s_scaled",
                  "sersic_n_input_p", "sersic_n_input_s", "distance_scaled")
OUTPUT_FEATURES = ("baseline_log_flux_radius", "baseline_mag_auto")


def output_features(radius, flux, zero_point=30.):
    radius, flux = np.asarray(radius, dtype=float), np.asarray(flux, dtype=float)
    if (radius.shape != flux.shape or radius.ndim != 1 or not np.isfinite(radius).all()
            or not np.isfinite(flux).all() or np.any(radius <= 0) or np.any(flux <= 0)):
        raise ValueError("aligned finite positive radius/flux outputs required")
    return np.column_stack((np.log(radius), zero_point - 2.5*np.log10(flux)))


class OutputConditionedResponse:
    """Load one explicitly named, hash-verified experimental response arm."""

    artifact_format = "experimental_output_conditioned_response_v1"
    response_coordinate = "physical_additive"

    def __init__(self, root, *, nthread=1, device="cpu"):
        import xgboost as xgb

        if device not in ("cpu", "cuda") or nthread < 1:
            raise ValueError("response device must be cpu/cuda with positive thread count")
        self.device, self.nthread = device, int(nthread)
        root = Path(root)
        self.metadata = json.loads((root / "metadata.json").read_text())
        m = self.metadata
        if (m.get("format") != self.artifact_format
                or m.get("arm") not in ("truth_only", "output_conditioned")):
            raise ValueError("unsupported experimental response artifact")
        self.conditioned = m["arm"] == "output_conditioned"
        self.features = TRUTH_FEATURES + (OUTPUT_FEATURES if self.conditioned else ())
        if (tuple(m["features"]) != self.features or m["response_shear"] != .2
                or not np.isfinite([m["y_mean"], m["y_std"]]).all() or m["y_std"] <= 0):
            raise ValueError("response feature, label or standardization mismatch")
        path = root / "model.json"
        if sha256(path.read_bytes()).hexdigest() != m["model_sha256"]:
            raise ValueError("response model hash mismatch")
        self.model = xgb.Booster(model_file=path)
        self.model.set_param({"device": device, "nthread": self.nthread})
        if self.model.feature_names != list(self.features):
            raise ValueError("serialized model feature order mismatch")
        # Exact global output-feature boundaries, not a fitted/interpolated
        # approximation. Between these thresholds every tree takes the same
        # output-feature branches for a fixed truth pair.
        thresholds = {name: set() for name in OUTPUT_FEATURES}
        def visit(node):
            if "leaf" in node:
                return
            if node["split"] in thresholds:
                thresholds[node["split"]].add(float(np.float32(node["split_condition"])))
            for child in node["children"]:
                visit(child)
        for tree in self.model.get_dump(dump_format="json"):
            visit(json.loads(tree))
        self.output_thresholds = tuple(np.asarray(sorted(thresholds[name]), dtype=np.float32)
                                       for name in OUTPUT_FEATURES)

    def transformed_outputs(self, radius, flux, *, allow_zero_limit=False):
        if not allow_zero_limit:
            return output_features(radius, flux)
        radius, flux = np.asarray(radius, dtype=float), np.asarray(flux, dtype=float)
        if (radius.shape != flux.shape or radius.ndim != 1 or not np.isfinite(radius).all()
                or not np.isfinite(flux).all() or np.any(radius < 0) or np.any(flux < 0)):
            raise ValueError("aligned finite nonnegative radius/flux outputs required")
        with np.errstate(divide="ignore"):
            features = np.column_stack((np.log(radius), 30.-2.5*np.log10(flux)))
        # Softplus may underflow to exactly zero in float32 flow tails. Each
        # finite tree has an exact one-sided limit there. Use a representative
        # feature beyond its outermost split, without altering physical draws,
        # selection, or any finite positive input.
        radius_edges, magnitude_edges = self.output_thresholds
        features[radius == 0, 0] = (np.nextafter(radius_edges[0], np.float32(-np.inf))
                                    if len(radius_edges) else 0.)
        features[flux == 0, 1] = magnitude_edges[-1] if len(magnitude_edges) else 0.
        return features

    def predict_pairs(self, pair_features, radius, flux, *, allow_zero_limit=False):
        x = pair_features[list(TRUTH_FEATURES)].to_numpy(np.float32)
        if self.conditioned:
            extra = self.transformed_outputs(radius, flux, allow_zero_limit=allow_zero_limit)
            if len(extra) != len(x):
                raise ValueError("output features do not align with pairs")
            x = np.column_stack((x, extra)).astype(np.float32)
        if not np.isfinite(x).all():
            raise ValueError("nonfinite response features")
        if self.device == "cuda":
            import xgboost as xgb

            # Ordinary DMatrix prediction avoids the virtual-memory API used
            # by external/quantile GPU training on the local virtual A40s.
            matrix = xgb.DMatrix(np.ascontiguousarray(x), feature_names=list(self.features), nthread=self.nthread)
            values = self.model.predict(matrix).astype(float)
        else:
            values = self.model.inplace_predict(np.ascontiguousarray(x)).astype(float)
        values = values * self.metadata["y_std"] + self.metadata["y_mean"]
        if not np.isfinite(values).all():
            raise ValueError("nonfinite response prediction")
        return values


class DiskResponseMoment(OutputConditionedResponse):
    """A separately fitted disk-transport parameter, never an additive R_blend."""

    artifact_format = "experimental_disk_response_moment_v1"
    response_coordinate = "disk_velocity"

    def __init__(self, root, **kwargs):
        super().__init__(root, **kwargs)
        if (not self.conditioned
                or self.metadata.get("velocity_convention") != "radial_tanh(pair_sum_b_times_external_shear)"
                or self.metadata.get("moment_objective") != "E[(s*b^2/2-z*b)/label_std^2]"):
            raise ValueError("disk response must use the fitted physical-moment transport convention")


def predict_scene_draw_responses(candidate, pairs, primary_ids, physical_draws, *, pair_batch_size=200000):
    """Sum pair responses for every joint draw, pooling exact tree cells.

The same radius/flux draw determines both response and subsequent selection.
Repeated draws occupying the same two-dimensional tree cell share one exact
prediction per neighbour. Isolated primaries receive an empty pair sum of
zero. No interpolation, fitted scaling or independent output marginal enters.
    """
    import pandas as pd

    ids = np.asarray(primary_ids, dtype=np.int64)
    values = np.asarray(physical_draws)
    if (values.ndim != 3 or values.shape[0] != len(ids) or values.shape[2] != 4
            or values.shape[1] < 1 or len(np.unique(ids)) != len(ids) or pair_batch_size < 1):
        raise ValueError("unique primary identities and aligned joint draws required")
    if pairs.duplicated(["index_input_p", "index_input_s"]).any():
        raise ValueError("duplicate response pair")
    locations = pd.Index(ids).get_indexer(pairs["index_input_p"])
    if (locations < 0).any():
        raise ValueError("response pair primary is absent from joint draws")
    if candidate.conditioned:
        features = candidate.transformed_outputs(values[..., 2].reshape(-1), values[..., 3].reshape(-1),
                                                 allow_zero_limit=True).astype(np.float32)
        radius_bins = np.searchsorted(candidate.output_thresholds[0], features[:, 0], side="right")
        magnitude_bins = np.searchsorted(candidate.output_thresholds[1], features[:, 1], side="right")
        n_mag = len(candidate.output_thresholds[1]) + 1
        n_cells = (len(candidate.output_thresholds[0]) + 1) * n_mag
        cells = radius_bins*n_mag + magnitude_bins
    else:
        cells = np.zeros(len(ids)*values.shape[1], dtype=np.int64)
        n_cells = 1
    keys = np.repeat(np.arange(len(ids)), values.shape[1])*n_cells + cells
    unique, representative, inverse = np.unique(keys, return_index=True, return_inverse=True)
    counts = np.bincount(unique // n_cells, minlength=len(ids))
    starts = np.cumsum(counts) - counts
    repetitions = counts[locations]
    pair_rows = np.repeat(np.arange(len(pairs)), repetitions)
    repeated_starts = np.repeat(starts[locations], repetitions)
    within = np.arange(len(pair_rows)) - np.repeat(np.cumsum(repetitions)-repetitions, repetitions)
    cell_indices = repeated_starts + within
    response = np.zeros(len(unique))
    flat = values.reshape(-1, 4)
    for start in range(0, len(pair_rows), pair_batch_size):
        stop = start + pair_batch_size
        targets = cell_indices[start:stop]
        output = flat[representative[targets]]
        prediction = candidate.predict_pairs(pairs.iloc[pair_rows[start:stop]], output[:, 2], output[:, 3],
                                             allow_zero_limit=True)
        response += np.bincount(targets, weights=prediction, minlength=len(unique))
    return response[inverse].reshape(values.shape[:2]), {
        "joint_draws": int(np.prod(values.shape[:2])), "unique_primary_output_cells": len(unique),
        "predicted_pair_cells": len(pair_rows), "unpooled_pair_draws": len(pairs)*values.shape[1],
        "zero_radius_draws": int((values[..., 2] == 0).sum()),
        "zero_flux_draws": int((values[..., 3] == 0).sum()),
    }
