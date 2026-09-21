"""Persisted, hash-bound V3.6 catalogue preparation (not a legacy scene store)."""
from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from .catalogue_disk_cache import DiskCatalogueModelCache
from .catalogue_disk_response import CatalogueDiskResponse
from .catalogue_likelihood import CatalogueModelCache, CatalogueModelView
from .catalogue_sampling import ProposalCoordinateTable
from .crowding import FEATURES, FLOW_FEATURES
from .shear_map import apply_shear_to_ellipticity


def file_hash(path):
    digest = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")


def stencil(center, h):
    if np.shape(center) != (2,) or not np.isfinite(center).all() or not np.isfinite(h) or h <= 0:
        raise ValueError("finite two-component center and positive step required")
    return [CatalogueModelCache._key(0., 0.)] + [
        CatalogueModelCache._key(center[0]+h*dx, center[1]+h*dy)
        for dx, dy in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1),
                       (1, 1), (1, -1), (-1, 1), (-1, -1))]


# A uniform subset carries one scalar weight for every atom.  A stratified
# subset over-samples a declared stratum and carries the inverse-probability
# weight of each atom in a per-shard file instead, so the represented
# population is the same one either way.
SUBSET_SAMPLING = {"uncut_disk_prior_subset_v1": "uniform_without_replacement",
                   "uncut_disk_prior_subset_v2": "uniform_plus_certain_stratum"}


def load_subset_manifest(path):
    path = Path(path)
    manifest = json.loads(path.read_text())
    shards = manifest["shards"]
    stratified = manifest["format"] == "uncut_disk_prior_subset_v2"
    if (manifest["status"] != "complete"
            or manifest["sampling"] != SUBSET_SAMPLING.get(manifest["format"])
            or manifest["truth_cuts"] is not None
            or sum(s["n_rows"] for s in shards) != manifest["n_rows"]
            or len({s["source_shard"] for s in shards}) != len(shards)
            or manifest["prior_weight"] != (None if stratified else 1./manifest["n_rows"])):
        raise ValueError("invalid complete uncut subset manifest")
    return manifest


def subset_weights(manifest, index=None):
    """Prior weight of every atom, in global subset order.

    A uniform subset reports its one scalar; a stratified subset reports the
    normalized inverse-probability weights written beside each shard.  Over the
    whole subset these sum to one, exactly as the uniform 1/n weights do.
    """
    receipts = manifest["shards"] if index is None else [manifest["shards"][index]]
    if manifest["prior_weight"] is not None:
        return np.full(sum(r["n_rows"] for r in receipts), manifest["prior_weight"])
    parts = []
    for receipt in receipts:
        weights = np.load(Path(receipt["root"]) / "prior_weight.npy").astype(float)
        if (weights.shape != (receipt["n_rows"],) or not np.isfinite(weights).all()
                or np.any(weights <= 0)):
            raise ValueError("positive finite prior weight per atom required")
        parts.append(weights)
    weights = np.concatenate(parts) if parts else np.empty(0)
    if index is None and abs(weights.sum()-1.) > 1e-9:
        raise ValueError("stratified prior weights must sum to one")
    return weights


def load_source_shard(manifest, index, *, detector, model, limit=None):
    receipt = manifest["shards"][index]
    root = Path(receipt["root"])
    if json.loads((root / "manifest.json").read_text()) != receipt:
        raise ValueError("subset shard receipt differs from parent manifest")
    for name, expected in receipt["output_sha256"].items():
        if file_hash(root / name) != expected:
            raise ValueError(f"subset artifact hash differs: {root/name}")
    n = receipt["n_rows"] if limit is None else min(limit, receipt["n_rows"])
    if n < 1:
        raise ValueError("nonempty source shard required")
    zero = pd.read_parquet(root / "flow_zero.parquet").iloc[:n].copy()
    ids = np.load(root / "source_atom_ids.npy", mmap_mode="r")[:n]
    ptr = np.load(root / "pair_indptr.npy", mmap_mode="r")[:n+1]
    pairs = np.load(root / "pair_features.npy", mmap_mode="r")[:ptr[-1]]
    secondary = np.load(root / "pair_secondary.npy", mmap_mode="r")[:ptr[-1]]
    if limit is not None and manifest["prior_weight"] is None:
        # Truncating a stratified shard would drop weight the assembler cannot
        # account for, so a pilot is only defined over the uniform subset.
        raise ValueError("a stratified subset cannot be truncated to a pilot")
    prior = SimpleNamespace(galaxies=zero, weights=subset_weights(manifest, index)[:n])
    cache = DiskCatalogueModelCache(prior, detector=detector, conditions=manifest["conditions"],
        zero_flow=zero, pair_indptr=ptr, pair_features=pairs, pair_secondary=secondary,
        source_atom_ids=ids)
    return cache, CatalogueDiskResponse(ptr, pairs, model)


class FrozenDiskCache(CatalogueModelCache):
    """State probabilities are frozen; missing shear nodes fail closed."""
    def __init__(self, zero, probabilities, conditions, weights=None):
        if tuple(zero.columns) != FLOW_FEATURES or not len(zero):
            raise ValueError("eight-column nonempty disk context required")
        zero = zero.copy(deep=False)
        zero.index = pd.RangeIndex(len(zero), name="primary_row")
        if weights is None:
            weights = np.full(len(zero), 1./len(zero))
        else:
            weights = np.asarray(weights, dtype=float)
            if (weights.shape != (len(zero),) or not np.isfinite(weights).all()
                    or np.any(weights <= 0)):
                raise ValueError("positive finite prior weight per atom required")
        prior = SimpleNamespace(galaxies=zero, weights=weights)
        super().__init__(prior, detector=SimpleNamespace(preprocessor=SimpleNamespace(feature_names=FEATURES)),
            conditions=conditions, detection_radius_arcsec=10., flow_neighbour_radius_arcsec=7.,
            crowding_radii_arcsec=(3., 7.), flow_features=FLOW_FEATURES, detection_features=FEATURES)
        self.zero_flow = zero
        self.probabilities = {}
        for point, probability in probabilities.items():
            values = np.asarray(probability)
            key = self._key(*point)
            if (key in self.probabilities or values.shape != (len(zero),)
                    or not np.isfinite(values).all() or np.any((values < 0) | (values > 1))):
                raise ValueError("unique shear keys and aligned probabilities in [0,1] required")
            self.probabilities[key] = values
        if (0., 0.) not in self.probabilities:
            raise ValueError("zero-shear proposal probabilities required")

    def get(self, g1, g2):
        key = self._key(g1, g2)
        if key not in self.probabilities:
            raise ValueError(f"unprepared disk shear node {key}")
        if key not in self._views:
            flow = self.zero_flow.copy(deep=False)
            flow["e1_input_p"], flow["e2_input_p"] = apply_shear_to_ellipticity(
                self.zero_flow.e1_input_p.to_numpy(), self.zero_flow.e2_input_p.to_numpy(), *key)
            self._views[key] = CatalogueModelView(*key, flow, pd.DataFrame(index=flow.index),
                self.probabilities[key], np.broadcast_to(np.zeros(2), (len(flow), 2)))
        return self._views[key]

    def validate_model_features(self, flow_model):
        if tuple(flow_model.condition_preprocessor.feature_names) != FLOW_FEATURES:
            raise ValueError("prepared disk context and flow features differ")

    def save(self, *args, **kwargs):
        raise NotImplementedError("use the disk preparation manifest, not legacy model persistence")


class ShardedDiskResponse:
    """Keep large CSR pair features memory-mapped in their original shards."""
    def __init__(self, responses):
        if not responses:
            raise ValueError("at least one response shard required")
        self.responses = tuple(responses)
        self.offsets = np.r_[0, np.cumsum([r.n_atoms for r in responses])]
        self.n_atoms = int(self.offsets[-1])

    def predict(self, atoms, radius, flux, **kwargs):
        atoms, radius, flux = np.asarray(atoms), np.asarray(radius), np.asarray(flux)
        if (not np.issubdtype(atoms.dtype, np.integer) or atoms.shape != radius.shape
                or atoms.shape != flux.shape or np.any(atoms < 0) or np.any(atoms >= self.n_atoms)):
            raise ValueError("valid aligned global atom/output arrays required")
        result = np.empty(atoms.shape, dtype=float)
        count = 0
        for i, response in enumerate(self.responses):
            mask = (atoms >= self.offsets[i]) & (atoms < self.offsets[i+1])
            if mask.any():
                result[mask], _ = response.predict(atoms[mask]-self.offsets[i], radius[mask], flux[mask], **kwargs)
                count += 1
        return result, {"response_shards_used": count}


def coordinate_table(values, dispersion, target_names):
    """Apply the same global robust scale and 1%-dispersion floor as from_flow."""
    values, dispersion = np.asarray(values), np.asarray(dispersion)
    if (values.ndim != 2 or values.shape != dispersion.shape or not len(values)
            or not np.isfinite(values).all()):
        raise ValueError("aligned finite coordinate summaries required")
    center = np.median(values, axis=0)
    q25, q75 = np.percentile(values, [25, 75], axis=0)
    scale = (q75-q25)/1.3489795003921634
    scale = np.where(np.isfinite(scale) & (scale > 0), scale, np.std(values, axis=0))
    scale = np.where(np.isfinite(scale) & (scale > 0), scale, 1.)
    good = np.isfinite(dispersion) & (dispersion > 0)
    floor = np.nanpercentile(np.where(good, dispersion, np.nan), 1., axis=0)
    floor = np.where(np.isfinite(floor) & (floor > 0), floor, np.maximum(1e-3*scale, np.finfo(float).eps))
    dispersion = np.maximum(np.where(good, dispersion, floor), floor)
    return ProposalCoordinateTable(values, tuple(target_names), center, scale,
        dispersion=dispersion, statistic="mean", dispersion_statistic="std", n_flow_samples=128)
