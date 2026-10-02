"""Exact output-conditioned disk response on arbitrary catalogue atom draws.

The sparse pair table is fixed at zero shear. Only the observed radius/flux
enters the tree prediction; the resulting coefficient is reusable across all
shear-stencil nodes. Pooling uses exact tree split cells, never interpolation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .output_conditioned_response import TRUTH_FEATURES


class CatalogueDiskResponse:
    """CSR neighbour features aligned to compact prior atom rows."""

    def __init__(self, indptr, features, model):
        self.indptr = np.asarray(indptr)
        self.features = np.asarray(features)
        self.model = model
        if (self.indptr.ndim != 1 or len(self.indptr) < 2
                or not np.issubdtype(self.indptr.dtype, np.integer)
                or self.indptr[0] != 0 or np.any(np.diff(self.indptr) < 0)
                or self.features.shape != (self.indptr[-1], len(TRUTH_FEATURES))):
            raise ValueError("aligned monotone CSR response pairs required")
        if model.response_coordinate != "disk_velocity":
            raise ValueError("disk-velocity response model required")
        self.n_atoms = len(self.indptr) - 1

    @classmethod
    def from_pairs(cls, ids, pairs, model):
        ids = pd.Index(ids)
        if not ids.is_unique or pairs.duplicated(["index_input_p", "index_input_s"]).any():
            raise ValueError("unique atom and pair identities required")
        rows = ids.get_indexer(pairs.index_input_p)
        if (rows < 0).any():
            raise ValueError("response pair primary absent from prior")
        if (pairs.index_input_p == pairs.index_input_s).any():
            raise ValueError("self-pairs are not neighbours")
        order = np.argsort(rows, kind="stable")
        features = pairs.loc[:, TRUTH_FEATURES].to_numpy(np.float32)[order]
        if not np.isfinite(features).all():
            raise ValueError("finite response features required")
        indptr = np.r_[0, np.cumsum(np.bincount(rows, minlength=len(ids)))]
        return cls(indptr, features, model)

    def predict(self, atoms, radius, flux, *, pair_batch_size=200000, cell_batch_size=4096):
        """Return one exact pair sum per aligned atom/radius/flux triple."""
        atoms = np.asarray(atoms)
        radius, flux = np.asarray(radius), np.asarray(flux)
        if (not np.issubdtype(atoms.dtype, np.integer) or atoms.shape != radius.shape
                or atoms.shape != flux.shape or np.any(atoms < 0)
                or np.any(atoms >= self.n_atoms) or pair_batch_size < 1 or cell_batch_size < 1):
            raise ValueError("valid aligned atoms/radius/flux and positive batch sizes required")
        shape = atoms.shape
        atoms, radius, flux = atoms.ravel(), radius.ravel(), flux.ravel()
        outputs = self.model.transformed_outputs(radius, flux).astype(np.float32)
        edges_r, edges_m = self.model.output_thresholds
        n_m = len(edges_m) + 1
        n_cells = (len(edges_r) + 1) * n_m
        if self.n_atoms > np.iinfo(np.int64).max // n_cells:
            raise ValueError("atom/output-cell key exceeds int64")
        cells = np.searchsorted(edges_r, outputs[:, 0], side="right") * n_m
        cells += np.searchsorted(edges_m, outputs[:, 1], side="right")
        keys, representatives, inverse = np.unique(
            atoms.astype(np.int64) * n_cells + cells, return_index=True, return_inverse=True
        )
        sums = np.zeros(len(keys), dtype=np.float64)
        n_predictions = 0
        for start in range(0, len(keys), cell_batch_size):
            stop = min(start + cell_batch_size, len(keys))
            selected = keys[start:stop] // n_cells
            counts = self.indptr[selected + 1] - self.indptr[selected]
            cell_rows = np.repeat(np.arange(stop - start), counts)
            offsets = np.arange(len(cell_rows)) - np.repeat(np.cumsum(counts) - counts, counts)
            pair_rows = np.repeat(self.indptr[selected], counts) + offsets
            n_predictions += len(pair_rows)
            for lo in range(0, len(pair_rows), pair_batch_size):
                hi = lo + pair_batch_size
                cell = cell_rows[lo:hi]
                rep = representatives[start + cell]
                frame = pd.DataFrame(self.features[pair_rows[lo:hi]], columns=TRUTH_FEATURES)
                prediction = self.model.predict_pairs(frame, radius[rep], flux[rep])
                sums[start:stop] += np.bincount(cell, weights=prediction, minlength=stop-start)
        return sums[inverse].reshape(shape), {
            "atom_output_terms": len(atoms), "unique_atom_output_cells": len(keys),
            "predicted_pair_cells": n_predictions,
        }
