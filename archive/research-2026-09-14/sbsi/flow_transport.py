"""Paired latent transport for flow diagnostics and simulation supervision.

This chooses the coupling induced by the flow's latent coordinates. It is not
an assertion that four latent values identify the simulator's pixel noise.
"""

import numpy as np
import torch

from .measurement_model import (
    ConditionalAffineFlow,
    ConditionalMeanFlow,
    ConditionalMeanFlowRA,
)


def paired_flow_transport(model, values, context_from, context_to):
    """Transport standardized measurements using a shared inferred latent.

    Unlike independent draws, inverse-then-forward preserves the source
    realization. Gradients traverse both legs when used for training. No
    residual offset or response scaling is estimated here.
    """
    if values.ndim != 2 or values.shape[1] != model.target_dim:
        raise ValueError("values must have shape (batch, target_dim)")
    expected = (len(values), model.context_dim)
    if tuple(context_from.shape) != expected or tuple(context_to.shape) != expected:
        raise ValueError("source and destination contexts must align with values")
    if isinstance(model, ConditionalMeanFlowRA):
        raise TypeError("realisation-aware mean heads require their full inverse")
    if isinstance(model, ConditionalMeanFlow):
        latent, _ = model.flow.inverse(
            values - model._mu(context_from), model._flow_ctx(context_from)
        )
        result, _ = model.flow.forward(latent, model._flow_ctx(context_to))
        return result + model._mu(context_to)
    if isinstance(model, ConditionalAffineFlow):
        latent, _ = model.inverse(values, context_from)
        return model.forward(latent, context_to)[0]
    raise TypeError("paired transport supports affine and plain mean-head flows")


def matched_transport_indices(frame):
    """Match two selected catalogues within an already case-disjoint split."""
    keys = ["case", "input_index"]
    f = frame.reset_index(drop=True)
    if set(f["__catalogue_index"].unique()) != {0, 1}:
        raise ValueError("paired transport requires exactly two catalogues")
    left = f.loc[f["__catalogue_index"] == 0, keys].copy()
    right = f.loc[f["__catalogue_index"] == 1, keys].copy()
    left["source_row"] = left.index
    right["destination_row"] = right.index
    joined = left.merge(right, on=keys, how="inner", validate="one_to_one")
    report = {"source_rows": len(left), "destination_rows": len(right),
              "matched_rows": len(joined), "unmatched_source": len(left)-len(joined),
              "unmatched_destination": len(right)-len(joined)}
    if joined.empty:
        raise ValueError("no paired transport rows")
    return joined[["source_row", "destination_row"]].to_numpy(np.int64), report


class PairedTransportBatches:
    """Index existing NLL tensors; unmatched rows remain in the NLL loader."""

    def __init__(self, loader, pairs, shuffle):
        tensors = loader.tensors if hasattr(loader, "tensors") else loader.dataset.tensors
        self.values, self.context = tensors[:2]
        self.pairs = torch.as_tensor(pairs, dtype=torch.long, device=self.values.device)
        self.batch_size = loader.batch_size
        self.shuffle = shuffle

    def __iter__(self):
        order = (torch.randperm(len(self.pairs), device=self.pairs.device)
                 if self.shuffle else torch.arange(len(self.pairs), device=self.pairs.device))
        for start in range(0, len(order), self.batch_size):
            pair = self.pairs[order[start:start+self.batch_size]]
            a, b = pair[:,0], pair[:,1]
            yield self.values[a], self.context[a], self.values[b], self.context[b]


def paired_transport_loss(model, batch, shear):
    """Mean squared standardized response error on matched forward pairs."""
    if shear <= 0:
        raise ValueError("paired shear must be positive")
    x0, c0, x1, c1 = batch
    predicted = paired_flow_transport(model, x0, c0, c1)
    return ((predicted-x1)/shear).square().mean()
