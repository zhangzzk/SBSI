"""Unified forward model (EXPERIMENTAL / DIAGNOSTIC).

Prototype toward a single learned forward density

    P(detection | true scene) * p(measured_e1, e2, flux, Re | true scene, detected)

where "true scene" = primary true properties + a variable SET of true neighbour
properties.  From ONE model, all three shear responses become derivatives of one
representation:

  * shape response  = d<mu_e>/dgamma   (self + blend, via the mean head under an
                      analytic shear of the intrinsic shapes in the conditioning),
  * selection resp. = d P(s=1)/dgamma  (the detection head under the same shear).

The point of unifying is NOT convenience: a separate additive R_blend scalar is a
per-(mag,crowd) lookup that mis-allocates across the isolated/blended axis (the
non-closure diagnosed in WORKLOG 2026-07-17).  A per-object derivative conditioned on
the TRUE neighbour set cannot mis-allocate that way.  But the loss must pin the
SEPARABLE pieces against their own clean targets (isolated->self-truth,
detection->b_true, blended->residual blend); matching only the total response
re-creates the same allocation ambiguity inside one network.  See the trainer.

This module is additive and is NOT wired into the certified parameter-free
m = R_sim/(R_flow+R_blend)-1.  It is kept out of the public sbs_shear API until proven.
"""

from __future__ import annotations

import torch
from torch import nn

from .measurement_model import ConditionalMeanFlow, ConditionalMeanFlowRA
from .scene_model import DeepSetsConditioner


def _act(name):
    name = name.lower()
    if name == "silu":
        return nn.SiLU
    if name == "gelu":
        return nn.GELU
    if name == "tanh":
        return nn.Tanh
    raise ValueError(f"Unsupported activation {name!r}")


class SetConditionedForwardModel(nn.Module):
    """DeepSets scene trunk feeding a shared context to two heads:

    (a) a ``ConditionalMeanFlow`` over the measured observables -- its explicit mean
        head ``mu(context)`` carries the (un-shrunk) conditional-mean shear response,
        the residual flow carries scatter;
    (b) a Bernoulli detection head ``P(s=1 | context)``.

    Both heads read the SAME scene context, so the induced shape response and selection
    response are derivatives of one representation.  With ``n_flows`` neighbours the
    DeepSets pooling handles a variable set (0 neighbours -> primary-only, so isolated
    objects are the same model evaluated with an empty set).

    Note: the ``flow_drop_indices`` "blind the residual flow to the shape features"
    trick from the tabular ConditionalMeanFlow is NOT applied here -- the DeepSets
    context is a learned embedding with no identifiable shape columns.  The mean head's
    response is instead pinned by the response loss in the trainer.
    """

    def __init__(
        self,
        target_dim,
        primary_dim,
        neighbor_dim,
        context_dim=128,
        set_hidden_dim=128,
        set_neighbor_layers=2,
        set_context_layers=2,
        flow_hidden_dim=256,
        flow_layers=3,
        n_flows=8,
        mean_hidden=0,
        det_hidden=128,
        det_layers=2,
        activation="silu",
        pooling="sum",
        base_flow="affine",
        scale_limit=3.0,
        shape_skip_dim=0,
        shape_skip_idx=None,
        shape_skip_hidden=0,
    ):
        super().__init__()
        self.target_dim = int(target_dim)
        self.context_dim = int(context_dim)
        self.conditioner = DeepSetsConditioner(
            primary_dim=primary_dim,
            neighbor_dim=neighbor_dim,
            context_dim=context_dim,
            hidden_dim=set_hidden_dim,
            neighbor_layers=set_neighbor_layers,
            context_layers=set_context_layers,
            activation=activation,
            pooling=pooling,
        )
        self.mean_flow = ConditionalMeanFlow(
            target_dim=target_dim,
            context_dim=context_dim,
            base_flow=base_flow,
            mean_hidden=mean_hidden,
            hidden_dim=flow_hidden_dim,
            n_layers=flow_layers,
            n_flows=n_flows,
            scale_limit=scale_limit,
            activation=activation,
        )
        # FENCE (realisation-aware head). `mu()` and `log_prob_obs()` below hand-roll
        # `target - mean_flow._mu(context)`. For a ConditionalMeanFlowRA the residual is
        # `x - mu(c) - A(c, u)`, so those two would be WRONG rather than stale, and silently so.
        # The constructor only ever builds a plain ConditionalMeanFlow today; this assertion makes
        # a future swap fail loudly instead.
        if isinstance(self.mean_flow, ConditionalMeanFlowRA):
            raise NotImplementedError(
                "SetConditionedForwardModel hand-rolls x - mu(c) and does not support "
                "ConditionalMeanFlowRA (residual is x - mu(c) - A(c, u)).")
        if det_layers < 1:
            raise ValueError("det_layers must be >= 1")
        act = _act(activation)
        layers = []
        dim = context_dim
        for _ in range(det_layers - 1):
            layers += [nn.Linear(dim, det_hidden), act()]
            dim = det_hidden
        layers.append(nn.Linear(dim, 1))
        self.det_head = nn.Sequential(*layers)

        # --- ABLATION FIX (worktree-only): optional DIRECT intrinsic-shape skip into the mean.
        # Gives the mean head an un-smeared path to the primary true shape (e1/e2_input_p),
        # bypassing the DeepSets trunk that otherwise mediates the shape->response mapping.
        # `shape_skip_idx` are column indices into the standardized PRIMARY feature vector.
        # Zero-initialised so at start the model is IDENTICAL to the trunk-only baseline; the
        # response (and NLL) then learn the direct term. Off (=None) => baseline V2 unchanged.
        self.shape_skip_dim = int(shape_skip_dim)
        if self.shape_skip_dim > 0:
            if shape_skip_idx is None:
                raise ValueError("shape_skip_dim>0 requires shape_skip_idx")
            self.register_buffer("shape_skip_idx",
                                 torch.as_tensor(list(shape_skip_idx), dtype=torch.long))
            if shape_skip_hidden and shape_skip_hidden > 0:
                self.shape_skip = nn.Sequential(
                    nn.Linear(self.shape_skip_dim, shape_skip_hidden), act(),
                    nn.Linear(shape_skip_hidden, target_dim))
                _last = self.shape_skip[-1]
            else:
                self.shape_skip = nn.Linear(self.shape_skip_dim, target_dim)
                _last = self.shape_skip
            nn.init.zeros_(_last.weight)
            nn.init.zeros_(_last.bias)
        else:
            self.shape_skip = None

    def context(self, primary, neighbors, neighbor_mask):
        return self.conditioner(primary, neighbors, neighbor_mask)

    def _shape_skip_term(self, primary):
        if self.shape_skip is None or primary is None:
            return None
        return self.shape_skip(primary.index_select(1, self.shape_skip_idx))

    def mu(self, context, primary=None):
        """Conditional mean of the observables (carries the shape response). If the shape-skip
        head is present and `primary` (the standardized primary feature vector) is passed, add
        the DIRECT intrinsic-shape term so the response has a path bypassing the trunk."""
        m = self.mean_flow._mu(context)
        skip = self._shape_skip_term(primary)
        return m if skip is None else m + skip

    def detection_logit(self, context):
        return self.det_head(context).squeeze(-1)

    def detection_prob(self, context):
        return torch.sigmoid(self.detection_logit(context))

    def log_prob_obs(self, target, context, primary=None):
        """log p(measured observables | scene, detected) -- detected rows only. When the
        shape-skip head is active, the density mean = trunk mean + skip (SAME mean the response
        uses), so NLL and response stay consistent."""
        skip = self._shape_skip_term(primary)
        if skip is None:
            return self.mean_flow.log_prob(target, context)
        resid = target - self.mean_flow._mu(context) - skip
        return self.mean_flow.flow.log_prob(resid, self.mean_flow._flow_ctx(context))
