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

from .measurement_model import ConditionalMeanFlow
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

    def context(self, primary, neighbors, neighbor_mask):
        return self.conditioner(primary, neighbors, neighbor_mask)

    def mu(self, context):
        """Conditional mean of the observables (carries the shape response)."""
        return self.mean_flow._mu(context)

    def detection_logit(self, context):
        return self.det_head(context).squeeze(-1)

    def detection_prob(self, context):
        return torch.sigmoid(self.detection_logit(context))

    def log_prob_obs(self, target, context):
        """log p(measured observables | scene, detected) -- detected rows only."""
        return self.mean_flow.log_prob(target, context)
