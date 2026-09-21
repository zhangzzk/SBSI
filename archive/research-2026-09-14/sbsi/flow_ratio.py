"""Balanced shear-leg classification using a conditional flow's density ratio.

The logit is log q(x|c1)-log q(x|c0), WITHOUT division by shear.
Matched pairs balance the base-galaxy distribution across labels but condition
on both legs having usable measurements. This does not correct that population
difference from the independently detected per-leg NLL population.
"""

import math

import numpy as np
import torch
from torch.nn import functional as F


def paired_ratio_losses(model, batch):
    """One balanced BCE per base galaxy; deterministic, differentiable log q only.

    A shared invertible output standardization contributes the same Jacobian
    under both contexts and cancels in each logit. No model samples are used.
    """
    x0, c0, x1, c1 = batch
    if (x0.ndim != 2 or x0.shape != x1.shape or len(x0) == 0
            or c0.ndim != 2 or c0.shape != c1.shape or len(c0) != len(x0)):
        raise ValueError("ratio loss requires nonempty aligned measurement/context pairs")
    z0 = model.log_prob(x0, c1) - model.log_prob(x0, c0)
    z1 = model.log_prob(x1, c1) - model.log_prob(x1, c0)
    if z0.shape != (len(x0),) or z1.shape != z0.shape:
        raise ValueError("log_prob must return one scalar per measurement")
    return 0.5 * (F.softplus(z0) + F.softplus(-z1))


def epoch_nll_ratio(model, loader, paired_loader, device, ratio_weight,
                    optimizer=None, max_grad_norm=None):
    """NLL on all usable leg rows plus balanced BCE on matched pairs.

    Training cycles a shuffled pair loader alongside the full NLL loader.
    Validation visits each NLL row and each matched pair once, with separate
    denominators. The returned means define held-out checkpoint selection.
    """
    if not math.isfinite(ratio_weight) or ratio_weight <= 0 or paired_loader is None:
        raise ValueError("positive finite ratio weight and paired data are required")
    training = optimizer is not None
    model.train(training)
    pair_iterator = iter(paired_loader) if training else None
    nll_sum = weight_sum = ratio_sum = pair_count = 0
    for batch in loader:
        values, context = (v.to(device, non_blocking=True) for v in batch[:2])
        weights = batch[2].to(device, non_blocking=True) if len(batch) > 2 else None
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            log_prob = model.log_prob(values, context)
            mass = len(values) if weights is None else float(weights.sum().detach())
            if mass <= 0:
                raise ValueError("NLL batch must have positive weight")
            nll = -log_prob.mean() if weights is None else -(weights * log_prob).sum() / mass
            loss = nll
            if training:
                try:
                    pair = next(pair_iterator)
                except StopIteration:
                    pair_iterator = iter(paired_loader)
                    pair = next(pair_iterator, None)
                    if pair is None:
                        raise ValueError("empty paired data")
                pair = tuple(v.to(device, non_blocking=True) for v in pair)
                scores = paired_ratio_losses(model, pair)
                loss = loss + ratio_weight * scores.mean()
                ratio_sum += float(scores.detach().sum())
                pair_count += len(scores)
            if not bool(torch.isfinite(loss)):
                raise ValueError("nonfinite NLL + ratio objective")
            if training:
                loss.backward()
                if max_grad_norm is not None and max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
        nll_sum += float(nll.detach()) * mass
        weight_sum += mass
    if not training:
        with torch.no_grad():
            for pair in paired_loader:
                scores = paired_ratio_losses(model, tuple(v.to(device, non_blocking=True) for v in pair))
                if not bool(torch.isfinite(scores).all()):
                    raise ValueError("nonfinite ratio objective")
                ratio_sum += float(scores.sum())
                pair_count += len(scores)
    if weight_sum <= 0 or pair_count == 0:
        raise ValueError("empty NLL or paired data")
    return nll_sum / weight_sum, ratio_sum / pair_count


def validate_ratio_options(args):
    """Fail closed rather than silently combine historical derivative penalties."""
    if not math.isfinite(args.ratio_response_weight) or args.ratio_response_weight < 0:
        raise ValueError("ratio response weight must be finite and nonnegative")
    if args.ratio_response_weight == 0:
        return
    competing = (args.response_weight, args.coupling_weight, args.paired_transport_weight,
                 args.response_dispersion_weight, args.distribution_response_weight,
                 args.preserve_weight, args.ra_weight, args.response_global_anchor)
    targets = (args.response_target_npz, args.coupling_target_npz,
               args.response_target_perobj, args.ra_target_npz)
    if any(v != 0 for v in competing) or any(v is not None for v in targets):
        raise ValueError("ratio response must be the single derivative regularizer; remove other losses/targets")
    if args.nll_weight != 1:
        raise ValueError("ratio response requires NLL weight 1")
    if (args.flow_type != "affine" or not args.fold_catalogue_shear
            or len(args.additional_catalogue or []) != 1
            or args.validation_group_column != "case"):
        raise ValueError("ratio response requires affine flow, folded two-catalogue inputs and case split")
    if args.noise_photoz or args.noise_sersic_frac or args.decorrelate_shape_size:
        raise ValueError("ratio trial requires shared base properties without added noise or decorrelation weights")


def verify_ratio_base_properties(frame, pairs, condition_features):
    """Check intrinsic pre-shear shape and every nonmoving condition per pair."""
    fixed = [f for f in condition_features if f not in ("e1_input_p", "e2_input_p")]
    checked = fixed + ["_ratio_base_e1", "_ratio_base_e2"]
    for feature in checked:
        values = frame[feature].to_numpy(float)
        if not np.array_equal(values[pairs[:, 0]], values[pairs[:, 1]], equal_nan=True):
            raise ValueError(f"ratio pairs change base-U feature {feature}")
    return checked
