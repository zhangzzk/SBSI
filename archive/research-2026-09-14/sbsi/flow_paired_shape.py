"""Grid-free, uniformly pair-weighted supervision of physical mean changes.

For measured d=e_plus-e_zero and h=|gamma|, the objective is
E_pair[(Delta_mu-d).(Delta_mu-d)/(4*h**2)]. Two independent latent
replicas on the SAME pairs estimate the product without penalizing flow Monte
Carlo variance. Each replica shares antithetic base draws between shear legs.
An optional third component adds radius divided by an explicit fixed length
scale, retaining divisor4 so the existing shape coefficient does not change.
There is no cell assignment, fitted response matrix, or covariance weighting.
"""
import math

import numpy as np
import torch

from .flow_physical_population import physical_context_means


class PairedResponsePopulation:
    """Shape, optionally plus physical radius in a declared fixed length unit."""
    def __init__(self, context, pairs, measured_delta, gamma, *, radius_scale=None):
        device = context.device
        self.context = context
        pairs = np.asarray(pairs)
        delta, gamma = np.asarray(measured_delta), np.asarray(gamma)
        components = 2 if radius_scale is None else 3
        if radius_scale is not None and (not math.isfinite(radius_scale) or radius_scale <= 0):
            raise ValueError("positive finite physical radius scale required")
        if (context.ndim != 2 or pairs.ndim != 2 or pairs.shape[1] != 2 or not len(pairs)
                or not np.issubdtype(pairs.dtype, np.integer) or np.any(pairs < 0)
                or np.any(pairs >= len(context)) or delta.shape != (len(pairs), components)
                or gamma.shape != pairs.shape or not np.isfinite(delta).all()
                or not np.isfinite(gamma).all()):
            raise ValueError("finite aligned shape differences, shear vectors and paired rows required")
        amplitude = np.linalg.norm(gamma, axis=1)
        if np.any(amplitude <= 0) or not np.isfinite(amplitude).all():
            raise ValueError("nonzero finite shear amplitude required; no flooring")
        self.pairs = torch.as_tensor(pairs, device=device, dtype=torch.long)
        self.measured_delta = torch.as_tensor(delta, device=device, dtype=torch.float64)
        self.amplitude = torch.as_tensor(amplitude, device=device, dtype=torch.float64)
        self.output_scale = torch.as_tensor([1., 1.] + ([] if radius_scale is None else [radius_scale]),
                                           device=device, dtype=torch.float64)

    def residual(self, model, indices, draws, seed):
        pair = self.pairs[indices]
        means = physical_context_means(
            model, [self.context[pair[:, 0]], self.context[pair[:, 1]]], draws, seed)
        components = len(self.output_scale)
        value = ((means[1][:, :components]-means[0][:, :components]-self.measured_delta[indices])
                 / self.amplitude[indices, None] / self.output_scale)
        if not bool(torch.isfinite(value).all()):
            raise ValueError("nonfinite paired shape residual")
        return value


class PairedShapePopulation(PairedResponsePopulation):
    """Backwards-compatible shape-only population; no radius scale accepted."""
    def __init__(self, context, pairs, measured_delta, gamma):
        super().__init__(context, pairs, measured_delta, gamma)


def paired_cross_score(a, c):
    """Per-pair cross error, with the original divisor4 retained when adding radius."""
    if a.ndim != 2 or a.shape[-1] not in (2, 3) or c.shape != a.shape:
        raise ValueError("aligned two- or three-component per-pair residuals required")
    return (a*c).sum(-1)/4


def paired_shape_backward(model, population, pairs_per_step, draws, *, chunk_size,
                          seed, weight=1.):
    """Accumulate a sampled per-pair loss gradient without changing existing gradients."""
    if (min(pairs_per_step, chunk_size) < 1 or draws < 2 or draws % 2 or seed < 0
            or not math.isfinite(weight) or weight < 0):
        raise ValueError("positive batch sizes, even draws, and nonnegative finite weight/seed required")
    device = population.context.device
    # Use a distinct generator family for galaxy selection. The epoch/step seed
    # arithmetic can overlap a previous latent seed, so sharing Torch's stream
    # here would couple later pair choices to an earlier optimizer update.
    indices = torch.as_tensor(np.random.default_rng(seed).integers(
        len(population.pairs), size=pairs_per_step), device=device, dtype=torch.long)
    total = 0.
    component_losses = torch.zeros(len(population.output_scale), device=device, dtype=torch.float64)
    for block, start in enumerate(range(0, pairs_per_step, chunk_size)):
        rows = indices[start:start+chunk_size]
        with torch.set_grad_enabled(weight > 0):
            a = population.residual(model, rows, draws, seed+100000+2*block)
            c = population.residual(model, rows, draws, seed+100001+2*block)
            loss = paired_cross_score(a, c).sum()/pairs_per_step
            if not bool(torch.isfinite(loss)):
                raise ValueError("nonfinite paired shape score")
            if weight:
                (weight*loss).backward()
        total += float(loss.detach())
        component_losses += (a.detach()*c.detach()).sum(0)/(4*pairs_per_step)
    return dict(loss=total, pairs_per_group=pairs_per_step,
                component_losses=component_losses.cpu().numpy())


def summarize_paired_residuals(residuals):
    """All distinct latent-group products per pair, with a latent-group jackknife.

    Independent groups must contain exactly the same pairs in the same order.
    This does not square an average across galaxies. The error describes Monte
    Carlo integration only, not simulation cases or training-model uncertainty.
    """
    residuals = np.asarray(residuals, dtype=np.float64)
    if (residuals.ndim != 3 or residuals.shape[0] < 4 or residuals.shape[1] < 1
            or residuals.shape[2] not in (2, 3) or not np.isfinite(residuals).all()):
        raise ValueError("at least four finite aligned latent groups of shape residuals required")
    def scores(values):
        count = len(values)
        return (values.sum(0)**2-(values**2).sum(0)).sum(-1)/(4*count*(count-1))
    per_pair = scores(residuals)
    deleted = np.array([scores(np.delete(residuals, k, axis=0)).mean()
                        for k in range(len(residuals))])
    se = math.sqrt((len(deleted)-1)/len(deleted)*np.sum((deleted-deleted.mean())**2))
    naive = np.mean(np.sum(residuals.mean(0)**2, axis=-1))/4
    component_scores = (residuals.sum(0)**2-(residuals**2).sum(0)).mean(0)/(4*len(residuals)*(len(residuals)-1))
    component_deleted = np.array([
        (v.sum(0)**2-(v**2).sum(0)).mean(0)/(4*len(v)*(len(v)-1))
        for v in (np.delete(residuals,k,axis=0) for k in range(len(residuals)))])
    def component_se(values):
        return float(np.sqrt((len(values)-1)/len(values)*np.sum((values-values.mean())**2)))
    result = dict(response_loss=float(per_pair.mean()),
                shape_loss=float(component_scores[:2].sum()), per_pair=per_pair,
                component_losses=component_scores,
                shape_mc_jackknife_se=component_se(component_deleted[:,:2].sum(1)),
                mc_jackknife_se=se, mc_deleted_scores=deleted,
                mean_quadratic=float(naive), integration_quadratic=float(naive-per_pair.mean()))
    if residuals.shape[2] == 3:
        result.update(radius_loss=float(component_scores[2]),
                      radius_mc_jackknife_se=component_se(component_deleted[:,2]))
    else:
        # Preserve the original summation order for the legacy selection metric.
        result['shape_loss'] = result['response_loss']
    return result


def evaluate_paired_shape(model, population, indices, *, draws=64, groups=4,
                          batch_size=1024, seed=3180000000001):
    if groups < 4 or groups > 100 or batch_size < 1 or len(indices) < 1 or len(indices) >= 1000000000:
        raise ValueError("valid disjoint group/batch sizes and nonempty pair indices required")
    model.eval()
    indices = torch.as_tensor(indices, dtype=torch.long, device=population.context.device)
    values = []
    with torch.no_grad():
        for group in range(groups):
            chunks = [population.residual(model, indices[start:start+batch_size], draws,
                      seed+group*1000000000+start).cpu().numpy()
                      for start in range(0, len(indices), batch_size)]
            values.append(np.concatenate(chunks))
    residuals = np.stack(values)
    return dict(**summarize_paired_residuals(residuals), group_residuals=residuals,
                draws=draws, groups=groups, seed=seed, pairs=len(indices))
