"""Primary-cut probability diagnostics for per-leg guard checkpoint selection."""
import numpy as np
import torch

from sbsi.flow_guard_response import physical_context_draws, torch_guard_weights


def primary_cut_index(population):
    matches = [i for i, cut in enumerate(population.cuts)
               if cut['radius_min'] == 3. and cut['magnitude_max'] == 25.8]
    if len(matches) != 1:
        raise ValueError('one primary measured radius>3px and magnitude<25.8 cut required')
    return matches[0]


@torch.no_grad()
def estimate_primary_mass(model, population, *, seed, draws=4, groups=2, batch_size=2048):
    """Integrate U0*w(Y0) over every parent, with soft and exact hard cuts."""
    if groups < 2 or batch_size < 1 or not len(population.pairs):
        raise ValueError('positive population/batch and at least two groups required')
    cut = population.cuts[primary_cut_index(population)]
    n, device = len(population.pairs), population.context.device
    estimates = []
    for group in range(groups):
        total = torch.zeros(2, dtype=torch.float64, device=device)
        for block, start in enumerate(range(0, n, batch_size)):
            rows = torch.arange(start, min(start+batch_size, n), device=device)
            values = physical_context_draws(model, [population.context[population.pairs[rows, 0]]],
                draws, seed+group*1000000000+block)[0]
            if not bool(torch.isfinite(values).all() and (values[..., 2:] >= 0).all()):
                raise ValueError('finite generated values and nonnegative photometry required')
            soft = torch_guard_weights(values, (cut,), radius_softness=population.radius_softness,
                magnitude_softness=population.magnitude_softness, zero_point=population.zero_point)[..., 0]
            magnitude = population.zero_point-2.5*torch.log10(values[..., 3])
            hard = (values[..., 2] > 3.) & (magnitude < 25.8)
            weights = torch.stack((soft, hard.to(soft.dtype)), -1).mean(1)
            total += (weights*population.usable[rows, 0, None]).double().sum(0)
        estimates.append((total/n).cpu().numpy())
    estimates = np.asarray(estimates)
    return {'coordinates': ['soft_primary', 'hard_primary'], 'estimate': estimates.mean(0),
        'group_estimates': estimates, 'integration_se': estimates.std(0, ddof=1)/np.sqrt(groups),
        'parents': n, 'draws': draws, 'groups': groups, 'seed': seed,
        'definition': 'baseline own-leg U times cut probability, divided by all truth parents'}
