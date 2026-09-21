"""Deterministic validation subsets with paired model/measurement WLS extraction."""
import numpy as np
from .flow_grid_response import fit_grid_response


def validation_pair_subset(payload, physical_shapes, limit, seed):
    """Refit the observed subset, retaining the original fixed precision as weights.

    The smaller validation population has its own measured response and design;
    full-population covariance weights stay fixed and are not uncertainty claims
    for this subset. Training data, targets and covariance are untouched.
    """
    pairs=np.asarray(payload['pairs'])
    if limit < 1 or limit > len(pairs):
        raise ValueError('validation pair cap must be within the original population')
    selected=np.sort(np.random.default_rng(seed).choice(len(pairs),limit,replace=False))
    used=pairs[selected]
    shapes=np.asarray(physical_shapes)
    delta=shapes[used[:,1],:2]-shapes[used[:,0],:2]
    stats=fit_grid_response(delta,np.asarray(payload['gamma'])[selected],
                           np.asarray(payload['bin_ids'])[selected],
                           np.asarray(payload['pair_keys'])[selected,0],len(payload['counts']))
    result=dict(payload)
    for name in ('pairs','bin_ids','gamma','pair_keys'):
        result[name]=np.asarray(payload[name])[selected]
    for name in ('counts','design','target'):
        result[name]=stats[name]
    report=dict(selected_pair_indices=selected,subset_target=stats['target'],
                subset_covariance=stats['covariance'],subset_counts=stats['counts'],
                retained_precision=np.asarray(payload['precision']),subset_case_coverage=stats['case_coverage'])
    return result,report
