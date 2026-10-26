"""Truth-only geometry features for a controlled usability-classifier experiment.

Pair distance is in units of the primary PSF-convolved size, as defined by
BlendEMU's rescale function. No learned response or measured output is used.
"""
import numpy as np
import pandas as pd

EDGES = np.array([0., .5, 1., 2., 4., np.inf])
GEOMETRY_FEATURES = ('nbr_log1p_fluxratio_d0_0p5', 'nbr_log1p_fluxratio_d0p5_1',
                     'nbr_log1p_fluxratio_d1_2', 'nbr_log1p_fluxratio_d2_4',
                     'nbr_log1p_fluxratio_d4_inf')
PAIR_COLUMNS = ('index_input_p', 'index_input_s', 'r_input_p_scaled', 'r_input_s_scaled', 'distance_scaled')


def geometry_features(ids, pairs):
    ids = np.asarray(ids)
    if ids.ndim != 1 or len(ids) == 0 or not np.issubdtype(ids.dtype, np.integer) or len(np.unique(ids)) != len(ids):
        raise ValueError('unique integer parent identities required')
    if not set(PAIR_COLUMNS) <= set(pairs.columns):
        raise ValueError('missing truth-pair columns')
    for column in PAIR_COLUMNS[:2]:
        if not np.issubdtype(pairs[column].dtype, np.integer):
            raise ValueError('integer pair identities required')
    if pairs.duplicated(list(PAIR_COLUMNS[:2])).any():
        raise ValueError('duplicate pair identities')
    if (pairs['index_input_p'] == pairs['index_input_s']).any():
        raise ValueError('self-pairs are not neighbours')
    row = pd.Index(ids).get_indexer(pairs['index_input_p'])
    if (row < 0).any():
        raise ValueError(f'{int((row < 0).sum())} pairs have primaries outside the parent')
    values = pairs[list(PAIR_COLUMNS[2:])].to_numpy(float)
    if not np.isfinite(values).all() or np.any(values[:, 2] < 0):
        raise ValueError('finite magnitudes and nonnegative scaled distances required')
    ratio = 10.**(-.4*(values[:, 1]-values[:, 0]))
    if not np.isfinite(ratio).all():
        raise ValueError('nonfinite neighbour/primary flux ratio')
    bins = np.searchsorted(EDGES[1:-1], values[:, 2], side='right')
    sums = np.bincount(row*len(GEOMETRY_FEATURES)+bins, weights=ratio,
                       minlength=len(ids)*len(GEOMETRY_FEATURES)).reshape(len(ids), -1)
    features = np.log1p(sums).astype(np.float32)
    if not np.isfinite(features).all():
        raise ValueError('nonfinite geometry features')
    count = np.bincount(row, minlength=len(ids))
    return features, {'parents': len(ids), 'pairs': len(pairs), 'parents_without_pairs': int((count == 0).sum()),
                      'unmatched_pair_primaries': 0, 'maximum_pairs_per_parent': int(count.max(initial=0)),
                      'pairs_per_annulus': np.bincount(bins, minlength=len(GEOMETRY_FEATURES)).tolist()}
