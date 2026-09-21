"""V3.6-like truth-only crowding features in state-major-axis coordinates."""
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

def major_convolved_scale(circularized_radius, ellipticity, psf_radius):
    radius, shape = np.asarray(circularized_radius, float), np.asarray(ellipticity, float)
    if (radius.ndim != 1 or shape.shape != (len(radius), 2) or not np.isfinite(radius).all()
            or not np.isfinite(shape).all() or np.any(radius <= 0)
            or not np.isfinite(psf_radius) or psf_radius <= 0):
        raise ValueError('positive radii and aligned finite two-component ellipticities required')
    modulus = np.linalg.norm(shape, axis=1)
    if np.any(modulus >= 1):
        raise ValueError('subunit ellipticity required')
    q = (1-modulus)/(1+modulus)
    return np.hypot(radius/np.sqrt(q), psf_radius)

def transport_major_distance(ids, context, target_shape, pairs, psf_radius):
    ids, context = np.asarray(ids), np.asarray(context, float)
    if ids.ndim != 1 or len(np.unique(ids)) != len(ids) or context.shape != (len(ids), 8):
        raise ValueError('unique parent identities and eight-input contexts required')
    row = pd.Index(ids).get_indexer(pairs['index_input_p'])
    if np.any(row < 0):
        raise ValueError('pair primary absent from parent')
    initial = major_convolved_scale(context[:, 4], context[:, :2], psf_radius)
    target = major_convolved_scale(context[:, 4], target_shape, psf_radius)
    # Validate the assumed cached coordinate rather than silently reconstructing
    # physical distances under a possibly different radius convention.
    expected = np.sqrt(1-(psf_radius/initial[row])**2)
    np.testing.assert_allclose(pairs['Re_input_p_scaled'], expected, rtol=2e-5, atol=2e-6)
    result = pairs.copy()
    result['distance_scaled'] = pairs['distance_scaled'].to_numpy()*(initial[row]/target[row])
    return result

SCALES = (.5, 1., 2., 4.)

def smooth_crowding(ids, pairs):
    # Reuse all finite-value, identity, duplicate and unmatched-row checks.
    geometry_features(ids, pairs)
    index = pd.Index(ids).get_indexer(pairs['index_input_p'])
    distance = pairs['distance_scaled'].to_numpy(float)
    ratio = 10.**(-.4*(pairs['r_input_s_scaled'].to_numpy(float)-pairs['r_input_p_scaled'].to_numpy(float)))
    features = np.column_stack([np.log1p(np.bincount(index,
        weights=ratio*np.exp(-.5*(distance/s)**2), minlength=len(ids))) for s in SCALES])
    if not np.isfinite(features).all():
        raise ValueError('finite smooth crowding required')
    return features

FLOW_FEATURES = ("e1_input_p", "e2_input_p", "sersic_n_input_p", "r_input_p",
                 "circularized_Re_input_p", "nbr_flux_near", "nbr_flux_far", "nbr_flux_max")

SMOOTH_FEATURES = ('nbr_gaussian_050', 'nbr_gaussian_100', 'nbr_gaussian_200', 'nbr_gaussian_400')

FEATURES = (*FLOW_FEATURES, *GEOMETRY_FEATURES, *SMOOTH_FEATURES)

FORMAT = 'scene_classifier_state_major_smooth_v1'

def augment_state(ids, context, state, pairs, psf_radius):
    if state.shape != (len(ids), 13):
        raise ValueError('aligned thirteen-input state required')
    np.testing.assert_array_equal(state[:, 2:8], context[:, 2:8])
    moved = transport_major_distance(ids, context, state[:, :2], pairs, psf_radius)
    geometry, _ = geometry_features(ids, moved)
    np.testing.assert_array_equal(state[:, 8:13], geometry)
    return np.column_stack((state, smooth_crowding(ids, moved))).astype(np.float32)

def classifier_features(ids, context, target_shape, pairs, psf_radius):
    """Build all 17 inputs, recomputing crowding for each sheared state.

    ``context`` is the original eight-column flow frame; ``pairs`` uses its
    PSF-convolved major-axis distance. Shapes change at fixed circularized
    radius and physical neighbour separation. Parent order is preserved.
    """
    context = np.asarray(context)
    moved = transport_major_distance(ids, context, target_shape, pairs, psf_radius)
    geometry, _ = geometry_features(ids, moved)
    state = np.column_stack((target_shape, context[:, 2:], geometry))
    return augment_state(ids, context, state, pairs, psf_radius)
