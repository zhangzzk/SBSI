"""Explicit major-axis distance coordinates at fixed circularized source size.

This diagnostic helper does not alter the frozen static-geometry classifier.
"""
import numpy as np
import pandas as pd


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
