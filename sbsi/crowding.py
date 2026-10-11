"""Truth-only physical crowding features of the V3.9-like model."""
import numpy as np
from scipy.spatial import cKDTree

V39_FEATURES = ('e1_input_p', 'e2_input_p', 'sersic_n_input_p', 'r_input_p',
                'circularized_Re_input_p', 'nbr_close_050', 'nbr_close_100', 'nbr_close_200')


def physical_close_crowding(scene, primary_rows=None, *, row_chunk=8192, workers=1):
    """Sum all nonself neighbours within 10 arcsec in one independent scene.

    Matches the training projection: RA*cos(mean DEC), DEC, in arcseconds.
    Restricting primary_rows never removes potential secondary neighbours.
    Return float64 sums after log1p; cast to float32 for the model contract.
    """
    values = scene[['RA', 'DEC', 'r']].to_numpy(float)
    if not len(values) or not np.isfinite(values).all() or row_chunk < 1:
        raise ValueError('nonempty finite scene and positive chunk required')
    if 'case' in scene and scene['case'].nunique() != 1:
        raise ValueError('one independent scene required; do not mix overlapping cases')
    rows = np.arange(len(scene)) if primary_rows is None else np.asarray(primary_rows)
    if rows.ndim != 1 or not np.issubdtype(rows.dtype, np.integer) or np.any((rows < 0) | (rows >= len(scene))):
        raise ValueError('valid integer primary rows required')
    xy = values[:, :2].copy()
    xy[:, 0] *= np.cos(np.deg2rad(values[:, 1].mean()))
    xy *= 3600.
    tree = cKDTree(xy)
    result = np.empty((len(rows), 3))
    for lo in range(0, len(rows), row_chunk):
        primary = rows[lo:lo+row_chunk]
        neighbours = tree.query_ball_point(xy[primary], 10., workers=workers, return_sorted=True)
        counts = np.array([len(x) for x in neighbours], dtype=np.int64)
        secondary = np.concatenate(neighbours).astype(np.int64)
        owner = np.repeat(np.arange(len(primary)), counts)
        nonself = secondary != primary[owner]
        secondary, owner = secondary[nonself], owner[nonself]
        distance2 = np.square(xy[primary[owner]]-xy[secondary]).sum(1)
        ratio = 10.**(-.4*(values[secondary, 2]-values[primary[owner], 2]))
        for column, scale in enumerate((.5, 1., 2.)):
            result[lo:lo+len(primary), column] = np.log1p(np.bincount(owner,
                weights=ratio*np.exp(-distance2/(2*scale**2)), minlength=len(primary)))
    if not np.isfinite(result).all():
        raise ValueError('nonfinite physical crowding')
    return result
