"""Pure selection and one-dimensional likelihood helpers for bright MH tests."""
import numpy as np
from scipy.optimize import brentq


def bright_cohorts(magnitude, limits, count):
    """First count observations per measured cut, with shared rows retained once."""
    magnitude = np.asarray(magnitude, float)
    if count < 1 or not np.isfinite(magnitude).all():
        raise ValueError('finite measured magnitudes and positive count required')
    selected = {str(float(c)): np.flatnonzero(magnitude < c)[:count] for c in limits}
    union = np.unique(np.concatenate(list(selected.values())))
    cohorts = {c: np.searchsorted(union, ids).tolist() for c, ids in selected.items()}
    return union, cohorts, all(len(v) == count for v in selected.values())


def selected_log_profile(log_numerators, selection, count, centre):
    """One common measured-selection factor per retained observation."""
    selection = np.asarray(selection, float)
    numerator = np.asarray(log_numerators, float)
    if (selection.shape != numerator.shape or np.any(selection <= 0)
            or not np.isfinite(selection).all() or not np.isfinite(numerator).all()):
        raise ValueError('finite profiles and positive selection required')
    return numerator-numerator[centre]-count*np.log(selection/selection[centre])


def profile_fit(grid, loglike):
    """Quadratic around the peak, linear outside; use that same curve for intervals."""
    x, y = np.asarray(grid, float), np.asarray(loglike, float)
    if (x.ndim != 1 or y.shape != x.shape or len(x) < 3 or not np.isfinite(x).all()
            or not np.isfinite(y).all() or np.any(np.diff(x) <= 0)):
        raise ValueError('ordered finite likelihood profile required')
    k = int(np.argmax(y))
    if k in (0, len(x)-1):
        return dict(valid=False, reason='profile maximum at grid boundary', grid_peak=float(x[k]))
    coefficient = np.polyfit(x[k-1:k+2]-x[k], y[k-1:k+2]-y[k], 2)
    a, b, _ = coefficient
    peak = x[k]-b/(2*a) if a < 0 else np.nan
    if not x[k-1] <= peak <= x[k+1]:
        return dict(valid=False, reason='no concave bracketed quadratic peak', grid_peak=float(x[k]))
    height = y[k]+np.polyval(coefficient, peak-x[k])
    def interpolated(value):
        if x[k-1] <= value <= x[k+1]:
            return float(y[k]+np.polyval(coefficient, value-x[k]))
        return float(np.interp(value, x, y))

    intervals = {}
    for label, drop in (('68', .5), ('95', 1.920729410347062)):
        level = height-drop
        bounds = []
        for nodes in (np.r_[peak, x[x < peak][::-1]], np.r_[peak, x[x > peak]]):
            crossing = None
            for left, right in zip(nodes[:-1], nodes[1:]):
                if (interpolated(left)-level)*(interpolated(right)-level) <= 0:
                    crossing = float(brentq(lambda value: interpolated(value)-level,
                                           min(left, right), max(left, right), xtol=1e-14))
                    break
            bounds.append(crossing)
        intervals[label] = bounds
    return dict(valid=True, estimate=float(peak), curvature_se=float(1/np.sqrt(-2*a)),
                intervals=intervals, grid_peak=float(x[k]),
                limitation='Conditional g1 fit at g2=0; quadratic peak segment and linear outer segments, same interpolation for intervals. Grid refinement and posterior convergence still required.')
