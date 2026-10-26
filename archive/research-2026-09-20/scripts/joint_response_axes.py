"""Explicit shear-axis helpers for joint truth-parent response validation."""
import numpy as np
import pandas as pd
import math

from plots.constgold_property_bias import response_from_arrays, summarize_branch
from sbsi.fixed_g0_domain import FLOW_FEATURES
from sbsi.shear_map import apply_shear_to_ellipticity
from scripts.train_fixed_g0_flow import write_json as _write_json


def json_ready(value):
    """Recursively preserve undefined pilot uncertainty as JSON null."""
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, np.generic):
        return json_ready(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path, result):
    _write_json(path, json_ready(result))


def shear_vector(amplitude, axis):
    if axis not in (1, 2) or not np.isfinite(amplitude) or abs(amplitude) >= 1:
        raise ValueError('axis 1 or 2 and finite subunit shear required')
    result = np.zeros(2)
    result[axis-1] = amplitude
    return result


def shear_frames(context, h, axis=1):
    context = np.asarray(context)
    if (context.ndim != 2 or context.shape[1] != len(FLOW_FEATURES)
            or not np.isfinite(context).all() or h <= 0):
        raise ValueError('finite physical context and positive shear step required')
    frames = []
    for sign in (1, -1):
        vector = shear_vector(sign*h, axis)
        shape = apply_shear_to_ellipticity(context[:, 0], context[:, 1], *vector)
        frame = context.copy()
        frame[:, :2] = np.column_stack(shape)
        frames.append(pd.DataFrame(frame, columns=FLOW_FEATURES))
    return tuple(frames)


def summarize_axis_branch(stats, weights, h, axis=1):
    """Use the diagonal response for m; keep output vectors in e1,e2 order.

For a deliberately single-case operational pilot, weights=None produces
point estimates with undefined uncertainty, not an artificial bootstrap.
"""
    shear_vector(h, axis)
    if h <= 0:
        raise ValueError('positive finite-difference step required')
    if weights is None:
        responses = {source: response_from_arrays(stats[source]['plus'], stats[source]['minus'], h)
                     for source in ('measured', 'model')}
        bins = len(responses['measured'])
        return {'measured_response': responses['measured'], 'model_response': responses['model'],
                'measured_response_standard_error': np.full(bins, np.nan),
                'model_response_standard_error': np.full(bins, np.nan),
                'm_percent': 100*(responses['measured'][:, axis-1]/responses['model'][:, axis-1]-1),
                'm_standard_error_percentage_points': np.full(bins, np.nan),
                'm_ci95_percent': np.full((bins, 2), np.nan)}
    if axis == 1:
        return summarize_branch(stats, weights, h)
    swapped = {source: {direction: {
        'denominator': entry['denominator'], 'numerator': entry['numerator'][..., ::-1]
    } for direction, entry in directions.items()} for source, directions in stats.items()}
    result = summarize_branch(swapped, weights, h)
    for key in ('measured_response', 'model_response'):
        result[key] = result[key][..., ::-1]
    return result
