"""Explicit invariant size for experimental, area-preserving shear flows.

The radius and axis ratio here are BOTH pre-shear catalogue properties. Never
combine the original semi-major radius with a post-shear axis ratio. This
module does not change catalogue cuts, the response grid, or measured outputs.
"""

import numpy as np
import pandas as pd

from .selection_model import TabularPreprocessor


CIRCULARIZED_FEATURE = "circularized_Re_input_p"
SEMIMAJOR_FEATURE = "Re_input_p"


def circularized_radius(semimajor, unsheared_axis_ratio):
    """Return a0 sqrt(q0), rejecting invalid truth instead of repairing it."""
    a, q = np.broadcast_arrays(
        np.asarray(semimajor, dtype=np.float64), np.asarray(unsheared_axis_ratio, dtype=np.float64)
    )
    if (
        not np.isfinite(a).all()
        or not np.isfinite(q).all()
        or np.any(a <= 0)
        or np.any(q <= 0)
        or np.any(q > 1)
    ):
        raise ValueError("finite positive pre-shear semi-major radius and 0 < q0 <= 1 required")
    return a * np.sqrt(q)


def fit_circularized_preprocessor(reference, training_radius):
    """Replace only the size feature and its train-fitted standardization."""
    if reference.feature_names.count(SEMIMAJOR_FEATURE) != 1:
        raise ValueError("exactly one original semi-major size feature required")
    if SEMIMAJOR_FEATURE in reference.log_features:
        raise ValueError("this trial requires an unlogged original size condition")
    radius = np.asarray(training_radius, dtype=np.float64)
    if radius.ndim != 1 or len(radius) < 2 or not np.isfinite(radius).all() or np.any(radius <= 0):
        raise ValueError("finite positive training-only circularized radii required")
    single = TabularPreprocessor.fit(pd.DataFrame({CIRCULARIZED_FEATURE: radius}), [CIRCULARIZED_FEATURE])
    state = reference.to_state()
    state = {key: value.copy() if hasattr(value, "copy") else value for key, value in state.items()}
    result = TabularPreprocessor.from_state(state)
    index = result.feature_names.index(SEMIMAJOR_FEATURE)
    result.feature_names[index] = CIRCULARIZED_FEATURE
    for name in ("fill_values", "means", "scales"):
        getattr(result, name)[index] = getattr(single, name)[0]
    return result


def circularized_context(original_context, radius, preprocessor):
    """Replace the size column of an aligned standardized context matrix.

    All other columns remain bitwise unchanged. The separately named feature in
    the saved preprocessor makes an old semi-major input fail instead of silently
    being reinterpreted when a trial checkpoint is loaded.
    """
    context = np.asarray(original_context)
    radius = np.asarray(radius, dtype=np.float64)
    if (
        context.ndim != 2
        or context.shape[1] != preprocessor.output_dim
        or radius.shape != (len(context),)
        or not np.isfinite(radius).all()
        or np.any(radius <= 0)
    ):
        raise ValueError("aligned contexts and finite positive circularized radii required")
    index = preprocessor.feature_names.index(CIRCULARIZED_FEATURE)
    out = context.copy()
    out[:, index] = (radius.astype(np.float32) - preprocessor.means[index]) / preprocessor.scales[index]
    if preprocessor.add_missing_indicators:
        out[:, index + len(preprocessor.feature_names)] = 0
    return out
