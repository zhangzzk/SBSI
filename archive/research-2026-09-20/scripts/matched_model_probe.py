"""Model evaluation primitives for the matched random-sign image experiment.

Signs label every generated source, not just usable objects or truth parents.
The flow context follows the primary's sign; a pair's disk velocity follows
the secondary's sign. This extends the existing pair-sum model to a declared
spatial shear field. It does not fit a new model or correct its predictions.
Actual usability must still come from the corresponding image leg in the
caller, and the influence estimator needs an explicitly shared baseline.
"""
import numpy as np
import pandas as pd

from sbsi.fixed_g0_domain import FLOW_FEATURES
from sbsi.output_conditioned_response import predict_scene_draw_responses
from sbsi.shear_map import apply_shear_to_ellipticity


def _identities(values):
    values = np.asarray(values)
    if values.ndim != 1 or values.dtype.kind not in 'iu':
        raise ValueError('one-dimensional integer source identities required')
    if values.dtype.kind == 'u' and np.any(values > np.iinfo(np.int64).max):
        raise ValueError('source identities exceed signed integer range')
    return values.astype(np.int64, copy=False)


class SourceShearSigns:
    """Immutable identity lookup; absence never silently becomes zero shear."""

    def __init__(self, source_ids, signs):
        ids = _identities(source_ids)
        signs = np.asarray(signs)
        if (signs.shape != ids.shape or not np.isin(signs, [-1, 1]).all()
                or len(np.unique(ids)) != len(ids) or not len(ids)):
            raise ValueError('unique generated-source identities and Rademacher signs required')
        self._index = pd.Index(ids.copy())
        self._signs = signs.astype(np.int8, copy=True)
        self._signs.flags.writeable = False

    def at(self, identities):
        locations = self._index.get_indexer(_identities(identities))
        if np.any(locations < 0):
            raise ValueError('source absent from the generated-source shear field')
        return self._signs[locations]


def signed_shear_frames(context, primary_ids, source_signs, *, h, axis):
    """Apply antithetic physical shear to each primary's intrinsic shape."""
    context = np.asarray(context, dtype=np.float64)
    ids = _identities(primary_ids)
    if (context.shape != (len(ids), len(FLOW_FEATURES))
            or len(np.unique(ids)) != len(ids) or not np.isfinite(context).all()
            or axis not in (1, 2) or not np.isfinite(h) or not 0 < h < 1):
        raise ValueError('aligned finite contexts, unique primaries, axis 1/2 and subunit positive step required')
    gamma = np.zeros((len(ids), 2))
    gamma[:, axis-1] = h*source_signs.at(ids)
    frames = []
    for orientation in (1, -1):
        shapes = apply_shear_to_ellipticity(context[:, 0], context[:, 1],
                                          orientation*gamma[:, 0], orientation*gamma[:, 1])
        frame = context.copy()
        frame[:, :2] = np.column_stack(shapes)
        frames.append(pd.DataFrame(frame, columns=FLOW_FEATURES))
    return tuple(frames)


class _SignedPairResponse:
    def __init__(self, candidate, source_signs):
        if candidate.response_coordinate != 'disk_velocity' or not candidate.conditioned:
            raise ValueError('output-conditioned disk-velocity model required')
        self._candidate, self._source_signs = candidate, source_signs
        self.conditioned = candidate.conditioned
        self.output_thresholds = candidate.output_thresholds

    def transformed_outputs(self, *args, **kwargs):
        return self._candidate.transformed_outputs(*args, **kwargs)

    def predict_pairs(self, pairs, radius, flux, **kwargs):
        coefficients = self._candidate.predict_pairs(pairs, radius, flux, **kwargs)
        return coefficients*self._source_signs.at(pairs['index_input_s'].to_numpy())


def signed_scene_draw_responses(candidate, pairs, primary_ids, physical_draws,
                                source_signs, *, pair_batch_size=200000):
    """Return sum_j b_ij(radius, flux)*s_j with exact output-cell pooling.

For a probe leg with orientation +/-1 along axis a, pass the velocity
``orientation*h*result[..., None]*unit_axis_a`` to disk_response_transport.
The sign belongs inside the pair sum, before radial_tanh. Multiplying the
coherent sum by the primary's sign would describe a different experiment.
The caller must use common latent draws across antithetic flow contexts.
"""
    source_signs.at(primary_ids)
    source_signs.at(pairs['index_input_s'].to_numpy())
    if (pairs['index_input_p'].to_numpy() == pairs['index_input_s'].to_numpy()).any():
        raise ValueError('self pairs are not neighbour response terms')
    return predict_scene_draw_responses(_SignedPairResponse(candidate, source_signs),
                                        pairs, primary_ids, physical_draws,
                                        pair_batch_size=pair_batch_size)
