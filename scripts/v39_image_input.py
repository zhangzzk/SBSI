"""Validate a fixed prefix of the existing randomly ordered ConstGold sample."""
import numpy as np

from sbsi.disk_precision import LOG_GAP_COLUMN, validate_shape_encoding


def selected_prefix(measurements, truth, manifest, count):
    """Reject incompatible input; never silently cut, clip, or reorder rows."""
    if (manifest['mock_kind'], manifest['catalogue'], manifest['leg']) != ('image', 'constgold', 'plus'):
        raise ValueError('require actual ConstGold plus-leg measurements')
    if manifest['primary_mag'] is not None or manifest['primary_re'] is not None:
        raise ValueError('truth-property cuts are prohibited')
    if manifest['sampling'] != 'uniform without replacement after independent plus-leg usability and selection':
        raise ValueError('require the existing randomly ordered uniform sample')
    if manifest['measured_selection_bounds'] != [
        'measured_flux_radius:>3.0:', 'measured_flux_from_mag_auto:>47.8630092322638:'
    ]:
        raise ValueError('incompatible measurement cuts')
    if len(measurements) != len(truth) or len(truth) != manifest['n_objects'] or not 0 < count <= len(truth):
        raise ValueError('unaligned or insufficient input rows')
    if truth.duplicated(['source_case', 'source_input_index']).any() or truth.duplicated(
        ['source_case', 'source_detection_id']
    ).any():
        raise ValueError('duplicate catalogue identities')
    observations = measurements.iloc[:count].copy().reset_index(drop=True)
    provenance = truth.iloc[:count].copy().reset_index(drop=True)
    columns = ['measured_ngmix_g1', 'measured_ngmix_g2', 'measured_flux_radius',
               'measured_flux_from_mag_auto', 'measured_mag_auto']
    values = observations[columns].to_numpy(dtype=float)
    norm2 = np.square(values[:, :2]).sum(1)
    valid = (np.isfinite(values).all(1) & (norm2 < 1) & (values[:, 2] > 3)
             & (values[:, 3] > 47.8630092322638) & (values[:, 4] < 25.8))
    if not valid.all():
        raise ValueError('input violates strict measured selection or disk usability')
    np.testing.assert_allclose(values[:, 4], 30-2.5*np.log10(values[:, 3]), atol=1.e-10, rtol=0)
    if not ((provenance.mock_kind == 'image').all()
            and (provenance.injected_g1 == .02).all() and (provenance.injected_g2 == 0).all()
            and (provenance.shear_transform == 'intrinsic_ellipticity_only_v1').all()):
        raise ValueError('incompatible image/shear provenance')
    # This is derived from the recorded interior Cartesian measurement. It does
    # not purport to recover extra precision from a rounded boundary value.
    gap = np.log1p(-norm2)
    if LOG_GAP_COLUMN in observations:
        np.testing.assert_array_equal(observations[LOG_GAP_COLUMN], gap)
    observations[LOG_GAP_COLUMN] = gap
    validate_shape_encoding(values[:, :2], gap)
    provenance['parent_sample_row'] = np.arange(count)
    return observations, provenance
