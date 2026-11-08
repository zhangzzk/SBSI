import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

spec = importlib.util.spec_from_file_location('v39_image_input',
    Path(__file__).resolve().parents[1]/'scripts/v39_image_input.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def example():
    mag = np.array([19., 25., 24.])
    measured = pd.DataFrame(dict(measured_ngmix_g1=[0., .9, .1], measured_ngmix_g2=[0., 0., .2],
        measured_flux_radius=[4., 5., 4.], measured_flux_from_mag_auto=10**((30-mag)/2.5),
        measured_mag_auto=mag))
    truth = pd.DataFrame(dict(source_case=[40, 41, 40], source_input_index=[1, 1, 2],
        source_detection_id=[11, 11, 12], mock_kind='image', injected_g1=.02, injected_g2=0.,
        shear_transform='intrinsic_ellipticity_only_v1', true_mag=[29., 30., 28.]))
    manifest = dict(mock_kind='image', catalogue='constgold', leg='plus', primary_mag=None, primary_re=None,
        sampling='uniform without replacement after independent plus-leg usability and selection',
        measured_selection_bounds=['measured_flux_radius:>3.0:', 'measured_flux_from_mag_auto:>47.8630092322638:'],
        n_objects=3)
    return measured, truth, manifest


def test_preserves_prefix_order_zero_shape_and_no_truth_cut():
    measured, truth, manifest = example()
    actual, provenance = module.selected_prefix(measured, truth, manifest, 2)
    pd.testing.assert_frame_equal(actual[measured.columns], measured.iloc[:2])
    assert provenance.true_mag.tolist() == [29., 30.]
    assert provenance.parent_sample_row.tolist() == [0, 1]
    np.testing.assert_allclose(actual.measured_shape_log_disk_gap, [0., np.log1p(-.81)])


@pytest.mark.parametrize('column,value', [('measured_flux_radius', 3.), ('measured_ngmix_g1', 1.),
                                        ('measured_mag_auto', 25.8), ('measured_flux_from_mag_auto', np.nan)])
def test_rejects_invalid_measurement_instead_of_replacing_row(column, value):
    measured, truth, manifest = example()
    measured.loc[0, column] = value
    with pytest.raises(ValueError, match='strict measured selection'):
        module.selected_prefix(measured, truth, manifest, 2)


def test_rejects_truth_filtered_parent_and_duplicate_identity():
    measured, truth, manifest = example()
    manifest['primary_mag'] = [18, 26]
    with pytest.raises(ValueError, match='truth-property'):
        module.selected_prefix(measured, truth, manifest, 2)
    manifest['primary_mag'] = None
    truth.loc[2, 'source_input_index'] = 1
    with pytest.raises(ValueError, match='duplicate'):
        module.selected_prefix(measured, truth, manifest, 2)
