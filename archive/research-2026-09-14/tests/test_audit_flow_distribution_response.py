from types import SimpleNamespace

import numpy as np
import pandas as pd

from scripts.audit_flow_distribution_response import coordinate_affine, observed_coordinates


def test_observed_joint_coordinates_use_measured_shapes_and_log_radius_each_leg():
    frame = pd.DataFrame({
        'measured_ngmix_g1': [.2], 'measured_ngmix_g2': [-.3],
        'measured_flux_radius': [np.exp(1.2)],
        'measured_ngmix_g1_sheared': [.4], 'measured_ngmix_g2_sheared': [-.1],
        'measured_flux_radius_sheared': [np.exp(1.7)],
    })
    np.testing.assert_allclose(observed_coordinates(frame, [0, 1, 3]), [[.2, -.3, 1.2]])
    np.testing.assert_allclose(observed_coordinates(frame, [0, 1, 3], '_sheared'), [[.4, -.1, 1.7]])
    np.testing.assert_allclose(observed_coordinates(frame, [3]), [[1.2]])


def test_different_model_standardizations_map_to_same_reference_joint_values():
    reference = SimpleNamespace(means=[.1, .2, 24., 1.5], scales=[.2, .3, 2., .4])
    model = SimpleNamespace(means=[-.1, .4, 20., 1.1], scales=[.5, .6, 1., .8])
    indices = [0, 1, 3]
    physical = np.array([[.3, -.2, 2.1], [-.5, .8, .7]])
    standardized = (physical-np.array(model.means)[indices])/np.array(model.scales)[indices]
    scale, offset = coordinate_affine(model, reference, indices)
    expected = (physical-np.array(reference.means)[indices])/np.array(reference.scales)[indices]
    np.testing.assert_allclose(standardized*scale+offset, expected)
