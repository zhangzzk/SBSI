import numpy as np
import pandas as pd
import torch

from scripts.evaluate_constgold_truth_parent import (
    FLUX_MINIMUM, measured_leg, output_cuts, selected_draw_moments,
)
from scripts.evaluate_constgold_fixed_g0_response import raw_root


def test_joint_selection_integrates_correlated_shape_not_product_of_means():
    values = torch.tensor([[[0.8, 0.2, 2.0, 100.0],
                            [0.2, 0.0, 4.0, 100.0]]], dtype=torch.float64)
    shift = torch.tensor([[0.03, -0.01]], dtype=torch.float64)
    mass, flow, blend = selected_draw_moments(values, shift)
    np.testing.assert_allclose(mass.numpy(), [[1, 0.5, 0.5]])
    np.testing.assert_allclose(flow[0, 1].numpy(), [0.1, 0.0])
    np.testing.assert_allclose(blend[0, 1].numpy(), [0.015, -0.005])
    np.testing.assert_allclose(((flow + blend) / mass[..., None])[0, 1].numpy(), [0.23, -0.01])
    assert flow[0, 1, 0] != values[..., 0].mean() * mass[0, 1]


def test_observed_and_generated_cut_boundaries_are_identical():
    values = np.array([[0, 0, 3, 100], [0, 0, 3.75, 100],
                       [0, 0, 4, FLUX_MINIMUM]], dtype=float)
    expected = [[True, False, False], [True, True, True], [True, False, False]]
    np.testing.assert_array_equal(output_cuts(values), expected)
    np.testing.assert_array_equal(output_cuts(torch.tensor(values)).numpy(), expected)


def test_output_conditioned_blend_preserves_selection_correlation():
    values = torch.tensor([[[0., 0., 2., 100.], [0., 0., 4., 100.]]], dtype=torch.float64)
    shifts = torch.tensor([[[.01, .02], [.07, -.04]]], dtype=torch.float64)
    mass, _, blend = selected_draw_moments(values, shifts)
    np.testing.assert_allclose(blend.numpy(), [[[.04, -.01], [.035, -.02], [.035, -.02]]])
    np.testing.assert_allclose((blend/mass[..., None])[0, 1], [.07, -.04])
    assert not torch.allclose(blend[0, 1], shifts.mean(1)[0]*mass[0, 1])


def test_raw_measurement_alignment_counts_missing_and_invalid_rows(tmp_path):
    root = raw_root(tmp_path, 40, 0.02)
    (root / "CrossMatch").mkdir(parents=True)
    (root / "Shapes").mkdir()
    pd.DataFrame({"id_input": [11, 12, 13], "id_detec": [1, 2, 3]}).to_feather(
        root / "CrossMatch" / "tile180.0_-0.5_rot0_matched.feather")
    pd.DataFrame({"NUMBER": [1, 2, 3, 4], "NGMIX_G1": [0.1, 0, 0.2, 0.1],
                  "NGMIX_G2": [0, 0, 0, 0], "MAG_AUTO": [25, 24, 25, 25],
                  "FLUX_RADIUS": [4, 4, -1, 4]}).to_feather(
        root / "Shapes" / "shape_catalogue_detect_position_all_tile180.0_-0.5.feather")
    values, usable, report = measured_leg(tmp_path, 40, 0.02, np.array([13, 12, 11, 99]))
    np.testing.assert_array_equal(usable, [False, True, True, False])
    assert np.isnan(values[-1]).all()
    assert report["parent_absent_crossmatch"] == 1
    assert report["usable_zero_shape_rows"] == 1
    assert report["shape_without_cross"] == 1
