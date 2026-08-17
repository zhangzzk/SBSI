import numpy as np
import pandas as pd

from scripts.analyze_halfshear_noiseless_pair_toys import (
    distribution,
    sign_comparison,
)
from scripts.prepare_halfshear_noiseless_pair_toys import (
    eligible_mask,
    keep_smallest_priority,
)
from scripts import run_halfshear_noiseless_pair_toy_shard as runner


def minimal_pair_frame():
    return pd.DataFrame({
        "case": [40, 39, 40, 40],
        "input_index": [1, 2, 3, 4],
        "RA_input_p": [180.0] * 4,
        "RA_input_s": [180.0001] * 4,
        "DEC_input_p": [0.0] * 4,
        "DEC_input_s": [0.0001] * 4,
        "Re_input_p": [0.6, 0.6, 0.5, 0.6],
        "Re_input_s": [0.3] * 4,
        "axis_ratio_input_p": [0.8] * 4,
        "axis_ratio_input_s": [0.7] * 4,
        "position_angle_input_p": [10.0] * 4,
        "position_angle_input_s": [20.0] * 4,
        "sersic_n_input_p": [1.0] * 4,
        "sersic_n_input_s": [1.5] * 4,
        "r_input_p": [24.0] * 4,
        "r_input_s": [25.0, 25.0, 25.0, np.nan],
        "distance": [1.0] * 4,
        "delta_et1": [0.1] * 4,
        "delta_et2": [0.0] * 4,
    })


def test_eligible_mask_uses_case_window_strict_cuts_and_finiteness():
    got = eligible_mask(minimal_pair_frame(), 40, 199)
    assert got.tolist() == [True, False, False, False]


def test_keep_smallest_priority_is_deterministic():
    frame = pd.DataFrame({
        "sample_priority": [0.8, 0.1, 0.4, 0.2],
        "identity": [8, 1, 4, 2],
    })
    got = keep_smallest_priority(frame, 2).sort_values("sample_priority")
    assert got.identity.tolist() == [1, 2]


def test_latent_offset_centers_primary_and_uses_sky_separation():
    row = pd.Series({
        "RA_input_p": 180.0,
        "RA_input_s": 180.0 + 1.0 / 3600.0,
        "DEC_input_p": 0.0,
        "DEC_input_s": 2.0 / 3600.0,
    })
    x, y, distance = runner.latent_offset(row)
    np.testing.assert_allclose([x, y, distance], [1.0, 2.0, np.sqrt(5.0)], atol=1e-9)


def test_eight_azimuth_grid_rotates_position_in_45_degree_steps():
    np.testing.assert_allclose(
        runner.azimuth_offsets_deg(8),
        [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0],
    )
    np.testing.assert_allclose(runner.rotate_offset(1.0, 2.0, 90.0), [-2.0, 1.0])
    x, y = runner.rotate_offset(1.0, 2.0, 45.0)
    np.testing.assert_allclose(np.hypot(x, y), np.sqrt(5.0), atol=1.0e-14)


def test_run_pair_averages_response_matrices_over_all_azimuths(monkeypatch):
    row = minimal_pair_frame().iloc[0].copy()
    row["sample_id"] = 7
    row["catalogue_row"] = 70
    row["R_emulator_v22"] = -0.01
    row["R_label_forward"] = -0.02
    row["R_label_null"] = 0.0

    monkeypatch.setattr(
        runner, "draw_source",
        lambda *args, **kwargs: np.zeros((2, 2), dtype=float),
    )

    def fake_compose(record, g, stamp, azimuth_offset_deg=0.0, primary=None):
        return {"offset": float(azimuth_offset_deg)}, {
            "native_dx_arcsec": 1.0,
            "native_dy_arcsec": 0.0,
            "rotated_dx_arcsec": 1.0,
            "rotated_dy_arcsec": 0.0,
            "latent_distance_arcsec": 1.0,
            "catalogue_minus_latent_distance_arcsec": 0.0,
        }

    def fake_response(images, g, fit_seed):
        value = images["offset"]
        return np.array([[value, 2.0 * value], [3.0 * value, 4.0 * value]])

    monkeypatch.setattr(runner, "compose_pair_arms", fake_compose)
    monkeypatch.setattr(runner, "response_matrix", fake_response)
    pair, rotations = runner.run_pair(row, 0.05, 112, 42, 8)
    mean_offset = np.mean(np.arange(8) * 45.0)
    np.testing.assert_allclose(
        [pair["R11_toy"], pair["R12_toy"], pair["R21_toy"], pair["R22_toy"]],
        [mean_offset, 2.0 * mean_offset, 3.0 * mean_offset, 4.0 * mean_offset],
    )
    assert pair["R_blend_toy"] == 2.5 * mean_offset
    assert pair["R_blend_toy_native"] == 0.0
    assert len(rotations) == 8
    assert [item["azimuth_offset_deg"] for item in rotations] == list(
        np.arange(8) * 45.0
    )


def test_response_matrix_uses_both_central_component_axes(monkeypatch):
    images = {
        "g1_plus": np.full((1, 1), 1.0),
        "g1_minus": np.full((1, 1), -1.0),
        "g2_plus": np.full((1, 1), 2.0),
        "g2_minus": np.full((1, 1), -2.0),
    }

    def fake_measure(image, fit_seed):
        value = float(image[0, 0])
        return np.array([value, 2.0 * value])

    monkeypatch.setattr(runner, "measured_shape", fake_measure)
    got = runner.response_matrix(images, g=0.5, fit_seed=42)
    np.testing.assert_allclose(got, [[2.0, 4.0], [4.0, 8.0]])
    assert 0.5 * np.trace(got) == 5.0


def test_distribution_and_sign_comparison_retain_negative_fraction():
    values = np.array([-2.0, -1.0, 1.0, 2.0])
    summary = distribution(values)
    assert summary["negative_fraction"] == 0.5
    assert summary["mean"] == 0.0
    assert summary["negative_part_mean_contribution"] == -0.75
    assert summary["positive_part_mean_contribution"] == 0.75
    assert summary["negative_thresholds"]["0.1"]["fraction"] == 0.5
    frame = pd.DataFrame({
        "R_blend_toy": values,
        "R_emulator_v22": [-1.0, 1.0, -1.0, 1.0],
        "R_label_forward": values[::-1],
    })
    signs = sign_comparison(frame)
    assert signs["counts"] == {
        "toy_negative_emulator_negative": 1,
        "toy_negative_emulator_nonnegative": 1,
        "toy_nonnegative_emulator_negative": 1,
        "toy_nonnegative_emulator_nonnegative": 1,
    }
    assert signs["sign_agreement_fraction"] == 0.5
