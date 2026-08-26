import numpy as np
import pandas as pd
import torch

from sbsi.detection_classifier import (
    PAIR_FRAME_SHAPE_FEATURES,
    build_detection_feature_frame,
    choose_representative_neighbours,
    detection_selection_response,
    discordant_transition_loss,
    log_neighbour_impact,
    render_catalogue_shape_shear,
)


CONDITIONS = {
    "pixel_size": 0.2,
    "zero_point": 30.0,
    "psf_fwhm": 0.73,
    "moffat_beta": 2.224,
    "pixel_rms": 0.312,
}


def _catalogue():
    arcsec = 1.0 / 3600.0
    return pd.DataFrame(
        {
            "index": [10, 11, 12, 13],
            "RA": [10.0, 10.0 + arcsec, 10.0 + 2 * arcsec, 11.0],
            "DEC": [0.0, 0.0, 0.0, 0.0],
            "redshift": [0.4, 0.5, 0.6, 0.7],
            "r": [24.0, 26.0, 21.0, 25.0],
            "Re": [0.5, 0.3, 0.9, 0.4],
            "sersic_n": [1.0, 1.5, 2.0, 2.5],
            "axis_ratio": [0.8, 0.7, 0.4, 0.9],
            "position_angle": [0.0, 0.2, 0.4, 0.6],
        }
    )


def test_exact_sampled_neighbour_search_can_choose_impact_over_nearest():
    nearest = choose_representative_neighbours(
        _catalogue(), [0, 3], radius_arcsec=3.0, neighbour_selection="nearest"
    )
    impact = choose_representative_neighbours(
        _catalogue(),
        [0, 3],
        radius_arcsec=3.0,
        neighbour_selection="impact",
        impact_exponent=2.0,
    )
    assert nearest.secondary_row.tolist() == [1, -1]
    assert impact.secondary_row.tolist() == [2, -1]
    assert nearest.candidate_count.tolist() == [2, 0]

    frame = build_detection_feature_frame(
        _catalogue(), impact, conditions=CONDITIONS
    )
    assert frame.index.tolist() == [0, 3]
    assert frame["input_index"].tolist() == [10, 13]
    assert np.isclose(frame.loc[0, "axis_ratio_input_s"], 0.4)
    assert np.isnan(frame.loc[3, "axis_ratio_input_s"])
    assert not frame.loc[3, "neighbored"]


def test_log_impact_is_flux_times_scaled_distance_up_to_a_constant():
    score = log_neighbour_impact(
        np.array([22.0, 23.0]),
        np.array([0.5, 1.0]),
        np.array([1.0, 2.0]),
        2.0,
    )
    # The size/distance factor is identical; the one-magnitude brighter source wins.
    assert np.isclose(score[0] - score[1], 0.4 * np.log(10.0))


def test_render_catalogue_shape_shear_composes_stored_shear_into_q():
    catalogue = _catalogue()
    catalogue["g1"] = [0.05, 0.0, 0.0, 0.0]
    catalogue["g2"] = 0.0
    rendered = render_catalogue_shape_shear(catalogue)
    assert not np.isclose(rendered.loc[0, "axis_ratio"], catalogue.loc[0, "axis_ratio"])
    np.testing.assert_allclose(
        rendered.loc[1:, "axis_ratio"], catalogue.loc[1:, "axis_ratio"]
    )
    np.testing.assert_allclose(rendered[["RA", "DEC"]], catalogue[["RA", "DEC"]])


def test_pair_frame_features_receive_the_rendered_primary_and_neighbour_shapes():
    catalogue = _catalogue()
    catalogue["g1"] = 0.0
    catalogue["g2"] = 0.0
    primary_rows = [0]
    choice = choose_representative_neighbours(
        catalogue, primary_rows, radius_arcsec=3.0, neighbour_selection="nearest"
    )
    zero = build_detection_feature_frame(catalogue, choice, conditions=CONDITIONS)
    catalogue.loc[[0, 1], "g1"] = 0.05
    rendered = render_catalogue_shape_shear(catalogue)
    sheared = build_detection_feature_frame(rendered, choice, conditions=CONDITIONS)

    assert set(PAIR_FRAME_SHAPE_FEATURES).issubset(sheared.columns)
    assert not np.allclose(
        zero[list(PAIR_FRAME_SHAPE_FEATURES)].to_numpy(dtype=float),
        sheared[list(PAIR_FRAME_SHAPE_FEATURES)].to_numpy(dtype=float),
    )


def test_centered_detection_response_is_zero_for_shear_invariant_selection():
    e0 = np.array([-0.3, -0.1, 0.2, 0.4])
    eg = e0 + 0.05
    weights = np.array([0.2, 0.8, 0.4, 0.7])
    response = detection_selection_response(
        e0, eg, weights, weights, np.full(4, 0.05)
    )
    assert abs(response) < 1.0e-12


def test_centered_detection_response_detects_changed_weighting():
    e0 = np.array([-0.3, -0.1, 0.2, 0.4])
    eg = e0 + 0.05
    response = detection_selection_response(
        e0,
        eg,
        np.ones(4),
        np.array([1.0, 1.0, 0.0, 0.0]),
        np.full(4, 0.05),
    )
    assert response < 0


def test_discordant_transition_loss_rewards_the_observed_flip_direction():
    y0 = torch.tensor([1.0, 0.0, 1.0])
    yg = torch.tensor([0.0, 1.0, 1.0])
    aligned, count = discordant_transition_loss(
        torch.tensor([2.0, -2.0, 100.0]),
        torch.tensor([-2.0, 2.0, -100.0]),
        y0,
        yg,
    )
    reversed_loss, reversed_count = discordant_transition_loss(
        torch.tensor([-2.0, 2.0, -100.0]),
        torch.tensor([2.0, -2.0, 100.0]),
        y0,
        yg,
    )
    assert count == reversed_count == 2
    assert aligned < reversed_loss


def test_discordant_transition_loss_can_exclude_input_identical_pairs():
    logits0 = torch.tensor([0.0, 1.0], requires_grad=True)
    logitsg = torch.tensor([0.0, -1.0], requires_grad=True)
    loss, count = discordant_transition_loss(
        logits0,
        logitsg,
        torch.tensor([1.0, 1.0]),
        torch.tensor([0.0, 0.0]),
        eligible=torch.tensor([False, False]),
    )
    assert count == 0
    assert loss.item() == 0.0
    loss.backward()
    assert torch.equal(logits0.grad, torch.zeros_like(logits0))
