"""Tests for the V3.9-like disk-model helpers that need no model files."""

import numpy as np
import pandas as pd
import pytest
import torch

from sbsi.crowding import V39_FEATURES
from sbsi.disk_predictor import (
    CUTS,
    FLUX_MINIMUM,
    DiskResponsePrediction,
    output_cuts,
    paired_draws,
    prepare_disk_inputs,
    selected_draw_moments,
    sheared_inputs,
)
from sbsi.shear_map import apply_shear_to_ellipticity

CONDITIONS = {"pixel_size": 0.2, "zero_point": 30.0, "psf_fwhm": 0.73,
              "moffat_beta": 2.224, "pixel_rms": 0.312}


def scene():
    arcsec = 1 / 3600
    return pd.DataFrame({
        "index": [7, 3, 5, 1],
        "RA": [180.0, 180.0 + 1.5 * arcsec, 180.0 + 3 * arcsec, 180.1],
        "DEC": [0.0, 0.0, 0.5 * arcsec, 0.0],
        "redshift": [0.5, 0.7, 0.9, 1.1],
        "r": [22.0, 23.5, 24.5, 25.0],
        "Re": [0.6, 0.4, 0.3, 0.2],
        "sersic_n": [1.0, 2.0, 1.5, 4.0],
        "axis_ratio": [0.5, 0.8, 0.9, 0.6],
        "position_angle": [10.0, 40.0, 80.0, 120.0],
    })


def test_prepare_disk_inputs_orders_and_scales():
    cat = scene()
    inputs, pairs = prepare_disk_inputs(cat, CONDITIONS)
    assert tuple(inputs.columns) == V39_FEATURES
    assert inputs.index.tolist() == [1, 3, 5, 7]
    np.testing.assert_array_equal(inputs.to_numpy(), inputs.to_numpy().astype(np.float32))
    row = cat.set_index("index").loc[7]
    q = row["axis_ratio"]
    assert inputs.loc[7, "circularized_Re_input_p"] == np.float32(row["Re"] * np.sqrt(q))
    assert np.hypot(inputs.loc[7, "e1_input_p"], inputs.loc[7, "e2_input_p"]) == pytest.approx((1 - q) / (1 + q), rel=1e-6)
    # The far galaxy has no neighbours; the close trio pairs among itself.
    assert inputs.loc[1, ["nbr_close_050", "nbr_close_100", "nbr_close_200"]].eq(0).all()
    assert 1 not in set(pairs["index_input_p"])
    assert len(pairs) == 6 and pairs["index_input_p"].is_monotonic_increasing
    assert {"r_input_p_scaled", "distance_scaled"} <= set(pairs.columns)


def test_sheared_inputs_moves_only_shapes():
    inputs, _ = prepare_disk_inputs(scene(), CONDITIONS)
    out = sheared_inputs(inputs, 0.02)
    e1, e2 = apply_shear_to_ellipticity(inputs["e1_input_p"].to_numpy(), inputs["e2_input_p"].to_numpy(), 0.02, 0.0)
    np.testing.assert_array_equal(out["e1_input_p"], e1)
    np.testing.assert_array_equal(out["e2_input_p"], e2)
    pd.testing.assert_frame_equal(out.drop(columns=["e1_input_p", "e2_input_p"]),
                                  inputs.drop(columns=["e1_input_p", "e2_input_p"]))


def test_output_cuts_and_moments():
    values = torch.tensor([[[0.1, 0.0, 3.0, 2 * FLUX_MINIMUM],
                            [0.3, 0.1, 3.5, 2 * FLUX_MINIMUM],
                            [0.5, 0.2, 3.75, 2 * FLUX_MINIMUM],
                            [0.7, 0.3, 5.0, 0.5 * FLUX_MINIMUM]]], dtype=torch.float64)
    keep = output_cuts(values)
    assert keep.shape == (1, 4, len(CUTS))
    assert keep[0, :, 1].tolist() == [False, True, True, False]
    assert keep[0, :, 2].tolist() == [False, False, True, False]
    shift = torch.full((1, 4, 2), 0.01, dtype=torch.float64)
    mass, flow, blend = selected_draw_moments(values, shift)
    np.testing.assert_allclose(mass[0].numpy(), [1.0, 0.5, 0.25])
    np.testing.assert_allclose(flow[0, :, 0].numpy(), [1.6 / 4, 0.8 / 4, 0.5 / 4])
    np.testing.assert_allclose(blend[0, :, 0].numpy(), 0.01 * mass[0].numpy())


class IdentityFlow(torch.nn.Module):
    target_dim, context_dim = 2, 1

    def forward_physical(self, z, context):
        return z + context, None


def test_paired_draws_share_antithetic_latents():
    contexts = [torch.zeros(3, 1, dtype=torch.float64), torch.ones(3, 1, dtype=torch.float64)]
    first, second = paired_draws(IdentityFlow(), contexts, 4, seed=11)
    np.testing.assert_array_equal(first[:, :2].numpy(), -first[:, 2:].numpy())
    np.testing.assert_allclose((second - first).numpy(), 1.0, rtol=0, atol=1e-12)
    again, _ = paired_draws(IdentityFlow(), contexts, 4, seed=11)
    np.testing.assert_array_equal(again.numpy(), first.numpy())
    with pytest.raises(ValueError):
        paired_draws(IdentityFlow(), contexts, 3, seed=11)


def prediction():
    rng = np.random.default_rng(3)
    n = 50
    mass = rng.uniform(0.2, 1.0, (2, n, 3))
    mass[:, :, 0] = 1.0
    return DiskResponsePrediction(
        model="test", index=np.arange(n), shear=0.02, mass=mass,
        flow=rng.normal(0, 0.3, (2, n, 3, 2)) * mass[..., None],
        blend=rng.normal(0, 0.01, (2, n, 3, 2)) * mass[..., None],
        p_usable=rng.uniform(0, 1, (2, n)))


def test_population_response_is_ratio_of_sums_per_leg():
    p = prediction()
    for cut, weight, w in (("main", None, np.ones((2, 50))), ("r075", "usable", p.p_usable)):
        c = CUTS.index(cut)
        expected = []
        for part in (p.flow, p.blend):
            ratio = (w * part[:, :, c, 0]).sum(1) / (w * p.mass[:, :, c]).sum(1)
            expected.append((ratio[0] - ratio[1]) / 0.04)
        np.testing.assert_allclose(p.response(cut, weight), expected, rtol=1e-12)
    galaxy = p.per_galaxy()
    np.testing.assert_allclose(galaxy["R_flow"].mean(), p.response("uncut")[0], rtol=1e-12)
    np.testing.assert_allclose(galaxy["R_total"], galaxy["R_flow"] + galaxy["R_blend"])


def test_summary_and_bias_carry_errors():
    p = prediction()
    table = p.summary(n_bootstrap=50)
    assert list(table.index) == ["uncut", "main cut", "main cut x usable probability"]
    assert (table[["R_flow_err", "R_blend_err", "R_model_err"]] > 0).all().all()
    np.testing.assert_allclose(table["R_model"], table["R_flow"] + table["R_blend"])
    model = table.loc["main cut x usable probability", "R_model"]
    m, sigma = p.multiplicative_bias(1.1 * model, n_bootstrap=50)
    assert m == pytest.approx(0.1) and sigma > 0
