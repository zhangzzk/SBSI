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
    draw_moments,
    output_cuts,
    paired_draws,
    selected_draw_moments,
    sheared_inputs,
)
from sbsi.forward_catalogue import prepare_flow_inputs
from sbsi.shear_map import apply_shear_to_ellipticity


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


def inputs():
    return prepare_flow_inputs(scene())[list(V39_FEATURES)]


def test_far_galaxy_has_no_crowding():
    crowding = inputs()[["nbr_close_050", "nbr_close_100", "nbr_close_200"]]
    assert crowding.loc[3].eq(0).all()
    assert crowding.loc[:2].gt(0).all().all()


def test_sheared_inputs_moves_only_shapes():
    inputs_ = inputs()
    out = sheared_inputs(inputs_, 0.02)
    e1, e2 = apply_shear_to_ellipticity(inputs_["e1_input_p"].to_numpy(), inputs_["e2_input_p"].to_numpy(), 0.02, 0.0)
    np.testing.assert_array_equal(out["e1_input_p"], e1)
    np.testing.assert_array_equal(out["e2_input_p"], e2)
    pd.testing.assert_frame_equal(out.drop(columns=["e1_input_p", "e2_input_p"]),
                                  inputs_.drop(columns=["e1_input_p", "e2_input_p"]))


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


class ShapeFlow:
    """Measured shape = input shape + antithetic noise; size and flux pass every cut."""

    class model:
        target_dim, context_dim = 4, 2

        @staticmethod
        def forward_physical(z, context):
            size = torch.full_like(z[:, :2], 4.0)
            size[:, 1] = 2 * FLUX_MINIMUM
            return torch.cat((context + 0.1 * z[:, :2], size), dim=1), None

    @staticmethod
    def context_tensor(frame):
        return torch.tensor(frame[["e1_input_p", "e2_input_p"]].to_numpy(), dtype=torch.float64)


def test_draw_moments_without_emulator_give_the_sheared_flow_mean():
    inputs_ = inputs()
    mass, kept, blend = draw_moments(ShapeFlow(), inputs_, shear=0.02, n_draws=4, batch_size=2, random_seed=1)
    for leg, sign in enumerate((1, -1)):
        e1, _ = apply_shear_to_ellipticity(inputs_["e1_input_p"].to_numpy(), inputs_["e2_input_p"].to_numpy(),
                                           sign * 0.02, 0.0)
        np.testing.assert_allclose(kept[leg, :, 0, 0], e1, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(mass[..., :2], 1.0)
    np.testing.assert_array_equal(blend, 0.0)
    with pytest.raises(ValueError, match="increasing"):
        draw_moments(ShapeFlow(), inputs_[::-1], shear=0.02, n_draws=4, batch_size=2, random_seed=1)


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
