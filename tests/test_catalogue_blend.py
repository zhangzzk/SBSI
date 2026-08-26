import numpy as np
import pandas as pd

from sbsi.catalogue_blend import CatalogueBlendResponse
from sbsi.forward_catalogue import EmulatorPairingConfig
from sbsi.scene_prior import ScenePrior


CONDITIONS = {
    "pixel_size": 0.2,
    "zero_point": 30.0,
    "psf_fwhm": 0.73,
    "moffat_beta": 2.224,
    "pixel_rms": 0.312,
}


def _prior():
    arcsec = 1.0 / 3600.0
    frame = pd.DataFrame(
        {
            "RA": [10.0, 10.0 + arcsec, 10.0 + 2 * arcsec],
            "DEC": [0.0, 0.0, 0.0],
            "redshift": [0.4, 0.5, 0.6],
            "r": [23.0, 24.0, 25.0],
            "Re": [0.7, 0.6, 0.8],
            "sersic_n": [1.0, 2.0, 3.0],
            "axis_ratio": [0.8, 0.7, 0.6],
            "position_angle": [0.0, 0.2, 0.4],
            "prior_weight": [1.0, 0.0, 1.0],
        }
    )
    return ScenePrior.from_catalogue(
        frame,
        guard_radius_arcsec=11.0,
        weight_column="prior_weight",
    )


def _config():
    return EmulatorPairingConfig(
        cuts=((18, 29), (18, 29), (0.1, 2), (0.1, 2), (0, 10)),
        r_max_arcsec=10.0,
        k=4,
        conditions=CONDITIONS,
    )


class FakeEmulator:
    def predict_on_pairs(self, pairs, task, rescaled):
        assert task == "response"
        assert rescaled
        out = pairs.copy()
        out["response"] = 0.1
        return out


def test_blend_response_is_evaluated_once_and_aligned_to_active_atoms(tmp_path):
    prior = _prior()
    response = CatalogueBlendResponse.from_emulator(
        prior,
        FakeEmulator(),
        config=_config(),
        metadata={"artifact": "fake"},
    )
    np.testing.assert_allclose(response.values, [0.2, 0.0, 0.2])
    assert response.report["n_response_pairs"] == 4
    assert response.report["n_active_atoms"] == 2

    response.save(tmp_path)
    loaded = CatalogueBlendResponse.load(tmp_path)
    np.testing.assert_array_equal(loaded.values, response.values)
    assert loaded.metadata == {"artifact": "fake"}
    assert loaded.report == response.report


def test_blend_shift_vanishes_at_zero_and_uses_exact_shape_displacement():
    prior = _prior()
    response = CatalogueBlendResponse(np.array([0.2, 0.0, -0.1]), metadata={}, report={})
    np.testing.assert_array_equal(response.shape_shift(prior.shear(0.0, 0.0)), 0.0)

    sheared = prior.shear(0.02, -0.01)
    shift = response.shape_shift(sheared)
    base = prior.shear(0.0, 0.0).flow_view(conditions=CONDITIONS)
    trial = sheared.flow_view(conditions=CONDITIONS)
    expected = response.values[:, None] * (
        trial[["e1_input_rot0_p", "e2_input_rot0_p"]].to_numpy()
        - base[["e1_input_rot0_p", "e2_input_rot0_p"]].to_numpy()
    )
    np.testing.assert_allclose(shift, expected, rtol=0, atol=1e-15)
