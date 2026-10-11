import numpy as np
import pandas as pd
import pytest

from sbsi.catalogue import load_catalogue
from sbsi.crowding import V39_FEATURES
from sbsi.disk_predictor import DiskResponsePrediction
from sbsi.domain import Domain
from sbsi.models import ModelPaths, get_model
from sbsi.paths import RELEASE_MODELS_ROOT, REPOSITORY_ROOT, example_path
from sbsi.forward_catalogue import (
    EmulatorPairingConfig,
    make_pair_catalogue,
    prepare_emulator_pairs,
    prepare_forward_catalogue,
)
from sbsi.response import MeasurementSample, ResponsePrediction, predict_blend_response

CONDITIONS = {
    "pixel_size": 0.2,
    "zero_point": 30.0,
    "psf_fwhm": 0.73,
    "moffat_beta": 2.224,
    "pixel_rms": 0.312,
}


def test_the_named_model_is_the_bundled_v39_path_preset():
    model = get_model("v3.9-LIKE")
    assert isinstance(model, ModelPaths)
    assert model.name == "V3.9-like"
    paths = (model.flow_checkpoint, model.detection_classifier,
             model.emulator_model, model.emulator_metadata)
    assert all(path.is_relative_to(RELEASE_MODELS_ROOT) for path in paths)
    assert not hasattr(model, "domain")
    with pytest.raises(KeyError, match="V3.9-like"):
        get_model("V3")


def test_checkout_resources_do_not_depend_on_working_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert example_path("data", "example_catalog.feather").is_file()
    assert (REPOSITORY_ROOT / "pyproject.toml").is_file()


def test_catalogue_loader_accepts_dataframe_and_external_feather(tmp_path):
    frame = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    copy = load_catalogue(frame, columns=["b"])
    assert copy.equals(frame[["b"]])
    assert copy is not frame
    path = tmp_path / "catalogue.feather"
    frame.to_feather(path)
    assert load_catalogue(path).equals(frame)


def test_domain_is_read_from_flow_metadata():
    cut = {"column": "r_input_p", "operator": "<", "value": 30.0}
    domain = Domain.from_flow_metadata({"population": {"truth_analysis_cut": cut}})
    assert domain.magnitude_max == 30.0
    assert domain.mask([24.0, 30.0], [0.8, 0.8]).tolist() == [True, False]
    with pytest.raises(ValueError, match="training cut"):
        Domain.from_flow_metadata({"population": {}})


def test_response_prediction_uses_sum_and_sbsi_bias_sign():
    prediction = ResponsePrediction(
        model="custom",
        index=np.arange(3),
        flow=np.array([0.75, 0.85, 0.95]),
        blend=np.array([0.1, 0.1, 0.1]),
    )
    np.testing.assert_allclose(prediction.total, [0.85, 0.95, 1.05])
    assert np.isclose(prediction.total_mean, 0.95)
    assert np.isclose(prediction.multiplicative_bias(0.95), 0.0)
    assert prediction.multiplicative_bias(1.0) > 0.0
    summary = prediction.summary()
    assert summary["R_total"] == pytest.approx(0.95)
    assert summary["R_total_err"] == pytest.approx(0.1 / np.sqrt(3))
    assert summary["R_blend_err"] == pytest.approx(0.0, abs=1e-15)


def _input_catalogue(case=(1, 1, 1)):
    arcsec = 1.0 / 3600.0
    return pd.DataFrame(
        {
            "RA": [10.0, 10.0 + arcsec, 10.0 + 2 * arcsec],
            "DEC": [0.0, 0.0, 0.0],
            "r": [23.0, 24.0, 25.0],
            "Re": [0.7, 0.6, 0.8],
            "sersic_n": [1.0, 2.0, 3.0],
            "axis_ratio": [0.8, 0.7, 0.6],
            "position_angle": [0.0, 0.2, 0.4],
            "case": list(case),
        }
    )


def _pairing_config():
    return EmulatorPairingConfig(r_max_arcsec=10, k=3, conditions=CONDITIONS)


def test_pairing_config_is_read_from_the_emulator():
    class Emulator:
        r_max_arcsec, k, conditions = 10.0, 20, CONDITIONS

    assert EmulatorPairingConfig.from_emulator(Emulator()) == EmulatorPairingConfig(10.0, 20, CONDITIONS)
    with pytest.raises(TypeError, match="pairing settings"):
        EmulatorPairingConfig.from_emulator(object())


def test_forward_pair_catalogue_is_sbs_owned_and_group_aware():
    catalogue = _input_catalogue(case=(1, 1, 2))
    pairs = make_pair_catalogue(
        catalogue,
        r_max_arcsec=10,
        k=3,
        group_column="case",
    )
    assert set(pairs["primary_row"]) == {0, 1}
    assert set(pairs["secondary_row"]) == {0, 1}
    assert not (pairs["primary_row"] == pairs["secondary_row"]).any()
    assert np.allclose(pairs["distance"], 1.0)


def test_prepared_pairs_match_blendemu_catalogue_conversion():
    # SBSI reimplements pair preparation so the library runs without BlendEMU; this is the
    # cross-check that the reimplementation still agrees with BlendEMU's own conversion. It
    # is therefore the one test that needs BlendEMU, and it skips when BlendEMU is absent
    # rather than failing -- the suite must pass on a checkout that has SBSI only.
    nz_utils = pytest.importorskip(
        "blendemu.nz_utils",
        reason="BlendEMU not on PYTHONPATH; skipping the cross-check against its converter",
    )

    catalogue = _input_catalogue().drop(columns="case").assign(redshift=0.5)
    config = _pairing_config()
    actual = prepare_emulator_pairs(catalogue, config=config)
    expected = nz_utils.icat2reg(
        catalogue,
        catalogue,
        model=None,
        conditions=config.conditions,
        apply_truth_cuts=False,  # SBSI keeps every pair
        r_max=config.r_max_arcsec,
        k=config.k,
    )
    columns = [
        "RA_input_p",
        "RA_input_s",
        "distance",
        "Re_input_p_scaled",
        "Re_input_s_scaled",
        "r_input_p_scaled",
        "r_input_s_scaled",
        "distance_scaled",
    ]
    np.testing.assert_allclose(actual[columns], expected[columns], rtol=0, atol=1e-12)


def test_forward_catalogue_builds_aligned_object_and_pair_views():
    catalogue = _input_catalogue(case=(1, 1, 2))
    prepared = prepare_forward_catalogue(catalogue, config=_pairing_config(), group_column="case")
    flow, pairs = prepared.flow_inputs, prepared.emulator_pairs
    features = list(V39_FEATURES)

    # The third galaxy is alone in its scene: no pair, so it is dropped.
    assert flow.index.name == "primary_row"
    assert flow.index.tolist() == [0, 1]
    assert {"r_input_p", "Re_input_p", *features} <= set(flow.columns)
    np.testing.assert_array_equal(flow[features], flow[features].astype(np.float32))
    assert flow.loc[0, "circularized_Re_input_p"] == np.float32(0.7 * np.sqrt(0.8))
    assert np.hypot(flow.loc[0, "e1_input_p"], flow.loc[0, "e2_input_p"]) == pytest.approx(0.2 / 1.8, rel=1e-6)
    assert (flow[["nbr_close_050", "nbr_close_100", "nbr_close_200"]] > 0).all().all()

    # Crowding never crosses scenes.
    same_scene = prepare_forward_catalogue(_input_catalogue(), config=_pairing_config(), group_column="case")
    assert same_scene.flow_inputs.loc[1, "nbr_close_200"] > flow.loc[1, "nbr_close_200"]

    # Every pair row carries its primary's flow inputs.
    carried = pairs.set_index("primary_row")[features]
    np.testing.assert_array_equal(carried.to_numpy(), flow.loc[carried.index, features].to_numpy())


def test_blend_response_runs_the_emulator_once_per_primary():
    prepared = prepare_forward_catalogue(_input_catalogue(), config=_pairing_config())

    class FakeEmulator:
        def predict(self, inputs, pairs, **kwargs):
            self.inputs, self.kwargs = inputs, kwargs
            n = len(inputs)
            blend = np.zeros((2, n, 3, 2))
            blend[0, :, :, 0] = 0.002 * np.arange(1, n + 1)[:, None]
            blend[1] = -blend[0]
            return DiskResponsePrediction("fake", inputs.index.to_numpy(), kwargs["shear"],
                                          np.ones((2, n, 3)), np.zeros((2, n, 3, 2)), blend,
                                          np.ones((2, n)))

    emulator = FakeEmulator()
    response = predict_blend_response(emulator, prepared.emulator_pairs)
    assert response.name == "R_blend"
    assert response.index.name == "primary_row"
    assert response.index.tolist() == [0, 1, 2]
    np.testing.assert_allclose(response, [0.1, 0.2, 0.3])
    pd.testing.assert_frame_equal(emulator.inputs, prepared.flow_inputs[list(V39_FEATURES)])
    assert emulator.kwargs == {"shear": 0.02, "n_draws": 64, "batch_size": 1024, "random_seed": 7301}


def test_measurement_sample_gives_magnitude_and_log_radius():
    names = ("measured_ngmix_g1", "measured_ngmix_g2", "measured_flux_radius", "measured_flux_from_mag_auto")
    samples = np.array([[[0.1, -0.2, np.e, 10 ** (-0.4 * (24 - 30))],
                         [0.0, 0.0, 1.0, 0.0]]])
    sample = MeasurementSample("m", np.array([5]), names, samples)
    np.testing.assert_allclose(sample.column("measured_log_flux_radius"), [[1.0, 0.0]])
    np.testing.assert_allclose(sample.column("measured_mag_auto"), [[24.0, np.inf]])
    np.testing.assert_array_equal(sample.column("measured_ngmix_g2"), [[-0.2, 0.0]])
    with pytest.raises(KeyError, match="measured_mag_auto"):
        sample.column("not_a_target")
