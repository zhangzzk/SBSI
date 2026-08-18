import numpy as np
import pandas as pd
import pytest
import torch

from sbsi.catalogue import load_catalogue
from sbsi.domain import Domain
from sbsi.flow import FlowTrainingConfig
from sbsi.flow_training import parse_args as parse_training_args
from sbsi.measurement_model import (
    ConditionalMeanFlow,
    MeasurementModelBundle,
    TargetStandardizer,
)
from sbsi.models import ModelPaths, SHAPE_SEEDS, get_model
from sbsi.paths import RELEASE_MODELS_ROOT, REPOSITORY_ROOT, example_path
from sbsi.forward_catalogue import (
    EmulatorPairingConfig,
    make_pair_catalogue,
    prepare_emulator_pairs,
    prepare_forward_catalogue,
)
from sbsi.posterior_shape import PosteriorShapeEstimator, make_e_grid
from sbsi.response import ResponsePrediction, predict_blend_response
from sbsi.selection_model import TabularPreprocessor


def test_named_models_are_path_presets_not_pipeline_configuration():
    v3 = get_model("v3")
    v3b = get_model("V3B")
    assert isinstance(v3, ModelPaths)
    assert len(v3.flow_checkpoints) == len(v3b.flow_checkpoints) == 16
    assert SHAPE_SEEDS == (501, 502, 503, *range(505, 518))
    assert not hasattr(v3, "domain")
    assert not hasattr(v3, "blend_lookup")
    assert not hasattr(v3, "evaluation_result")
    assert v3.emulator_metadata.name == "emulator_metadata_lsst_r_extnbr_v22.json"


def test_checkout_resources_do_not_depend_on_working_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert example_path("data", "example_catalog.feather").is_file()
    assert get_model("V3").emulator_model.is_file()
    assert (REPOSITORY_ROOT / "pyproject.toml").is_file()
    assert get_model("V3").flow_checkpoints[0].is_relative_to(RELEASE_MODELS_ROOT)


def test_training_recipe_uses_only_explicit_user_paths(tmp_path):
    catalogue = tmp_path / "train.feather"
    response = tmp_path / "response.npz"
    coupling = tmp_path / "coupling.npz"
    for path in (catalogue, response, coupling):
        path.touch()
    config = FlowTrainingConfig(
        catalogue=catalogue,
        output=tmp_path / "flow.pt",
        response_target=response,
        coupling_target=coupling,
        seed=507,
        primary_magnitude_max=25.4,
        primary_half_light_radius_min=0.42,
    )
    argv = config.to_argv()
    assert argv[argv.index("--catalogue") + 1] == str(catalogue)
    assert argv[argv.index("--primary-mag-max") + 1] == "25.4"
    assert argv[argv.index("--primary-re-min") + 1] == "0.42"
    assert argv[argv.index("--response-target-npz") + 1] == str(response)
    assert argv[argv.index("--coupling-target-npz") + 1] == str(coupling)
    parsed = parse_training_args(argv)
    assert parsed.catalogue == str(catalogue)
    config.validate()


def test_catalogue_loader_accepts_dataframe_and_external_feather(tmp_path):
    frame = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    copy = load_catalogue(frame, columns=["b"])
    assert copy.equals(frame[["b"]])
    assert copy is not frame
    path = tmp_path / "catalogue.feather"
    frame.to_feather(path)
    assert load_catalogue(path).equals(frame)


def test_domain_is_read_from_flow_metadata():
    domain = Domain.from_flow_metadata(
        {"selection_cuts": [[18, 28], [18, 25.8], [0.1, 1.5], [0.5, 1.5], [0, 5]]}
    )
    assert domain == Domain(18.0, 25.8, 0.5, 1.5)
    assert domain.mask([24.0, 25.8], [0.8, 0.8]).tolist() == [True, False]


def test_response_prediction_uses_sum_and_sbsi_bias_sign():
    prediction = ResponsePrediction(
        model="custom",
        index=np.arange(3),
        flow_by_seed=np.array([[0.7, 0.8, 0.9], [0.8, 0.9, 1.0]]),
        blend=np.array([0.1, 0.1, 0.1]),
    )
    np.testing.assert_allclose(prediction.total, [0.85, 0.95, 1.05])
    assert np.isclose(prediction.total_mean, 0.95)
    assert np.isclose(prediction.multiplicative_bias(0.95), 0.0)
    assert prediction.multiplicative_bias(1.0) > 0.0


def _input_catalogue(case=(1, 1, 1)):
    arcsec = 1.0 / 3600.0
    return pd.DataFrame(
        {
            "RA": [10.0, 10.0 + arcsec, 10.0 + 2 * arcsec],
            "DEC": [0.0, 0.0, 0.0],
            "redshift": [0.4, 0.5, 0.6],
            "r": [23.0, 24.0, 25.0],
            "Re": [0.7, 0.6, 0.8],
            "sersic_n": [1.0, 2.0, 3.0],
            "axis_ratio": [0.8, 0.7, 0.6],
            "position_angle": [0.0, 0.2, 0.4],
            "case": list(case),
        }
    )


def _pairing_config():
    return EmulatorPairingConfig(
        cuts=((18, 29), (18, 29), (0.1, 2), (0.1, 2), (0, 10)),
        r_max_arcsec=10,
        k=3,
        conditions={
            "pixel_size": 0.2,
            "zero_point": 30.0,
            "psf_fwhm": 0.73,
            "moffat_beta": 2.224,
            "pixel_rms": 0.312,
        },
    )


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

    catalogue = _input_catalogue().drop(columns="case")
    config = _pairing_config()
    actual = prepare_emulator_pairs(catalogue, config=config)
    expected = nz_utils.icat2reg(
        catalogue,
        catalogue,
        model=None,
        conditions=config.conditions,
        cuts=config.cuts,
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


def test_blendemu_pair_predictions_are_summed_and_aligned_by_primary():
    class FakeEmulator:
        def predict_on_pairs(self, pairs, task, rescaled):
            assert task == "response"
            assert rescaled
            assert "distance_scaled" in pairs
            out = pairs.copy()
            out["response"] = 0.1
            return out

    catalogue = _input_catalogue()
    emulator = FakeEmulator()
    pairs = prepare_emulator_pairs(
        catalogue,
        config=_pairing_config(),
        group_column="case",
    )
    response = predict_blend_response(emulator, pairs)
    assert response.index.name == "primary_row"
    np.testing.assert_allclose(response, [0.2, 0.2, 0.2])


def test_forward_catalogue_builds_aligned_object_and_pair_views():
    catalogue = _input_catalogue()
    prepared = prepare_forward_catalogue(
        catalogue,
        config=_pairing_config(),
        group_column="case",
    )

    assert prepared.flow_inputs.index.name == "primary_row"
    assert prepared.flow_inputs.index.tolist() == [0, 1, 2]
    assert set(prepared.emulator_pairs["primary_row"]) == {0, 1, 2}
    required = {
        "e1_input_rot0_p",
        "e2_input_rot0_p",
        "nbr_flux_near",
        "nbr_flux_far",
        "nbr_flux_max",
        "r_input_p",
        "Re_input_p",
    }
    assert required.issubset(prepared.flow_inputs.columns)
    assert (prepared.flow_inputs["nbr_flux_near"] > 0).all()
    np.testing.assert_allclose(prepared.flow_inputs["nbr_flux_far"], 0.0)
    assert np.isfinite(
        prepared.flow_inputs[["e1_input_rot0_p", "e2_input_rot0_p"]]
    ).all().all()


def test_posterior_grid_accepts_four_output_flow():
    torch.manual_seed(2)
    frame = pd.DataFrame(
        {
            "e1_input_p": [-0.2, 0.0, 0.2],
            "e2_input_p": [0.1, -0.1, 0.0],
            "scene": [0.5, 0.6, 0.7],
            "measured_ngmix_g1": [0.02, 0.01, -0.01],
            "measured_ngmix_g2": [0.00, -0.02, 0.01],
            "measured_mag_auto": [24.0, 24.5, 25.0],
            "measured_log_flux_radius": [-0.2, -0.1, 0.0],
        }
    )
    preprocessor = TabularPreprocessor.fit(
        frame, ["e1_input_p", "e2_input_p", "scene"], add_missing_indicators=True
    )
    target_names = [
        "measured_ngmix_g1",
        "measured_ngmix_g2",
        "measured_mag_auto",
        "measured_log_flux_radius",
    ]
    targets = TargetStandardizer.fit(frame, target_names)
    model = ConditionalMeanFlow(
        target_dim=4,
        context_dim=preprocessor.output_dim,
        base_flow="affine",
        mean_hidden=8,
        hidden_dim=8,
        n_layers=1,
        n_flows=2,
        flow_drop_indices=(0, 1, 3, 4),
    )
    bundle = MeasurementModelBundle(model, preprocessor, targets)
    grid, _ = make_e_grid(n=7, emax=0.8, rmax=0.8)
    estimator = PosteriorShapeEstimator(bundle, grid)
    observed = frame[target_names].to_numpy(float)
    likelihood = estimator.log_likelihood(frame, observed, chunk=2, out_dtype=np.float32)
    mean, evidence = estimator.posterior_mean(likelihood, np.zeros(len(grid)))
    assert likelihood.shape == (len(frame), len(grid))
    assert mean.shape == (len(frame), 2)
    assert np.isfinite(mean).all() and np.isfinite(evidence).all()


def test_sample_measurement_pools_ensemble_draws(tmp_path):
    from sbsi.measurement_model import (
        TargetStandardizer,
        build_flow,
        save_measurement_model,
    )
    from sbsi.preprocessing import rescale
    from sbsi.response import sample_measurement

    catalogue = pd.DataFrame({
        "RA": [10.0, 10.0 + 0.5 / 3600.0],
        "DEC": [0.0, 0.0],
        "redshift": [0.5, 0.6],
        "r": [22.0, 23.0],
        "Re": [0.8, 0.7],
        "sersic_n": [1.5, 2.0],
        "axis_ratio": [0.7, 0.6],
        "position_angle": [0.3, 1.1],
    })
    config = EmulatorPairingConfig(
        cuts=((13.0, 29.0), (18.0, 25.8), (0.0, 10.0), (0.5, 1.5), (0.0, 5.0)),
        r_max_arcsec=5.0,
        k=3,
        conditions={"pixel_size": 0.2, "zero_point": 30.0, "psf_fwhm": 0.73,
                    "moffat_beta": 2.224, "pixel_rms": 0.312},
    )
    prepared = prepare_forward_catalogue(catalogue, config=config)

    features = ["e1_input_p", "e2_input_p", "sersic_n_input_p",
                "r_input_p_scaled", "Re_input_p_scaled"]
    preprocessor = TabularPreprocessor.fit(
        rescale(prepared.flow_inputs.copy()), features, add_missing_indicators=False
    )
    targets = ["measured_ngmix_g1", "measured_ngmix_g2",
               "measured_mag_auto", "measured_log_flux_radius"]
    rng = np.random.default_rng(4)
    standardizer = TargetStandardizer.fit(
        pd.DataFrame({name: rng.normal(size=len(prepared.flow_inputs)) for name in targets}),
        targets,
    )
    model_config = {"flow_type": "affine", "target_dim": 4,
                    "context_dim": len(features), "hidden_dim": 8,
                    "n_layers": 1, "n_flows": 2}
    metadata = {"selection_cuts": [[18, 28], [18, 25.8], [0.1, 1.5], [0.5, 1.5], [0, 5]]}
    checkpoints = []
    for seed in (1, 2):
        torch.manual_seed(seed)
        model = build_flow(model_config)
        path = tmp_path / f"flow_s{seed}.pt"
        save_measurement_model(path, model, preprocessor, standardizer,
                               model_config, metadata)
        checkpoints.append(path)

    sample = sample_measurement(checkpoints, prepared.flow_inputs,
                                n_samples=8, random_seed=7)
    n_objects = len(prepared.flow_inputs)
    assert sample.target_names == tuple(targets)
    assert sample.samples.shape == (n_objects, 2 * 8, 4)
    assert sample.column("measured_mag_auto").shape == (n_objects, 16)
    assert sample.index.tolist() == prepared.flow_inputs.index.tolist()

    repeat = sample_measurement(checkpoints, prepared.flow_inputs,
                                n_samples=8, random_seed=7)
    np.testing.assert_array_equal(sample.samples, repeat.samples)
    with pytest.raises(KeyError):
        sample.column("not_a_target")
