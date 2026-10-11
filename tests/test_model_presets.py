"""Model identity and loaders."""
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from sbsi import cli
from sbsi.models import get_model, load_disk_response, load_emulator

CONDITIONS = {"pixel_size": 0.2, "zero_point": 30.0, "psf_fwhm": 0.73,
              "moffat_beta": 2.224, "pixel_rms": 0.312}


def test_disk_loader_checks_both_pins_before_loading(tmp_path, monkeypatch):
    model_path, metadata_path = tmp_path / "model.json", tmp_path / "metadata.json"
    model_path.write_bytes(b"model")
    metadata_path.write_bytes(b"metadata")
    paths = replace(get_model("V3.9-like"), emulator_model=model_path, emulator_metadata=metadata_path,
                    emulator_sha256=sha256(b"model").hexdigest(),
                    emulator_metadata_sha256=sha256(b"metadata").hexdigest())
    calls = []
    monkeypatch.setitem(sys.modules, "sbsi.output_conditioned_response", SimpleNamespace(
        DiskResponseMoment=lambda *args, **kwargs: calls.append((args, kwargs)) or "loaded"))
    assert load_disk_response(paths, nthread=2) == "loaded"
    assert calls == [((tmp_path,), {"device": "cpu", "nthread": 2})]
    metadata_path.write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="hash mismatch"):
        load_disk_response(paths)
    assert len(calls) == 1
    metadata_path.write_bytes(b"metadata")
    model_path.write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="hash mismatch"):
        load_disk_response(paths)
    assert len(calls) == 1


def test_v39_pins_owner_selected_model(capsys):
    root = Path(__file__).resolve().parents[1]
    spec = json.loads((root / 'configs/models_v3_9_like.json').read_text())
    model = get_model('v3.9-LIKE')
    assert model.flow_sha256 == 'bd51dbbbeeb72763f6982a958cb8f3a0b2d3906d4f78935d9d959fa85a17797e'
    assert model.detection_classifier_sha256 == '231bbbdf1f885ab22ccfca2525ae46b6bdfd068474c42ebac24973feb861cb6a'
    assert model.emulator_sha256 == '194c1d11b67d6fd52039060270663e8e1cd610b478b9e9a66e34271701761fe7'
    assert spec['classifier']['training_seed'] == 20260913
    assert spec['flow']['feature_names'] == spec['classifier']['feature_names']
    assert len(spec['flow']['feature_names']) == 8
    assert spec['decision']['development_non_degradation_screen_pass'] is False
    assert spec['decision']['independent_acceptance'] is False
    for entry in (spec['replicate_classifier'], *spec['evidence'].values()):
        assert sha256((root / 'models' / entry['path']).read_bytes()).hexdigest() == entry['sha256']
    assert cli.main(['show-model', 'V3.9-like']) == 0
    assert json.loads(capsys.readouterr().out)['flow_sha256'] == model.flow_sha256


def test_v39_bundled_artifacts_load_and_predict_together():
    from sbsi.measurement_model import load_measurement_model
    from sbsi.models import load_detection_classifier
    from sbsi.output_conditioned_response import TRUTH_FEATURES

    model = get_model('V3.9-like')
    artifacts = (model.flow_checkpoint, model.detection_classifier,
                 model.emulator_model, model.emulator_metadata)
    if not all(path.is_file() for path in artifacts):
        pytest.skip('bundled V3.9-like artifacts are missing from this checkout')
    pytest.importorskip('xgboost')
    model.validate()
    flow = load_measurement_model(model.flow_checkpoint)
    classifier = load_detection_classifier(model)
    disk = load_disk_response(model)
    names = ['e1_input_p', 'e2_input_p', 'sersic_n_input_p', 'r_input_p',
             'circularized_Re_input_p', 'nbr_close_050', 'nbr_close_100', 'nbr_close_200']
    assert flow.condition_preprocessor.feature_names == classifier.preprocessor.feature_names == names
    assert flow.condition_preprocessor.add_missing_indicators
    assert not classifier.preprocessor.add_missing_indicators
    frame = pd.DataFrame([[.1, -.05, 1., 24., .4, .05, .1, .2]], columns=names)
    probability = classifier.predict_proba(frame)
    assert probability.shape == (1,) and np.all((probability >= 0) & (probability <= 1))
    import torch
    torch.manual_seed(7301)
    draws = flow.sample(frame, n_samples=4)
    assert draws.shape == (1, 4, 4) and np.isfinite(draws).all()
    assert np.all(np.square(draws[..., :2]).sum(-1) < 1)
    assert np.all(draws[..., 2:] >= 0)
    pairs = pd.DataFrame([[.6, .5, 24., 25., 1., 1., 2.]] * 4, columns=TRUTH_FEATURES)
    response = disk.predict_pairs(pairs, draws[0, :, 2], draws[0, :, 3])
    assert response.shape == (4,) and np.isfinite(response).all()


def test_v39_through_the_tutorial_api_shares_flow_draws():
    """R_flow from ResponsePredictor uses the same draws as the emulator's own prediction."""
    pytest.importorskip('xgboost')
    from sbsi import (EmulatorPairingConfig, ResponsePredictor, example_path, load_catalogue,
                      predict_blend_response, prepare_forward_catalogue)

    catalogue = load_catalogue(example_path('data', 'example_catalog.feather')).iloc[:300]
    model = get_model('V3.9-like')
    emulator = load_emulator(model, conditions=CONDITIONS, device='cpu', nthread=2)
    config = EmulatorPairingConfig.from_emulator(emulator)
    assert (config.r_max_arcsec, config.k) == (10.0, 20)
    prepared = prepare_forward_catalogue(catalogue, config=config)
    r_blend = predict_blend_response(emulator, prepared.emulator_pairs, n_samples=8, batch_size=128)
    prediction = ResponsePredictor.load(model, device='cpu').predict(
        prepared.flow_inputs, blend_response=r_blend, n_samples=8, batch_size=128)
    joint = emulator.predict(prepared.flow_inputs, prepared.emulator_pairs, n_draws=8, batch_size=128).per_galaxy()
    np.testing.assert_array_equal(prediction.flow, joint['R_flow'].to_numpy())
    np.testing.assert_array_equal(prediction.blend, joint['R_blend'].to_numpy())
    detection = emulator.predict_detection(catalogue.reset_index(drop=True))
    assert detection['detection_prob'].between(0, 1).all() and len(detection) == len(catalogue)
