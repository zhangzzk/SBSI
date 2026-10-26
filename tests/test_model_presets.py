"""Identity, backend separation, and physical crowding invariants."""
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
from sbsi.crowding import FEATURES, SCALES, classifier_features, major_convolved_scale
from sbsi.models import get_model, load_disk_response, load_emulator


def test_v36_is_single_original_model_not_replica_or_v35_alias(capsys):
    model = get_model("v3.6-LIKE")
    spec = json.loads((Path(__file__).resolve().parents[1] / "configs/models_v3_6_like.json").read_text())
    assert model.flow_sha256s == ("5434cfafb94ecb169b8b6a012752191bb191b82a787b196de8ed6dac2f1501a5",)
    assert model.detection_classifier_sha256s == (spec["classifier"]["sha256"],)
    assert len(model.detection_classifiers) == len(model.flow_checkpoints) == 1
    assert model.response_backend == "disk_response_moment"
    assert model.emulator_sha256 == "194c1d11b67d6fd52039060270663e8e1cd610b478b9e9a66e34271701761fe7"
    assert get_model("V3.5-like").response_backend == "blendemu"
    assert model.flow_sha256s != get_model("V3.5-like").flow_sha256s
    assert spec["independent_validation"]["result_available"] is False
    cli.main(["show-model", "V3.6-like"])
    assert json.loads(capsys.readouterr().out)["response_backend"] == "disk_response_moment"


def test_disk_predictor_cannot_enter_additive_interface():
    with pytest.raises(ValueError, match="cannot be used as a fixed additive"):
        load_emulator(get_model("V3.6-like"), conditions={})
    with pytest.raises(ValueError, match="disk_response_moment"):
        load_disk_response(get_model("V3.5-like"))


def test_disk_loader_checks_both_pins_before_loading(tmp_path, monkeypatch):
    model_path, metadata_path = tmp_path / "model.json", tmp_path / "metadata.json"
    model_path.write_bytes(b"model")
    metadata_path.write_bytes(b"metadata")
    paths = replace(get_model("V3.6-like"), emulator_model=model_path, emulator_metadata=metadata_path,
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


def test_all_seventeen_features_follow_shape_and_preserve_parent_order():
    ids = np.array([9, 3])
    context = np.array([[.6, 0, 1, 25, .4, 0, 0, 0], [0, 0, 1, 24, .3, 0, 0, 0]], dtype=np.float32)
    initial = major_convolved_scale(context[:, 4], context[:, :2], .5)
    pairs = pd.DataFrame({"index_input_p": [9], "index_input_s": [2],
                          "r_input_p_scaled": [25.], "r_input_s_scaled": [25.],
                          "distance_scaled": [.49], "Re_input_p_scaled": [np.sqrt(1-(.5/initial[0])**2)]})
    before = pairs.copy(deep=True)
    target_shape = np.zeros((2, 2))
    result = classifier_features(ids, context, target_shape, pairs, .5)
    assert result.shape == (2, len(FEATURES)) == (2, 17)
    np.testing.assert_array_equal(result[:, :2], target_shape)
    np.testing.assert_array_equal(result[:, 2:8], context[:, 2:8])
    target = major_convolved_scale(context[:, 4], target_shape, .5)
    # Same physical separation moves from the innermost annulus to [0.5, 1).
    np.testing.assert_allclose(result[0, 8:13], [0, np.log(2), 0, 0, 0], rtol=1e-7)
    expected = np.log1p(np.exp(-.5*(.49*initial[0]/target[0]/np.array(SCALES))**2))
    np.testing.assert_allclose(result[0, 13:], expected, rtol=1e-6)
    np.testing.assert_array_equal(result[1, 8:], np.zeros(9))
    pd.testing.assert_frame_equal(pairs, before)


def test_model_manifest_is_not_an_inference_configuration():
    from _script_loader import load_script_module

    runner = load_script_module("run_inference.py")
    path = Path(__file__).resolve().parents[1] / "configs/models_v3_6_like.json"
    with pytest.raises(ValueError, match="schema_version"):
        runner._load_release_config(path, kind="likelihood")


def test_v39_pins_owner_selected_model_and_preserves_evidence(capsys):
    root = Path(__file__).resolve().parents[1]
    spec = json.loads((root / 'configs/models_v3_9_like.json').read_text())
    model = get_model('v3.9-LIKE')
    assert model.flow_sha256s == ('bd51dbbbeeb72763f6982a958cb8f3a0b2d3906d4f78935d9d959fa85a17797e',)
    assert model.detection_classifier_sha256s == ('231bbbdf1f885ab22ccfca2525ae46b6bdfd068474c42ebac24973feb861cb6a',)
    assert model.emulator_sha256 == get_model('V3.6-like').emulator_sha256
    assert model.flow_sha256s != get_model('V3.6-like').flow_sha256s
    assert len(model.detection_classifiers) == 1
    assert spec['classifier']['training_seed'] == 20260913
    assert spec['flow']['feature_names'] == spec['classifier']['feature_names']
    assert len(spec['flow']['feature_names']) == 8
    assert spec['decision']['development_non_degradation_screen_pass'] is False
    assert spec['decision']['independent_acceptance'] is False
    for receipt in spec['evidence'].values():
        assert sha256((root / receipt['path']).read_bytes()).hexdigest() == receipt['sha256']
    comparison = json.loads((root / spec['evidence']['comparison']['path']).read_text())
    assert spec['results']['primary'] == comparison['summaries']['C8_20260913']
    assert spec['results']['classifier_seed_replicate'] == comparison['summaries']['C8_20260914']
    assert cli.main(['show-model', 'V3.9-like']) == 0
    assert json.loads(capsys.readouterr().out)['flow_sha256s'] == list(model.flow_sha256s)
    with pytest.raises(ValueError, match='cannot be used as a fixed additive'):
        load_emulator(model, conditions={})


def test_v39_external_artifacts_load_and_predict_together():
    from sbsi.measurement_model import load_measurement_model
    from sbsi.models import load_detection_classifier
    from sbsi.output_conditioned_response import TRUTH_FEATURES

    model = get_model('V3.9-like')
    artifacts = (*model.flow_checkpoints, *model.detection_classifiers,
                 model.emulator_model, model.emulator_metadata)
    if not all(path.is_file() for path in artifacts):
        pytest.skip('external V3.9-like artifact roots are not configured')
    pytest.importorskip('xgboost')
    model.validate()
    flow = load_measurement_model(model.flow_checkpoints[0])
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
    response = disk.predict_pairs(pairs, draws[0, :, 2], draws[0, :, 3], allow_zero_limit=True)
    assert response.shape == (4,) and np.isfinite(response).all()
