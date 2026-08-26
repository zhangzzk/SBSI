import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from sbsi.catalogue_closure import MockCatalogue
from sbsi.scene_prior import SHEAR_TRANSFORM
from _script_loader import load_script_module


MODULE = load_script_module("run_section5_numerical_recenter.py")

TARGETS = (
    "measured_ngmix_g1",
    "measured_ngmix_g2",
    "measured_mag_auto",
    "measured_log_flux_radius",
)


def test_mean_observed_shape_is_only_the_raw_two_component_mean():
    frame = pd.DataFrame(
        {
            "measured_ngmix_g1": [0.01, 0.03, -0.02],
            "measured_ngmix_g2": [-0.04, 0.02, 0.05],
            "measured_mag_auto": [23.0, 24.0, 25.0],
            "measured_log_flux_radius": [0.0, 0.1, 0.2],
        }
    )
    center, names = MODULE._mean_observed_shape(frame, TARGETS)
    np.testing.assert_allclose(center, [0.02 / 3.0, 0.01])
    assert names == ("measured_ngmix_g1", "measured_ngmix_g2")


def test_mean_observed_shape_rejects_nonfinite_or_missing_shape():
    frame = pd.DataFrame({"measured_ngmix_g1": [0.1], "measured_ngmix_g2": [np.nan]})
    with pytest.raises(ValueError, match="non-finite"):
        MODULE._mean_observed_shape(frame, TARGETS)
    with pytest.raises(ValueError, match="missing"):
        MODULE._mean_observed_shape(frame, ("measured_ngmix_g1",))


def test_calibrated_mean_observed_shape_applies_full_affine_response():
    frame = pd.DataFrame(
        {
            "measured_ngmix_g1": [0.13, 0.15],
            "measured_ngmix_g2": [-0.03, -0.01],
        }
    )
    response = ((2.0, 0.5), (-0.25, 1.5))
    offset = (0.01, -0.02)
    center, names, raw = MODULE._calibrated_mean_observed_shape(
        frame,
        TARGETS,
        offset=offset,
        response=response,
    )
    expected = np.linalg.solve(np.asarray(response), np.asarray([0.14, -0.02]) - np.asarray(offset))
    np.testing.assert_allclose(center, expected)
    np.testing.assert_allclose(raw, [0.14, -0.02])
    assert names == ("measured_ngmix_g1", "measured_ngmix_g2")


def test_matrix2_rejects_singular_response():
    np.testing.assert_allclose(MODULE._matrix2("1,2,3,4"), [[1, 2], [3, 4]])
    with pytest.raises(Exception, match="invertible"):
        MODULE._matrix2("1,2,2,4")


def test_full_ladder_report_uses_paired_full_2d_influences():
    score = np.array(
        [
            [[1.0, 2.0], [3.0, -1.0], [-2.0, 1.0]],
            [[1.2, 1.8], [2.8, -0.8], [-1.7, 0.7]],
        ]
    )
    information = np.broadcast_to(np.array([[[4.0, 0.5], [0.5, 3.0]]]), (2, 3, 2, 2)).copy()
    moments = SimpleNamespace(
        center=(0.02, -0.01),
        draw_ladder=(32, 64),
        ladder_score=score,
        ladder_information=information,
    )
    report = MODULE._full_ladder_one_step_report(moments)
    assert set(report) == {"32", "64"}
    assert report["32"]["positive_definite"]
    assert "previous_rung_shift" not in report["32"]
    first = np.asarray(moments.center) + np.linalg.solve(information[0].sum(axis=0), score[0].sum(axis=0))
    second = np.asarray(moments.center) + np.linalg.solve(information[1].sum(axis=0), score[1].sum(axis=0))
    np.testing.assert_allclose(report["32"]["estimate"], first)
    np.testing.assert_allclose(report["64"]["estimate"], second)
    np.testing.assert_allclose(report["64"]["previous_rung_shift"], second - first)
    assert np.isfinite(report["64"]["previous_rung_paired_standard_error"]).all()


def _image_mock(root: Path) -> MockCatalogue:
    mock = MockCatalogue(
        measurements=pd.DataFrame(
            [[0.1, -0.2, 24.0, 1.2]],
            columns=TARGETS,
        ),
        truth=pd.DataFrame(
            {
                "mock_kind": ["image"],
                "shear_transform": [SHEAR_TRANSFORM],
                "source_case": [40],
                "source_input_index": [7],
                "source_detection_id": [9],
                "injected_g1": [0.02],
                "injected_g2": [0.0],
            }
        ),
    )
    mock.save(root)
    manifest = {
        "mock_kind": "image",
        "shear_transform": SHEAR_TRANSFORM,
        "injected_g1": 0.02,
        "injected_g2": 0.0,
        "target_names": list(TARGETS),
        "measurement_model_sha256": "flow-hash",
        "output_sha256": MODULE._file_hashes(root, ("measurements.parquet", "truth.parquet")),
    }
    (root / "image_mock_manifest.json").write_text(json.dumps(manifest) + "\n")
    return mock


def _validate(root: Path, mock: MockCatalogue):
    return MODULE._validate_loaded_mock_manifest(
        root,
        mock,
        measurement_model_sha256="flow-hash",
        target_names=TARGETS,
    )


def test_image_mock_manifest_validation_accepts_complete_identity(tmp_path):
    mock = _image_mock(tmp_path)
    payload = _validate(tmp_path, mock)
    assert payload["injected_g1"] == pytest.approx(0.02)


def test_image_mock_manifest_validation_requires_manifest(tmp_path):
    mock = _image_mock(tmp_path)
    (tmp_path / "image_mock_manifest.json").unlink()
    with pytest.raises(RuntimeError, match="lacks image_mock_manifest"):
        _validate(tmp_path, mock)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("measurement_model_sha256", "other", "different measurement-flow"),
        ("target_names", list(reversed(TARGETS)), "target names"),
        ("shear_transform", "other", "shear transform"),
        ("injected_g1", 0.03, "injected shear"),
        (
            "output_sha256",
            {"measurements.parquet": "bad", "truth.parquet": "bad"},
            "files do not match",
        ),
    ),
)
def test_image_mock_manifest_validation_rejects_identity_mismatch(tmp_path, field, value, message):
    mock = _image_mock(tmp_path)
    manifest_path = tmp_path / "image_mock_manifest.json"
    payload = json.loads(manifest_path.read_text())
    payload[field] = value
    manifest_path.write_text(json.dumps(payload) + "\n")
    with pytest.raises(RuntimeError, match=message):
        _validate(tmp_path, mock)


def _likelihood_mock(root: Path):
    mock = MockCatalogue(
        measurements=pd.DataFrame(
            [[0.1, -0.2, 24.0, 1.2]],
            columns=TARGETS,
        ),
        truth=pd.DataFrame(
            {
                "scene_row": [3],
                "shear_transform": [SHEAR_TRANSFORM],
                "injected_g1": [0.02],
                "injected_g2": [0.0],
            }
        ),
    )
    mock.save(root)
    identity = {"scene": "scene-hash", "selection_cut_key": None}
    implementation = {"sbsi/catalogue_closure.py": "implementation-hash"}
    manifest = {
        "mock_kind": "likelihood",
        "generation_identity": identity,
        "implementation_sha256": implementation,
        "output_sha256": MODULE._file_hashes(root, ("measurements.parquet", "truth.parquet")),
    }
    (root / "likelihood_mock_manifest.json").write_text(json.dumps(manifest) + "\n")
    return mock, identity, implementation


def test_likelihood_mock_manifest_validation_accepts_complete_identity(tmp_path):
    mock, identity, implementation = _likelihood_mock(tmp_path)
    payload = MODULE._validate_loaded_likelihood_manifest(
        tmp_path,
        mock,
        generation_identity=identity,
        implementation_sha256=implementation,
    )
    assert payload["mock_kind"] == "likelihood"


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("missing", "lacks likelihood_mock_manifest"),
        ("identity", "different scene/model/selection"),
        ("implementation", "different implementation"),
        ("hash", "files do not match"),
    ),
)
def test_likelihood_mock_manifest_validation_rejects_mismatch(tmp_path, mutation, message):
    mock, identity, implementation = _likelihood_mock(tmp_path)
    path = tmp_path / "likelihood_mock_manifest.json"
    if mutation == "missing":
        path.unlink()
    else:
        payload = json.loads(path.read_text())
        if mutation == "identity":
            identity = {**identity, "scene": "other"}
        elif mutation == "implementation":
            implementation = {"sbsi/catalogue_closure.py": "other"}
        elif mutation == "hash":
            payload["output_sha256"]["truth.parquet"] = "bad"
            path.write_text(json.dumps(payload) + "\n")
    with pytest.raises(RuntimeError, match=message):
        MODULE._validate_loaded_likelihood_manifest(
            tmp_path,
            mock,
            generation_identity=identity,
            implementation_sha256=implementation,
        )
