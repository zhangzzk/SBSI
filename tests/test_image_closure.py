from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sbsi.catalogue_closure import MockCatalogue
from sbsi.image_closure import blendemu_case_paths, build_image_mock
from sbsi.scene_prior import SHEAR_TRANSFORM


TARGETS = (
    "measured_ngmix_g1",
    "measured_ngmix_g2",
    "measured_mag_auto",
    "measured_log_flux_radius",
)


def _write_case(root: Path, *, case: int = 4, g1: float = 0.02):
    catalogues = root / f"case{case}_0.02" / "real0" / "catalogues"
    input_dir = catalogues / "input"
    match_dir = catalogues / "CrossMatch"
    shape_dir = catalogues / "Shapes"
    for path in (input_dir, match_dir, shape_dir):
        path.mkdir(parents=True, exist_ok=True)
    tile = "tile180.0_-0.5"
    pd.DataFrame(
        {
            "index_input": [0, 1, 2, 3],
            "r_input": [24.0, 26.0, 23.5, 24.5],
            "Re_input": [0.8, 0.9, 0.7, 1.0],
            "gamma1_input": [g1] * 4,
            "gamma2_input": [0.0] * 4,
        }
    ).to_feather(input_dir / f"gals_info_{tile}.feather")
    pd.DataFrame(
        {
            "id_input": [0, 1, 2, 3],
            "id_detec": [1, 2, 3, 4],
            "distance_pixel_CM": [0.2, 0.1, 0.3, 0.2],
        }
    ).to_feather(match_dir / f"{tile}_rot0_matched.feather")
    pd.DataFrame(
        {
            "NGMIX_G1": [0.1, 0.2, -1.0, -0.05],
            "NGMIX_G2": [0.0, 0.1, -1.0, 0.03],
            "MAG_AUTO": [24.1, 25.9, 23.8, 24.6],
            "FLUX_RADIUS": [2.0, 2.2, 1.8, 3.0],
        }
    ).to_feather(
        shape_dir / f"shape_catalogue_detect_position_all_{tile}.feather"
    )


def test_image_mock_maps_single_leg_products_and_records_source(tmp_path):
    _write_case(tmp_path)
    paths = blendemu_case_paths(tmp_path, case=4, shear_label="0.02")
    mock, report = build_image_mock(
        [paths],
        target_names=TARGETS,
        injected_g1=0.02,
        injected_g2=0.0,
        primary_mag_bounds=(18.0, 25.8),
        primary_re_bounds=(0.5, 1.5),
        sample_seed=7,
    )

    assert mock.kind == "image"
    assert mock.shear_transform == SHEAR_TRANSFORM
    assert len(mock.measurements) == 2
    assert set(mock.truth["source_input_index"]) == {0, 3}
    assert list(mock.measurements) == list(TARGETS)
    radius_by_input = dict(
        zip(
            mock.truth["source_input_index"],
            mock.measurements["measured_log_flux_radius"],
        )
    )
    assert radius_by_input[0] == pytest.approx(np.log(2.0))
    assert radius_by_input[3] == pytest.approx(np.log(3.0))
    assert report["n_available"] == 2
    assert report["cases"][0]["n_invalid_measurements"] == 1


def test_image_mock_rejects_mislabeled_noncoherent_shear(tmp_path):
    _write_case(tmp_path, g1=0.01)
    paths = blendemu_case_paths(tmp_path, case=4, shear_label="0.02")
    with pytest.raises(ValueError, match="input shear does not match"):
        build_image_mock(
            [paths],
            target_names=TARGETS,
            injected_g1=0.02,
            injected_g2=0.0,
            primary_mag_bounds=(18.0, 25.8),
            primary_re_bounds=(0.5, 1.5),
        )


def test_mock_kind_is_explicit_for_images_and_legacy_for_old_mocks():
    measurements = pd.DataFrame({"x": [0.0]})
    legacy = MockCatalogue(
        measurements,
        pd.DataFrame({"injected_g1": [0.0], "injected_g2": [0.0]}),
    )
    assert legacy.kind == "likelihood"
    image = MockCatalogue(
        measurements,
        pd.DataFrame(
            {
                "mock_kind": ["image"],
                "injected_g1": [0.0],
                "injected_g2": [0.0],
            }
        ),
    )
    assert image.kind == "image"
    with pytest.raises(ValueError, match="mock_kind"):
        MockCatalogue(
            pd.DataFrame({"x": [0.0, 1.0]}),
            pd.DataFrame(
                {
                    "mock_kind": ["image", "likelihood"],
                    "injected_g1": [0.0, 0.0],
                    "injected_g2": [0.0, 0.0],
                }
            ),
        )
