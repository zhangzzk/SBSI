import numpy as np
import pandas as pd
import pytest

from scripts.prepare_anchorblend_guarded_catalogues import (
    guarded_assignment,
    guarded_mask,
    materialize_simulation_support,
    sparse_anchor_layer,
    stable_anchor_directions,
)


def test_guarded_mask_excludes_anchor_and_far_source():
    # Near zero declination, 1 arcsec is exactly 1/3600 degree for this test.
    frame = pd.DataFrame({
        "RA": [0.0, 1.0 / 3600.0, 11.0 / 3600.0],
        "DEC": [0.0, 0.0, 0.0],
    })
    local, distance = guarded_mask(frame, np.array([True, False, False]), 10.0)
    assert local.tolist() == [False, True, False]
    assert np.allclose(distance, [0.0, 1.0, 11.0])


def test_guarded_assignment_returns_nearest_anchor_owner():
    frame = pd.DataFrame({
        "RA": np.array([0.0, 1.0, 20.0, 19.0]) / 3600.0,
        "DEC": np.zeros(4),
    })
    local, distance, owner = guarded_assignment(
        frame, np.array([True, False, True, False]), 5.0,
    )
    assert local.tolist() == [False, True, False, True]
    assert np.allclose(distance, [0.0, 1.0, 0.0, 1.0])
    assert owner.tolist() == [0, 0, 2, 2]


def test_stable_anchor_directions_are_unit_and_keyed():
    ids = np.array([10, 20, 30])
    u1, u2 = stable_anchor_directions(4, ids, 25876)
    v1, v2 = stable_anchor_directions(4, ids, 25876)
    np.testing.assert_array_equal(u1, v1)
    np.testing.assert_array_equal(u2, v2)
    np.testing.assert_allclose(u1 * u1 + u2 * u2, 1.0)
    w1, _ = stable_anchor_directions(5, ids, 25876)
    assert not np.array_equal(u1, w1)


def test_sparse_anchor_layers_are_disjoint_and_cover_candidates():
    frame = pd.DataFrame({
        "RA": np.array([0.0, 20.0, 40.0, 80.0]) / 3600.0,
        "DEC": np.zeros(4),
    })
    candidate = np.ones(4, dtype=bool)
    layer0 = sparse_anchor_layer(frame, candidate, 30.01, 0)
    layer1 = sparse_anchor_layer(frame, candidate, 30.01, 1)
    assert layer0.tolist() == [True, False, True, True]
    assert layer1.tolist() == [False, True, False, False]
    assert not np.any(layer0 & layer1)
    assert np.all(layer0 | layer1)
    assert not sparse_anchor_layer(frame, candidate, 30.01, 2).any()


def test_materialize_simulation_support_redirects_paths(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    output.mkdir()
    (source / "noise.csv").write_text("label,rms_r\n0,1\n")
    for shear in (0.05, -0.05):
        label = str(float(shear))
        (output / f"gals200_{label}.feather").touch()
        (source / f"sim_config_case200_{label}.ini").write_text(
            "[Paths]\n"
            f"out_dir = {source}/case200_{label}\n"
            "[GalInfo]\n"
            f"cata_file = {source}/gals200_{label}.feather\n"
        )

    materialize_simulation_support(str(source), str(output), [200], (0.05, -0.05))

    assert (output / "noise.csv").read_text() == "label,rms_r\n0,1\n"
    text = (output / "sim_config_case200_0.05.ini").read_text()
    assert str(source) not in text
    assert f"cata_file = {output}/gals200_0.05.feather" in text

    (output / "sim_config_case200_0.05.ini").write_text("different")
    with pytest.raises(RuntimeError, match="REFUSING differing"):
        materialize_simulation_support(str(source), str(output), [200], (0.05, -0.05))
