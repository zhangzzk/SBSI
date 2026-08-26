import numpy as np
import pandas as pd

from sbsi.forward_catalogue import EmulatorPairingConfig, prepare_emulator_pairs
from sbsi.scene_prior import ScenePrior


CONDITIONS = {
    "pixel_size": 0.2,
    "zero_point": 30.0,
    "psf_fwhm": 0.73,
    "moffat_beta": 2.224,
    "pixel_rms": 0.312,
}


def _catalogue():
    arcsec = 1.0 / 3600.0
    return pd.DataFrame(
        {
            "RA": [10.0, 10.0 + 2.9 * arcsec, 10.0 + 3.1 * arcsec, 10.0 + 8.0 * arcsec],
            "DEC": [0.0, 0.0, 0.0, 0.0],
            "redshift": [0.4, 0.5, 0.6, 0.7],
            "r": [23.0, 24.0, 25.0, 26.0],
            "Re": [0.7, 0.6, 0.8, 0.5],
            "sersic_n": [1.0, 2.0, 3.0, 1.5],
            "axis_ratio": [1.0, 0.7, 0.6, 0.8],
            "position_angle": [0.0, 0.2, 0.4, 0.6],
        }
    )


def test_scene_transform_changes_only_intrinsic_ellipticity():
    prior = ScenePrior.from_catalogue(_catalogue(), guard_radius_arcsec=12.0)
    sheared = prior.shear(0.05, 0.0)

    expected_q = (1.0 - 0.05) / (1.0 + 0.05)
    assert np.isclose(sheared.galaxies.loc[0, "axis_ratio"], expected_q)
    np.testing.assert_array_equal(sheared.galaxies["Re"], _catalogue()["Re"])
    np.testing.assert_array_equal(sheared.galaxies["r"], _catalogue()["r"])
    np.testing.assert_array_equal(sheared.dx_arcsec, prior.dx_arcsec)
    np.testing.assert_array_equal(sheared.dy_arcsec, prior.dy_arcsec)


def test_model_views_keep_classifier_and_flow_neighbour_conventions_separate():
    prior = ScenePrior.from_catalogue(_catalogue(), guard_radius_arcsec=12.0)
    view = prior.shear(0.0, 0.0)
    detection = view.detection_view(conditions=CONDITIONS, radius_arcsec=3.0)
    flow = view.flow_view(
        conditions=CONDITIONS,
        neighbour_radius_arcsec=7.0,
        crowding_radii_arcsec=(3.0, 7.0),
    )

    assert detection.loc[0, "secondary_row"] == 1
    assert np.isclose(detection.loc[0, "distance"], 2.9)
    assert flow.loc[0, "secondary_row"] == 1
    assert flow.loc[0, "nbr_flux_near"] > 0
    assert flow.loc[0, "nbr_flux_far"] > 0

    # The image simulations keep positions fixed under shear, so neighbour
    # identities, distances, and hard aperture membership must be invariant.
    compressed = prior.shear(-0.05, 0.0).detection_view(
        conditions=CONDITIONS, radius_arcsec=3.0
    )
    pd.testing.assert_series_equal(
        compressed["secondary_row"], detection["secondary_row"]
    )
    pd.testing.assert_series_equal(compressed["distance"], detection["distance"])


def test_detection_view_can_choose_flux_size_impact_instead_of_nearest():
    catalogue = _catalogue()
    catalogue.loc[2, ["r", "Re", "axis_ratio"]] = [21.0, 0.8, 0.35]
    prior = ScenePrior.from_catalogue(catalogue, guard_radius_arcsec=4.0)
    view = prior.shear(0.0, 0.0)

    nearest = view.detection_view(
        conditions=CONDITIONS,
        radius_arcsec=4.0,
        neighbour_selection="nearest",
        primary_indices=np.array([0]),
    )
    impact = view.detection_view(
        conditions=CONDITIONS,
        radius_arcsec=4.0,
        neighbour_selection="impact",
        impact_exponent=2.0,
        primary_indices=np.array([0]),
    )

    assert nearest.index.tolist() == [0]
    assert nearest.loc[0, "secondary_row"] == 1
    assert impact.loc[0, "secondary_row"] == 2
    assert np.isclose(impact.loc[0, "axis_ratio_input_p"], 1.0)
    assert np.isclose(impact.loc[0, "axis_ratio_input_s"], 0.35)
    expected = -0.4 * np.log(10.0) * 21.0 + 2.0 * (
        np.log(0.8) - np.log(3.1)
    )
    assert np.isclose(impact.loc[0, "neighbour_log_impact"], expected)

    # Shapes change under shear, but fixed positions, fluxes, and sizes make
    # the impact-ranked neighbour identity invariant.
    sheared = prior.shear(0.05, 0.0).detection_view(
        conditions=CONDITIONS,
        radius_arcsec=4.0,
        neighbour_selection="impact",
        impact_exponent=2.0,
        primary_indices=np.array([0]),
    )
    assert sheared.loc[0, "secondary_row"] == impact.loc[0, "secondary_row"]
    assert not np.isclose(
        sheared.loc[0, "axis_ratio_input_p"],
        impact.loc[0, "axis_ratio_input_p"],
    )


def test_impact_neighbour_validation_is_explicit():
    prior = ScenePrior.from_catalogue(_catalogue(), guard_radius_arcsec=4.0)
    view = prior.shear(0.0, 0.0)
    for selection, exponent in (("brightest", 2.0), ("impact", -1.0)):
        try:
            view.detection_view(
                conditions=CONDITIONS,
                radius_arcsec=4.0,
                neighbour_selection=selection,
                impact_exponent=exponent,
            )
        except ValueError:
            pass
        else:
            raise AssertionError("invalid impact-neighbour configuration was accepted")


def test_scene_store_round_trip(tmp_path):
    prior = ScenePrior.from_catalogue(
        _catalogue(), guard_radius_arcsec=12.0, weight_column=None
    )
    prior.save(tmp_path / "scene")
    restored = ScenePrior.load(tmp_path / "scene")

    pd.testing.assert_frame_equal(restored.galaxies, prior.galaxies)
    np.testing.assert_array_equal(restored.indptr, prior.indptr)
    np.testing.assert_array_equal(restored.secondary_row, prior.secondary_row)
    np.testing.assert_allclose(restored.dx_arcsec, prior.dx_arcsec)
    np.testing.assert_allclose(restored.weights, prior.weights)


def test_cached_response_pairs_match_existing_catalogue_builder_at_zero_shear():
    catalogue = _catalogue()
    config = EmulatorPairingConfig(
        cuts=((18, 29), (18, 29), (0.1, 2), (0.1, 2), (0, 10)),
        r_max_arcsec=10.0,
        k=4,
        conditions=CONDITIONS,
    )
    prior = ScenePrior.from_catalogue(catalogue, guard_radius_arcsec=11.0)
    cached = prior.shear(0.0, 0.0).response_pairs(config=config)
    direct = prepare_emulator_pairs(catalogue, config=config)
    columns = ["primary_row", "secondary_row", "distance"]
    np.testing.assert_allclose(cached[columns], direct[columns], rtol=0, atol=1e-12)


def test_response_pair_primary_filter_keeps_all_secondaries():
    prior = ScenePrior.from_catalogue(_catalogue(), guard_radius_arcsec=11.0)
    config = EmulatorPairingConfig(
        cuts=((18, 29), (18, 29), (0.1, 2), (0.1, 2), (0, 10)),
        r_max_arcsec=10.0,
        k=4,
        conditions=CONDITIONS,
    )
    pairs = prior.shear(0.0, 0.0).response_pairs(
        config=config,
        primary_indices=np.array([1, 3]),
    )
    assert set(pairs["primary_row"]) == {1, 3}
    assert set(pairs["secondary_row"]) - {1, 3}


def test_shape_only_shear_does_not_expand_required_guard_radius():
    prior = ScenePrior.from_catalogue(_catalogue(), guard_radius_arcsec=10.0)
    view = prior.shear(0.08, 0.0)
    detection = view.detection_view(conditions=CONDITIONS, radius_arcsec=10.0)
    assert len(detection) == len(_catalogue())


def test_shape_only_shear_rejects_convergence():
    prior = ScenePrior.from_catalogue(_catalogue(), guard_radius_arcsec=10.0)
    try:
        prior.shear(0.0, 0.0, kappa=0.01)
    except ValueError as error:
        assert "does not implement convergence" in str(error)
    else:
        raise AssertionError("shape-only shear silently accepted convergence")
