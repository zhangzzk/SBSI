import numpy as np

from scripts.build_neighbor_flux_shells import scene_flux_summaries


def test_scene_flux_shells_and_thirdplus():
    arcsec = 1.0 / 3600.0
    pos = np.array([
        [0.0, 0.0],
        [0.5 * arcsec, 0.0],
        [2.0 * arcsec, 0.0],
        [5.0 * arcsec, 0.0],
        [20.0 * arcsec, 0.0],
    ])
    flux = np.array([10.0, 4.0, 3.0, 2.0, 100.0])
    got = scene_flux_summaries(pos, flux)

    np.testing.assert_allclose(got["shell_flux"][:, 0], [4.0, 3.0, 2.0])
    np.testing.assert_array_equal(got["shell_count"][:, 0], [1, 1, 1])
    assert got["top1"][0] == 4.0
    assert got["top2"][0] == 3.0
    assert got["thirdplus"][0] == 2.0
    assert got["shell_count"][:, 4].sum() == 0


def test_scene_flux_no_pairs():
    pos = np.array([[0.0, 0.0], [1.0, 1.0]])
    got = scene_flux_summaries(pos, np.array([1.0, 2.0]))
    assert not got["shell_flux"].any()
    assert not got["shell_count"].any()
    assert not got["thirdplus"].any()
