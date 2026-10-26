import numpy as np
import pandas as pd

from scripts.plot_disk_truth_corners import build_context, validity


def test_explicit_degree_angle_and_twenty_nonself_neighbours():
    n = 25
    scene = pd.DataFrame(dict(index=np.arange(n), RA=np.arange(n)*.3/3600,
        DEC=np.zeros(n), redshift=np.ones(n), Re=np.ones(n), axis_ratio=np.full(n, .5),
        position_angle=np.full(n, 5.), sersic_n=np.ones(n), r=np.full(n, 28.)))
    raw, pairs = build_context(scene, 0)
    np.testing.assert_allclose(raw[0, :2], np.array([np.cos(np.deg2rad(10)), np.sin(np.deg2rad(10))])/3)
    assert raw[0, 4] == np.sqrt(.5)
    np.testing.assert_array_equal(pairs.index_input_s, np.arange(1, 21))
    assert (raw[0, 5:] > 0).all()  # faint neighbours are not cut
    isolated, empty = build_context(scene.iloc[:1], 0)
    np.testing.assert_array_equal(isolated[0, 5:], 0.)
    assert len(empty) == 0


def test_validity_does_not_apply_measured_cuts_or_repair_values():
    x = np.array([[.1, .2, .1, .01], [.1, .2, 2., 100.], [1., 0., 2., 100.],
                  [.1, .2, 0., 100.], [np.nan, .2, 2., 100.]])
    saved = x.copy()
    valid, counts = validity(x)
    np.testing.assert_array_equal(valid, [True, True, False, False, False])
    np.testing.assert_array_equal(x, saved)
    assert counts == dict(nonfinite=1, nonpositive_auxiliary=1, outside_open_disk=2)
