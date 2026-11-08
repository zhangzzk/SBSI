import numpy as np

from sbsi.flow_coupling_target import _estimate_grid, centered_slope, coupling_bin_ids, grid_case_diagnostics


def test_centered_slope_fits_intercept_and_drops_nonfinite():
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0, np.nan])
    y = 3.5 * x - 7.0
    assert np.isclose(centered_slope(y, x), 3.5)


def test_coupling_grid_recovers_cell_resolved_mag_and_size_slopes():
    import pandas as pd

    rng = np.random.default_rng(9)
    rows = []
    for flux_bin, mag_slope, size_slope in [(0, -0.4, 0.3), (1, 1.2, -0.2)]:
        for _ in range(200):
            predictor = rng.normal(scale=0.01)
            rows.append(
                {
                    "r_input_p": 20.0 + 4.0 * flux_bin,
                    "Re_input_p": 0.8,
                    "r_blend": 0.0,
                    "orientation_shear": predictor,
                    "delta_mag": 2.0 + mag_slope * predictor,
                    "delta_log_size": -1.0 + size_slope * predictor,
                }
            )
    frame = pd.DataFrame(rows)
    result = _estimate_grid(
        frame,
        np.array([18.0, 22.0, 26.0]),
        np.array([0.5, 1.5]),
        np.array([-1.0, 1.0]),
        min_count=100,
    )
    mag, size, counts, fallback, *_ = result
    np.testing.assert_allclose(mag[:, 0, 0], [-0.4, 1.2], atol=1e-12)
    np.testing.assert_allclose(size[:, 0, 0], [0.3, -0.2], atol=1e-12)
    np.testing.assert_array_equal(counts[:, 0, 0], [200, 200])
    assert not fallback.any()


def test_ellipticity_axis_resolves_different_slopes_and_preserves_case_diagnostics():
    import pandas as pd
    rng = np.random.default_rng(13)
    rows = []
    for case in range(6):
        for e, bmag, bsize in [(0.1,-.4,.3),(.3,.2,.5),(.7,1.2,.9)]:
            for _ in range(50):
                x = rng.normal(scale=.01)
                rows.append(dict(case=case,r_input_p=22.,Re_input_p=.8,r_blend=0.,
                                 e1_input_rot0_p=e,e2_input_rot0_p=0.,orientation_shear=x,
                                 delta_mag=2+bmag*x,delta_log_size=-1+bsize*x))
    frame = pd.DataFrame(rows)
    edges = ([18.,26.],[.37,1.5],[-1.,1.],[0.,.2,.5,1.])
    mag,size,counts,fallback,*_ = _estimate_grid(frame,*edges[:3],100,edges_e_abs=edges[3])
    np.testing.assert_allclose(mag.ravel(),[-.4,.2,1.2],atol=1e-12)
    np.testing.assert_allclose(size.ravel(),[.3,.5,.9],atol=1e-12)
    np.testing.assert_array_equal(counts.ravel(),[300,300,300])
    assert not fallback.any()
    ids = coupling_bin_ids(frame,*edges)
    rotated = frame.copy()
    rotated['e2_input_rot0_p'] = rotated['e1_input_rot0_p']
    rotated['e1_input_rot0_p'] = 0.
    np.testing.assert_array_equal(ids,coupling_bin_ids(rotated,*edges))
    diagnostics=grid_case_diagnostics(frame,ids,3)
    np.testing.assert_array_equal(diagnostics['case_coverage'],[6,6,6])
    assert np.max(np.abs(diagnostics['size_case_half_difference'])) < 1e-10
    assert np.max(diagnostics['mag_case_jackknife_se']) < 1e-10
