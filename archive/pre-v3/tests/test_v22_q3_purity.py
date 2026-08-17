import galsim
import numpy as np
import pandas as pd

from scripts.compute_v22_q3_purity import overlap_purity


def source(index=1, ra=180.0, dec=0.0, mag=24.0):
    return pd.Series({
        "index": index, "RA": ra, "DEC": dec, "position_angle": 0.0,
        "Re": 0.7, "axis_ratio": 0.8, "sersic_n": 1.0, "r": mag,
    })


def test_eq17_purity_is_one_for_isolated_source_and_half_for_identical_overlap():
    psf = galsim.Moffat(beta=2.224068, fwhm=0.73, trunc=4.5 * 0.73)
    anchor = source()
    empty = pd.DataFrame(columns=anchor.index)
    isolated = overlap_purity(
        anchor, empty, stamp_size=48, pixel_scale=0.2, psf=psf, threshold=0.0,
    )
    assert np.isclose(isolated["purity_eq17"], 1.0, atol=1e-12)
    neighbour = pd.DataFrame([source(index=2)])
    overlap = overlap_purity(
        anchor, neighbour, stamp_size=48, pixel_scale=0.2, psf=psf, threshold=0.0,
    )
    assert np.isclose(overlap["purity_eq17"], 0.5, rtol=2e-6)
    assert np.isclose(overlap["true_blendedness_eq17"], 0.5, rtol=2e-6)


def test_purity_is_asymmetric_for_unequal_flux_pair():
    psf = galsim.Moffat(beta=2.224068, fwhm=0.73, trunc=4.5 * 0.73)
    bright = source(index=1, mag=23.0)
    faint = source(index=2, mag=25.0)
    rho_bright = overlap_purity(
        bright, pd.DataFrame([faint]), stamp_size=48, pixel_scale=0.2,
        psf=psf, threshold=0.0,
    )["purity_eq17"]
    rho_faint = overlap_purity(
        faint, pd.DataFrame([bright]), stamp_size=48, pixel_scale=0.2,
        psf=psf, threshold=0.0,
    )["purity_eq17"]
    assert rho_bright > rho_faint
