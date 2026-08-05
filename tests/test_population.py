import numpy as np
import pandas as pd

from sbs_shear import domain
from sbs_shear.population import EXTENDED_PAIR_CUTS, LSST_PAIR_CUTS, pair_mask, primary_mask


def frame(**overrides):
    values = dict(
        r_input_p=[24.0], Re_input_p=[0.8],
        r_input_s=[24.0], Re_input_s=[0.8], distance=[2.0],
    )
    values.update(overrides)
    return pd.DataFrame(values)


def test_standard_lsst_pair_passes():
    assert pair_mask(frame()).tolist() == [True]


def test_primary_domain_is_intrinsic_and_v21():
    faint = float(domain.sn_limiting_mag(0.8)) + 0.01
    assert primary_mask(frame(r_input_p=[faint])).tolist() == [False]
    assert primary_mask(frame(Re_input_p=[0.5])).tolist() == [False]
    assert primary_mask(frame(Re_input_p=[0.5001])).tolist() == [True]


def test_secondary_and_separation_boundaries_are_strict():
    assert not pair_mask(frame(r_input_s=[LSST_PAIR_CUTS.secondary_mag[0]])).item()
    assert not pair_mask(frame(Re_input_s=[LSST_PAIR_CUTS.secondary_re[1]])).item()
    assert not pair_mask(frame(distance=[LSST_PAIR_CUTS.separation[1]])).item()


def test_no_measured_or_blending_columns_are_required():
    minimal = frame()
    assert "detected" not in minimal and "r_blend" not in minimal and "neighbored" not in minimal
    assert pair_mask(minimal).item()


def test_nan_secondary_excludes_isolated_pair():
    assert not pair_mask(frame(r_input_s=[np.nan], Re_input_s=[np.nan], distance=[np.nan])).item()


def test_extended_neighbour_support_is_explicit():
    wide = frame(r_input_s=[15.0], Re_input_s=[5.0], distance=[9.0])
    assert not pair_mask(wide, cuts=LSST_PAIR_CUTS).item()
    assert pair_mask(wide, cuts=EXTENDED_PAIR_CUTS).item()
