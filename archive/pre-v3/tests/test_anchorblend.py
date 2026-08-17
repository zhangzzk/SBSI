import numpy as np
import pandas as pd

from sbs_shear.anchorblend import assert_anchor_spacing, sparse_anchor_mask
from scripts.build_anchorblend_response import unsupported_near_anchor


def test_sparse_anchors_respect_spacing_and_eligibility():
    # 1 arcsec steps on a local line. With a 2.5 arcsec exclusion, rows 0,3,6,9 survive.
    ra = 180.0 + np.arange(10) / 3600.0
    dec = np.zeros(10)
    eligible = np.ones(10, dtype=bool)
    eligible[3] = False
    keep = sparse_anchor_mask(ra, dec, eligible, min_separation_arcsec=2.5)
    assert np.all(~keep | eligible)
    assert keep.sum() >= 3
    assert assert_anchor_spacing(ra, dec, keep, 2.5) > 2.5


def test_sparse_anchors_are_deterministic():
    rng = np.random.default_rng(3)
    ra = 180 + rng.uniform(0, 0.01, 100)
    dec = rng.uniform(0, 0.01, 100)
    eligible = rng.uniform(size=100) > 0.2
    a = sparse_anchor_mask(ra, dec, eligible, 5.0)
    b = sparse_anchor_mask(ra, dec, eligible, 5.0)
    np.testing.assert_array_equal(a, b)


def test_empty_anchor_input():
    out = sparse_anchor_mask([], [], [], 20.0)
    assert out.dtype == bool and out.size == 0


def test_unsupported_secondary_excludes_only_nearby_anchor(tmp_path):
    pd.DataFrame({
        "index": [1, 2, 3],
        "RA": [180.0, 180.0 + 5.0 / 3600.0, 180.0 + 30.0 / 3600.0],
        "DEC": [0.0, 0.0, 0.0],
        "r": [22.0, 12.0, 22.0],
        "Re": [0.8, 1.0, 0.8],
    }).to_feather(tmp_path / "gals0_0.05.feather")
    got = unsupported_near_anchor(tmp_path, 0, 0.05, [1, 3], radius_arcsec=10.0)
    np.testing.assert_array_equal(got, [True, False])
