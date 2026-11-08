import importlib.util
from pathlib import Path

import numpy as np


spec = importlib.util.spec_from_file_location("disk_prior_preparation", Path(__file__).resolve().parents[1] / "scripts/prepare_disk_prior_shard.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_nearest_pairs_respect_strict_aperture_and_row_chunk():
    indptr = np.array([0, 3, 3, 5, 6])
    secondary = np.array([1, 2, 3, 0, 3, 0])
    distance = np.array([2., 1., 3., 1., 2., 1.])
    p, s, d = module.select_edges(indptr, secondary, distance, 0, 3, radius=2., neighbours=1)
    np.testing.assert_array_equal(p, [0, 2])
    np.testing.assert_array_equal(s, [2, 0])
    np.testing.assert_array_equal(d, [1., 1.])
    p, s, d = module.select_edges(indptr, secondary, distance, 1, 2, radius=2., neighbours=20)
    assert len(p) == len(s) == len(d) == 0
