from types import SimpleNamespace

import numpy as np

from scripts.plot_disk_joint_likelihood import compact_response, slice_points
from sbsi.catalogue_disk_response import CatalogueDiskResponse
from sbsi.disk_inference_store import ShardedDiskResponse
from sbsi.output_conditioned_response import TRUTH_FEATURES


def test_slice_axes_and_fixed_outputs():
    observed = np.array([.1, -.2, 5., 100.])
    axes = np.array([np.linspace(v-1, v+1, 5) for v in observed])
    points, xx, yy = slice_points(observed, (0, 3), axes)
    np.testing.assert_allclose(points[12], observed, rtol=0, atol=1e-14)
    np.testing.assert_array_equal(points[:, 0], xx.ravel())
    np.testing.assert_array_equal(points[:, 3], yy.ravel())
    np.testing.assert_array_equal(points[:, 1:3], np.broadcast_to(observed[1:3], (25, 2)))


def test_compact_response_preserves_requested_atom_order_and_empty_pairs():
    model = SimpleNamespace(response_coordinate='disk_velocity')
    features = np.arange(4*len(TRUTH_FEATURES)).reshape(4, len(TRUTH_FEATURES))
    original = ShardedDiskResponse([
        CatalogueDiskResponse(np.array([0, 1, 1, 2]), features[:2], model),
        CatalogueDiskResponse(np.array([0, 2]), features[2:], model)])
    compact = compact_response(original, np.array([3, 1, 0]))
    np.testing.assert_array_equal(compact.indptr, [0, 2, 2, 3])
    np.testing.assert_array_equal(compact.features, features[[2, 3, 0]])
