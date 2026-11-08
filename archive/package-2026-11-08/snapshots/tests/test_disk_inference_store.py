from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from sbsi.catalogue_sampling import ProposalCoordinateTable
from sbsi.crowding import FLOW_FEATURES
from sbsi.disk_inference_store import FrozenDiskCache, ShardedDiskResponse, coordinate_table, stencil
from sbsi.shear_map import apply_shear_to_ellipticity


class Response:
    def __init__(self, n, offset):
        self.n_atoms, self.offset = n, offset

    def predict(self, atoms, radius, flux):
        return atoms+self.offset+radius+flux, {}


def test_sharded_response_preserves_order_and_repeated_atoms():
    response = ShardedDiskResponse([Response(2, 0), Response(3, 2)])
    atoms = np.array([[4, 0, 1], [3, 0, 2]])
    result, report = response.predict(atoms, np.ones_like(atoms), 2*np.ones_like(atoms))
    np.testing.assert_array_equal(result, atoms+3)
    assert report["response_shards_used"] == 2
    with pytest.raises(ValueError):
        response.predict(np.array([5]), np.array([1]), np.array([2]))


def test_frozen_cache_shears_context_and_rejects_missing_node():
    zero = pd.DataFrame(np.ones((3, 8)), columns=FLOW_FEATURES)
    zero["e1_input_p"], zero["e2_input_p"] = .1, -.2
    probabilities = {(0., 0.): np.full(3, .3), (.02, .001): np.full(3, .4)}
    cache = FrozenDiskCache(zero, probabilities, {})
    actual = cache.get(.02, .001)
    expected = np.column_stack(apply_shear_to_ellipticity(np.full(3, .1), np.full(3, -.2), .02, .001))
    np.testing.assert_array_equal(actual.flow.iloc[:, :2], expected)
    np.testing.assert_array_equal(cache.zero_flow.iloc[:, 0], np.full(3, .1))
    np.testing.assert_array_equal(actual.detection_probability, np.full(3, .4))
    np.testing.assert_allclose(cache.prior.weights.sum(), 1.)
    with pytest.raises(ValueError, match="unprepared"):
        cache.get(.01, 0.)
    with pytest.raises(ValueError, match="probabilities"):
        FrozenDiskCache(zero, {(0., 0.): np.ones(2)}, {})


def test_global_coordinate_floor_matches_existing_pipeline():
    draws = np.random.default_rng(14).normal(size=(20, 128, 4))
    class Flow:
        def sample(self, frame, **kwargs):
            return draws[frame.index]
    raw = pd.DataFrame(np.ones((20, 2)))
    cache = SimpleNamespace(get=lambda *g: SimpleNamespace(flow=raw),
        prior=SimpleNamespace(weights=np.ones(20)/20))
    likelihood = SimpleNamespace(target_names=("a", "b", "c", "d"), cache=cache, flow_model=Flow())
    oracle = ProposalCoordinateTable.from_flow(likelihood, target_names=likelihood.target_names,
        n_flow_samples=128, statistic="mean", dispersion_statistic="std", row_chunk=7)
    actual = coordinate_table(draws.mean(1), draws.std(1), likelihood.target_names)
    for name in ("values", "dispersion", "center", "scale"):
        # from_flow's advanced target-column indexing changes array strides
        # and hence float64 reduction order, not the mathematical summaries.
        np.testing.assert_allclose(getattr(actual, name), getattr(oracle, name), atol=5e-16, rtol=2e-15)


def test_stencil_contains_zero_and_complete_nine_point_grid():
    points = stencil((.02, .0001), .001)
    assert len(set(points)) == 10
    assert points[0] == (0., 0.)
    assert (.019, -.0009) in points
    with pytest.raises(ValueError):
        stencil((0., 0.), -1.)
