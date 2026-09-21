import numpy as np
import pandas as pd
import pytest

from sbsi.catalogue_disk_response import CatalogueDiskResponse
from sbsi.output_conditioned_response import TRUTH_FEATURES, output_features


class Trees:
    response_coordinate = "disk_velocity"
    output_thresholds = (np.array([1.], np.float32), np.array([24.], np.float32))
    transformed_outputs = staticmethod(output_features)

    def predict_pairs(self, pairs, radius, flux):
        x = self.transformed_outputs(radius, flux).astype(np.float32)
        return pairs.iloc[:, 0].to_numpy() + (x[:, 0] >= 1) + 2 * (x[:, 1] >= 24)


def test_ragged_pair_sums_pool_exact_cells_and_keep_isolated_atoms():
    frame = pd.DataFrame(np.ones((3, 7)), columns=TRUTH_FEATURES)
    frame[TRUTH_FEATURES[0]] = [1, 2, 4]
    frame["index_input_p"] = [11, 10, 11]
    frame["index_input_s"] = [10, 11, 12]
    model = Trees()
    response = CatalogueDiskResponse.from_pairs([10, 11, 12], frame, model)
    atoms = np.array([[0, 1, 1, 2], [1, 0, 1, 2]])
    radius = np.array([[2., 3., 3., 1.], [3.1, 1., 2., 1.]])
    flux = np.full(atoms.shape, 100.)
    actual, report = response.predict(atoms, radius, flux, pair_batch_size=1, cell_batch_size=1)
    expected = np.zeros(atoms.shape)
    for slot in np.ndindex(atoms.shape):
        pairs = frame[frame.index_input_p == [10, 11, 12][atoms[slot]]]
        expected[slot] = model.predict_pairs(pairs, np.full(len(pairs), radius[slot]),
                                             np.full(len(pairs), flux[slot])).sum()
    np.testing.assert_array_equal(actual, expected)
    assert report["unique_atom_output_cells"] < atoms.size
    with pytest.raises(ValueError, match="valid aligned"):
        response.predict(np.array([-1]), [2.], [100.])
    with pytest.raises(ValueError, match="positive radius"):
        response.predict(np.array([0]), [0.], [100.])


def test_empty_query():
    response = CatalogueDiskResponse([0, 0], np.empty((0, 7)), Trees())
    values, report = response.predict(np.array([], dtype=int), [], [])
    assert values.size == report["predicted_pair_cells"] == 0
