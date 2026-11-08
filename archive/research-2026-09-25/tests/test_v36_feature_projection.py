import numpy as np
import pytest

from scripts.project_v36_flow_features import INDICES, project_arrays


def test_projection_preserves_invalid_labels_ids_and_row_order():
    source = dict(context=np.arange(33, dtype=np.float32).reshape(3, 11),
                  input_index=np.array([9, 2, 7]), usable=np.array([True, False, True]),
                  target=np.array([[1., 2.], [np.nan, np.nan], [3., 4.]]),
                  gamma=np.array([[.05, 0.], [0., .05], [-.05, 0.]]))
    result = project_arrays(source)
    np.testing.assert_array_equal(result['context'], source['context'][:, [0, 1, 2, 3, 4, 8, 9, 10]])
    assert INDICES == [0, 1, 2, 3, 4, 8, 9, 10]
    for key in source:
        if key != 'context':
            assert result[key] is source[key]


@pytest.mark.parametrize('width', [8, 10, 12])
def test_projection_rejects_already_projected_or_incompatible_context(width):
    with pytest.raises(ValueError):
        project_arrays({'context': np.zeros((3, width))})
