import numpy as np
import pytest

from scripts.prepare_v36_classifier_f8_inputs import project


def fixture():
    ids = np.array([10, 20, 30])
    context = np.arange(33, dtype=np.float32).reshape(3, 11)
    x = np.zeros((3, 17), dtype=np.float32)
    x[:, :8] = context[:, :8]
    return dict(ids=ids, base_context=context[:, :8], x0=x.copy(), x1=x.copy(),
                u0=np.array([True, False, True]), u1=np.array([False, True, True])), ids, context


def test_physical_crowding_joins_by_identity_and_preserves_labels():
    a, ids, context = fixture()
    a['x1'][:, 0] += .02
    order = np.array([2, 0, 1])
    b = project(a, ids[order], context[order])
    np.testing.assert_array_equal(b['x0'], np.column_stack([a['x0'][:, :5], context[:, 8:]]))
    np.testing.assert_array_equal(b['x1'][:, :5], a['x1'][:, :5])
    for key in ('ids', 'u0', 'u1', 'base_context'):
        np.testing.assert_array_equal(b[key], a[key])


def test_missing_or_duplicate_identity_rejected():
    a, ids, context = fixture()
    with pytest.raises(ValueError):
        project(a, np.array([10, 20, 40]), context)
    with pytest.raises(ValueError):
        project(a, np.array([10, 10, 30]), context)


def test_changed_parent_properties_rejected():
    a, ids, context = fixture()
    context = context.copy()
    context[0, 3] += 1
    with pytest.raises(AssertionError):
        project(a, ids, context)
