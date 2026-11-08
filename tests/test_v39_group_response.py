from collections import OrderedDict
from types import SimpleNamespace

import numpy as np
import pytest
from _script_loader import load_script_module


M = load_script_module("v39_group_response.py")


def test_response_reuse_respects_row_atom_mask_and_fixed_group():
    calls = []

    def predict(indices, raw, valid):
        calls.append(indices.copy())
        return np.where(valid, indices * raw[:, 2, None] + raw[:, 3, None], 0.0)

    likelihood = SimpleNamespace(
        _coefficients=predict, _response_cache=OrderedDict(), disk_response=SimpleNamespace(n_atoms=7)
    )
    raw = np.array([[0.1, 0.2, 3.0, 8.0], [-0.3, 0.0, 9.0, 2.0]])
    full = np.broadcast_to(np.arange(7), (2, 7))
    with M.GroupResponseMemo(likelihood, raw) as memo:
        likelihood._coefficients(full, raw, np.ones_like(full, bool))
        picked = np.array([[5, 1, 5, 0], [0, 5, 3, 3]])
        valid = np.array([[True, False, True, True], [False, True, True, True]])
        result = likelihood._coefficients(picked, raw, valid)
        assert len(calls) == 1
        expected = np.where(valid, picked * raw[:, 2, None] + raw[:, 3, None], 0.0)
        np.testing.assert_array_equal(result, expected)
        memo.replay = True
        np.testing.assert_array_equal(likelihood._coefficients(picked, raw, valid), expected)
        assert len(calls) == 2 and memo.replay_calls == 1
        with pytest.raises(ValueError, match="fixed observation"):
            likelihood._coefficients(picked, raw[::-1], valid)
        memo.values[0, 5] += 1.0
        with pytest.raises(AssertionError):
            likelihood._coefficients(picked, raw, valid)
    assert likelihood._coefficients is predict
