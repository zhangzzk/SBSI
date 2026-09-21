import pandas as pd

from scripts.audit_constgold_response_domain import _bounds, _require_equal_bounds
from scripts.build_constgold_blend_lookup import _eligible_primary_ids


def test_domain_bounds_are_strictly_compared():
    assert _bounds([18, 25.8], label="test") == (18.0, 25.8)
    _require_equal_bounds((0.37, 1.5), (0.37, 1.5), label="test")


def test_domain_bounds_reject_mismatch():
    try:
        _require_equal_bounds((0.5, 1.5), (0.37, 1.5), label="size")
    except ValueError as error:
        assert "size mismatch" in str(error)
    else:
        raise AssertionError("domain mismatch was accepted")


def test_response_lookup_support_includes_isolated_in_domain_primary():
    frame = pd.DataFrame(
        {
            "index": [1, 2, 3, 4],
            "r": [22.0, 25.7, 20.0, 19.0],
            "Re": [0.37, 0.5, 1.5, 1.0],
        }
    )
    cuts = [[13, 29], [18, 25.8], [0, 10], [0.37, 1.5], [0, 10]]
    assert _eligible_primary_ids(frame, cuts).tolist() == [2, 4]
