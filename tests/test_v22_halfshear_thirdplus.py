import numpy as np

from scripts.diag_v22_halfshear_thirdplus import balance_statistics


def test_balance_statistics_passes_balanced_groups():
    ids = np.tile(np.arange(4), 100)
    x = np.repeat(np.arange(100, dtype=float), 4)
    result = balance_statistics(ids, {"x": x}, max_smd=0.1)
    assert result["x"] == 0.0


def test_balance_statistics_rejects_imbalanced_groups():
    ids = np.repeat(np.arange(4), 100)
    x = ids.astype(float)
    try:
        balance_statistics(ids, {"x": x}, max_smd=0.1)
    except RuntimeError as exc:
        assert "balance gate failed" in str(exc)
    else:
        raise AssertionError("imbalanced groups passed the balance gate")
