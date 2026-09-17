"""Focused tests for the flow self-response radius diagnostic."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.diagnose_flow_self_response_by_radius import (
    add_zero_radius,
    radius_labels,
)


def test_radius_labels_are_half_open_with_open_ends():
    assert radius_labels((0.62, 0.64)) == [
        "<0.6200",
        "[0.6200,0.6400)",
        ">=0.6400",
    ]


def test_add_zero_radius_aligns_by_identity_not_row_order():
    measured = pd.DataFrame(
        {
            "case": [7, 7],
            "input_index": [10, 20],
            "R_self_measured": [0.4, 0.5],
        }
    )
    target = np.zeros((2, 4), dtype=float)
    target[:, 2] = [4.0, 3.1]
    zero = {
        "case": np.array([7, 7]),
        "input_index": np.array([20, 10]),
        "target": target,
    }
    joined = add_zero_radius(measured, zero)
    np.testing.assert_allclose(joined["g0_flux_radius_arcsec"], [0.62, 0.8])


def test_add_zero_radius_refuses_missing_identity():
    measured = pd.DataFrame(
        {"case": [7], "input_index": [99], "R_self_measured": [0.4]}
    )
    zero = {
        "case": np.array([7]),
        "input_index": np.array([10]),
        "target": np.ones((1, 4)),
    }
    with pytest.raises(RuntimeError, match="lack a physical"):
        add_zero_radius(measured, zero)
