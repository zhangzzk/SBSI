"""Focused tests for re-weighting the quiet-blend flow read to the cohort."""

from __future__ import annotations

import numpy as np
import pytest

from scripts.reweight_constgold_flow_error import (
    estimate,
    memberships,
    responses,
)

H = 0.02


def planted(cells, bands, per_cell, cases=3):
    """Sufficient-statistic arrays for cells with a planted response each.

    ``per_cell[cell]`` is ``(objects, R_measured, R_flow, R_blend)``.
    """
    numerator = np.zeros((cases, len(cells), 3, 2))
    denominator = np.zeros((cases, len(cells), 3, 2))
    for case in range(cases):
        for index, cell in enumerate(cells):
            count, *values = per_cell[cell]
            for term, value in enumerate(values):
                for side, sign in enumerate((1.0, -1.0)):
                    numerator[case, index, term, side] = sign * value * H * count
                    denominator[case, index, term, side] = count
    return numerator, denominator


def test_memberships_split_the_quiet_bins_out_of_each_band():
    cells = [f"bin{i:02d}|{b}" for i in range(3) for b in ("lo", "hi")]
    whole, narrow = memberships(cells, ("lo", "hi"), quiet_bins=1)
    #  every cell of a band counts in the whole-band row
    assert whole.sum(axis=1).tolist() == [3.0, 3.0]
    #  only bin00 counts in the quiet row
    assert narrow.sum(axis=1).tolist() == [1.0, 1.0]
    assert narrow[0, cells.index("bin00|lo")] == 1.0
    assert narrow[0, cells.index("bin01|lo")] == 0.0


def test_responses_pool_cells_by_object_count():
    cells = ["a", "b"]
    numerator, denominator = planted(cells, (), {"a": (900, 0.4, 0.4, 0.0),
                                                 "b": (100, 0.8, 0.8, 0.0)})
    members = np.array([[1.0, 1.0]])
    values, objects = responses(numerator, denominator, members, H)
    assert objects[0] == pytest.approx(3000.0)
    assert values[0, 1] == pytest.approx(0.9 * 0.4 + 0.1 * 0.8)


def test_reweighting_recovers_a_magnitude_dependent_flow_error():
    #  Plant a flow error that depends only on magnitude -- +10% on the bright
    #  band, 0% on the faint one -- and make the quiet bin 90% bright while the
    #  cohort is 90% faint.  The quiet read alone would say +9%; the cohort
    #  answer must be the response-weighted +10%/0% mix, which is far smaller.
    bands = ("bright", "faint")
    cells = [f"bin{i:02d}|{b}" for i in range(2) for b in bands]
    flow = {"bright": 1.0, "faint": 0.2}
    error = {"bright": 0.10, "faint": 0.0}
    counts = {"bin00|bright": 900, "bin00|faint": 100,
              "bin01|bright": 100, "bin01|faint": 900}
    per_cell = {cell: (counts[cell], flow[cell.split("|")[1]] * (1 + error[cell.split("|")[1]]),
                       flow[cell.split("|")[1]], 0.0) for cell in cells}
    numerator, denominator = planted(cells, bands, per_cell)
    whole, narrow = memberships(cells, bands, quiet_bins=1)
    out = estimate(numerator, denominator, whole, narrow, H)

    assert out["epsilon"] == pytest.approx([0.10, 0.0])
    #  the cohort is half and half by object count, but the bright band carries
    #  five times the response, so it dominates the response-weighted answer
    assert out["share"] == pytest.approx([0.5, 0.5])
    assert out["flow_model"] == pytest.approx(0.5 * 1.0 + 0.5 * 0.2)
    assert out["flow_error"] == pytest.approx(100.0 * (0.5 * 0.10 * 1.0) / 0.6)
    #  an object-weighted average would have said +5%; response weighting +8.3%
    assert out["flow_error"] == pytest.approx(8.3333, abs=1e-3)


def test_the_blend_error_is_what_the_measured_sum_leaves_over():
    #  ConstGold measures R_measured = R_flow + R_blend, so once the flow is
    #  pinned the blending term is not a second measurement, it is a remainder.
    bands = ("only",)
    cells = ["bin00|only", "bin01|only"]
    #  quiet cell: flow 1.0, model 1.0, measured 1.02 -> flow is 2% low
    #  loud cell: blending carries the rest
    per_cell = {"bin00|only": (1000, 1.02, 1.00, 0.00),
                "bin01|only": (1000, 1.50, 1.00, 0.60)}
    numerator, denominator = planted(cells, bands, per_cell)
    whole, narrow = memberships(cells, bands, quiet_bins=1)
    out = estimate(numerator, denominator, whole, narrow, H)
    assert out["epsilon"][0] == pytest.approx(0.02)
    assert out["flow_true"] == pytest.approx(1.02)
    #  measured mean is 1.26, so the blending term can only be 0.24 against a
    #  model 0.30 -- the model is 25% high
    assert out["R_measured"] == pytest.approx(1.26)
    assert out["blend_model"] == pytest.approx(0.30)
    assert out["blend_true"] == pytest.approx(0.24)
    assert out["blend_error"] == pytest.approx(25.0)


def test_coverage_reports_how_much_of_a_band_the_quiet_read_sees():
    bands = ("b",)
    cells = ["bin00|b", "bin01|b"]
    per_cell = {"bin00|b": (200, 0.5, 0.5, 0.0), "bin01|b": (800, 0.5, 0.5, 0.0)}
    numerator, denominator = planted(cells, bands, per_cell)
    whole, narrow = memberships(cells, bands, quiet_bins=1)
    out = estimate(numerator, denominator, whole, narrow, H)
    assert out["coverage"][0] == pytest.approx(0.2)
    #  no planted error anywhere, so the cohort answer must be exactly zero
    assert out["flow_error"] == pytest.approx(0.0, abs=1e-12)
