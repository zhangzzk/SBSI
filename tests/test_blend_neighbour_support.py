"""Focused tests for the R_blend neighbour-support bands."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from pyarrow import feather

from scripts.diagnose_constgold_blend_neighbour_support import (
    GUARD_EDGES,
    GUARD_RADIUS_ARCSEC,
    GUARD_RATIO_DELTA_MAG,
    accumulate,
    band_names,
    banded,
    guard_cell,
    label_side_bands,
    report,
)

EDGES = (-np.inf, -2.0, 0.0, 2.0, np.inf)


def test_banded_places_values_in_half_open_bands():
    index = banded(np.array([-10.0, -2.0, -1.0, 0.0, 1.9, 2.0, 99.0]), EDGES)
    assert list(index) == [0, 1, 1, 2, 2, 3, 3]


def test_band_names_are_half_open_and_name_the_infinities():
    assert band_names(EDGES) == ["[-inf,-2)", "[-2,0)", "[0,2)", "[2,inf)"]


def test_accumulate_sums_pairs_response_and_absolute_response():
    table = {}
    values = np.array([-3.0, -1.0, 1.0, 1.0, 5.0])
    response = np.array([0.1, -0.2, 0.3, 0.4, -0.5])
    accumulate(table, "k", EDGES, values, response)
    accumulate(table, "k", EDGES, values, response)
    entry = table["k"]
    assert list(entry["pairs"]) == [2, 2, 4, 2]
    assert entry["response"][2] == pytest.approx(2 * 0.7)
    assert entry["response"][1] == pytest.approx(2 * -0.2)
    #  a band that cancels in the signed sum still shows its weight
    assert entry["absolute_response"][1] == pytest.approx(2 * 0.2)


def test_report_shares_sum_to_one_and_track_the_signed_total():
    table = {}
    accumulate(table, "k", EDGES, np.array([-3.0, -1.0, 1.0, 5.0]),
               np.array([0.1, 0.2, 0.3, 0.4]))
    rows = report(table["k"], EDGES)
    assert [row["band"] for row in rows] == band_names(EDGES)
    assert sum(row["pair_share"] for row in rows) == pytest.approx(1.0)
    assert sum(row["response_share"] for row in rows) == pytest.approx(1.0)
    assert rows[3]["summed_response"] == pytest.approx(0.4)
    assert rows[3]["response_share"] == pytest.approx(0.4)


def test_report_returns_none_shares_when_a_total_cancels_exactly():
    table = {}
    accumulate(table, "k", EDGES, np.array([-3.0, 1.0]), np.array([0.5, -0.5]))
    rows = report(table["k"], EDGES)
    assert all(row["response_share"] is None for row in rows)
    #  the absolute weighting still resolves the two bands
    assert rows[0]["absolute_share"] == pytest.approx(0.5)


def test_guard_cell_splits_on_the_rejection_rule():
    #  the guard refuses a neighbour more than five times brighter (delta mag
    #  above 1.7474) within three arcseconds.  Only the fourth cell is refused.
    delta = np.array([0.0, 2.0, 0.0, 2.0, GUARD_RATIO_DELTA_MAG, 2.0])
    distance = np.array([10.0, 10.0, 1.0, 1.0, 1.0, GUARD_RADIUS_ARCSEC])
    cells = [band_names(GUARD_EDGES)[i]
             for i in banded(guard_cell(delta, distance), GUARD_EDGES)]
    assert cells == ["faint_or_far", "bright_and_far", "faint_and_near",
                     "bright_and_near", "faint_and_near", "bright_and_far"]


def _write_catalogue(path, rows):
    frame = pd.DataFrame(rows)
    feather.write_feather(pa.Table.from_pandas(frame, preserve_index=False), path)


def test_label_side_divides_the_measured_column_by_the_response_shear(tmp_path):
    #  delta_et1 is a shear difference over 2h; the emulator predicts a
    #  response.  Dividing by the response shear puts the two on one footing --
    #  scripts/diagnose_corner_extrapolation.py does exactly this.
    path = tmp_path / "labels.feather"
    _write_catalogue(path, {
        "case": [0, 0],
        "r_input_p": [24.0, 24.0],
        "r_input_s": [25.0, 22.0],
        "distance": [1.0, 1.0],
        "delta_et1": [0.04, 0.10],
    })
    tenth = label_side_bands(path, cases=1, response_shear=0.1)
    fifth = label_side_bands(path, cases=1, response_shear=0.2)
    assert tenth["rows"] == 2
    assert tenth["bands"]["guard_region"]["response"].sum() == pytest.approx(1.4)
    assert fifth["bands"]["guard_region"]["response"].sum() == pytest.approx(0.7)


def test_label_side_files_a_bright_close_neighbour_in_the_refused_cell(tmp_path):
    path = tmp_path / "labels.feather"
    _write_catalogue(path, {
        "case": [0, 0],
        #  row 0 is 3 mag brighter at 1" -- refused by the guard
        #  row 1 is the same neighbour at 5" -- kept
        "r_input_p": [24.0, 24.0],
        "r_input_s": [21.0, 21.0],
        "distance": [1.0, 5.0],
        "delta_et1": [0.06, 0.02],
    })
    bands = {row["band"]: row for row in report(
        label_side_bands(path, 1, 0.2)["bands"]["guard_region"], GUARD_EDGES)}
    assert bands["bright_and_near"]["pairs"] == 1
    assert bands["bright_and_near"]["summed_response"] == pytest.approx(0.3)
    assert bands["bright_and_far"]["pairs"] == 1
    assert bands["faint_or_far"]["pairs"] == 0
