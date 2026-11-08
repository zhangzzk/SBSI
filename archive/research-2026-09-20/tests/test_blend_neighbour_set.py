"""Focused tests for the R_blend neighbour-set component audit."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.diagnose_blend_neighbour_set import (
    QUANTITIES,
    REPORTED_GROUPS,
    bootstrap_group_summary,
    case_group_means,
    full_scene_response,
)


class StubPredictor:
    """Return a fixed per-pair frame so aggregation can be checked exactly."""

    def __init__(self, frame):
        self.frame = frame

    def predict_response(self, primaries, truth):
        return self.frame.copy()


def make_frame(n=6, offset=0.0):
    return pd.DataFrame(
        {
            "input_index": np.arange(n, dtype=np.int64),
            "R_blend_full_scene": np.linspace(0.2, 0.4, n) + offset,
            "R_blend_secondary_half": 0.6 * (np.linspace(0.2, 0.4, n) + offset),
            "R_blend_primary_half": 0.4 * (np.linspace(0.2, 0.4, n) + offset),
            "pairs_secondary_half": np.full(n, 9, dtype=np.int64),
            "pairs_primary_half": np.full(n, 7, dtype=np.int64),
            "R_blend_labelled_model": np.linspace(0.1, 0.2, n) + offset,
            "R_blend_labelled_measured": np.linspace(0.1, 0.2, n) + offset,
            "pairs_full_scene": np.full(n, 9, dtype=np.int64),
            "pairs_labelled": np.full(n, 4, dtype=np.int64),
            # three inside the old support, three outside it in different ways
            "true_r_magnitude": np.array([20.0, 21.0, 22.0, 26.0, 20.5, 27.0]),
            "true_Re_semimajor_arcsec": np.array([0.5, 0.6, 0.7, 0.5, 0.2, 0.1]),
        }
    )


def test_full_scene_response_sums_pairs_per_primary():
    pairs = pd.DataFrame(
        {
            "index_input_p": np.array([1, 1, 2], dtype=np.int64),
            "index_input_s": np.array([3, 11, 12], dtype=np.int64),
            "response": np.array([0.1, 0.2, 0.5]),
            "distance": np.array([1.0, 4.0, 2.0]),
        }
    )
    table, report = full_scene_response(
        StubPredictor(pairs), None, None, np.array([1, 2, 3], dtype=np.int64), 10
    )
    assert list(table["input_index"]) == [1, 2]
    assert table.loc[table["input_index"] == 1, "R_blend_full_scene"].item() == pytest.approx(0.3)
    assert table.loc[table["input_index"] == 1, "pairs_full_scene"].item() == 2
    row1 = table["input_index"] == 1
    assert table.loc[row1, "R_blend_primary_half"].item() == pytest.approx(0.1)
    assert table.loc[row1, "R_blend_secondary_half"].item() == pytest.approx(0.2)
    assert table.loc[row1, "pairs_secondary_half"].item() == 1
    assert report["pair_rows"] == 3
    assert report["secondary_half_pair_rows"] == 2
    assert report["primary_half_size"] == 10
    assert report["anchor_without_full_scene_pairs"] == 1
    assert report["distance_arcsec_quantiles"]["1.0"] == pytest.approx(4.0)


def test_full_scene_response_rejects_non_anchor_primaries():
    pairs = pd.DataFrame(
        {
            "index_input_p": np.array([7], dtype=np.int64),
            "index_input_s": np.array([1], dtype=np.int64),
            "response": np.array([0.1]),
            "distance": np.array([1.0]),
        }
    )
    with pytest.raises(RuntimeError, match="non-anchor"):
        full_scene_response(
            StubPredictor(pairs), None, None, np.array([1], dtype=np.int64), 10
        )


def test_full_scene_response_rejects_non_finite_predictions():
    pairs = pd.DataFrame(
        {
            "index_input_p": np.array([1], dtype=np.int64),
            "index_input_s": np.array([2], dtype=np.int64),
            "response": np.array([np.nan]),
            "distance": np.array([1.0]),
        }
    )
    with pytest.raises(RuntimeError, match="non-finite"):
        full_scene_response(
            StubPredictor(pairs), None, None, np.array([1], dtype=np.int64), 10
        )


def test_case_group_means_partitions_and_averages():
    means = case_group_means(make_frame())
    assert set(means) == set(REPORTED_GROUPS)
    assert means["inside_old_truth_support"]["n"] == 3
    assert means["outside_magnitude_only"]["n"] == 1
    assert means["outside_size_only"]["n"] == 1
    assert means["outside_both"]["n"] == 1
    assert means["all_fixed_g0"]["n"] == 6
    # the disjoint outside partitions reconstruct the union count
    assert means["outside_old_truth_support"]["n"] == 3
    assert means["all_fixed_g0"]["pairs_full_scene"] == pytest.approx(9.0)


def test_bootstrap_group_summary_reports_ratios_and_errors():
    per_case = {
        0: case_group_means(make_frame(offset=0.0)),
        1: case_group_means(make_frame(offset=0.02)),
        2: case_group_means(make_frame(offset=-0.02)),
    }
    summary = bootstrap_group_summary(per_case, n_boot=200, seed=7)
    entry = summary["all_fixed_g0"]
    assert entry["status"] == "complete"
    assert entry["cases"] == 3
    assert entry["objects"] == 18
    assert summary["truth_large"]["status"] == "not_evaluable_on_every_case"
    for quantity in QUANTITIES:
        assert entry[quantity]["standard_error"] >= 0.0
    expected = entry["R_blend_full_scene"]["mean"] / entry["R_blend_labelled_model"]["mean"]
    assert entry["full_over_labelled_model"]["ratio"] == pytest.approx(expected)
    assert entry["labelled_model_over_measured"]["ratio"] == pytest.approx(1.0)


def test_bootstrap_group_summary_marks_groups_empty_in_some_case():
    frame = make_frame()
    frame.loc[:, "true_r_magnitude"] = 20.0
    frame.loc[:, "true_Re_semimajor_arcsec"] = 0.5
    per_case = {0: case_group_means(frame), 1: case_group_means(make_frame())}
    summary = bootstrap_group_summary(per_case, n_boot=10, seed=1)
    assert summary["all_fixed_g0"]["status"] == "complete"
    assert summary["outside_size_only"]["status"] == "not_evaluable_on_every_case"
    assert summary["outside_size_only"]["cases_without_members"] == [0]
    # rare tail groups empty in every case are reported, never averaged
    assert summary["truth_large"]["status"] == "not_evaluable_on_every_case"
