"""Focused tests for the rejection-boundary emulator comparison."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.diagnose_corner_extrapolation import (
    FIELDS,
    accumulate,
    bootstrap,
    cell_indices,
    cell_shape,
    collapse,
    corner_mask_flat,
    empty_totals,
    pooled,
    primary_band_index,
    primary_band_names,
    primary_bands,
    region_report,
    stack_cases,
)
from scripts.diagnose_label_rejection import (
    DELTA_MAG_EDGES,
    DISTANCE_EDGES,
    REJECTION_DELTA_MAG,
    grid_histogram,
)


def test_cell_indices_agree_with_the_support_audit_binning():
    distance = np.array([0.75, 0.5, 10.0, 2.9, 12.0, np.nan])
    delta_mag = np.array([2.0, 2.0, 0.5, REJECTION_DELTA_MAG, 2.0, 2.0])
    flat, inside = cell_indices(distance, delta_mag)
    assert list(inside) == [True, True, True, True, False, False]
    rows, columns = cell_shape()
    counts = np.zeros(rows * columns)
    np.add.at(counts, flat, 1.0)
    # the same pairs binned by the support audit must land identically
    reference = grid_histogram(distance, delta_mag).reshape(-1)
    assert np.array_equal(counts, reference)


def test_corner_mask_covers_the_rejection_geometry_only():
    rows, columns = cell_shape()
    mask = corner_mask_flat().reshape(rows, columns)
    distance_bin = DISTANCE_EDGES.index(3.0)
    magnitude_bin = DELTA_MAG_EDGES.index(REJECTION_DELTA_MAG)
    # every separation bin below 3" and every magnitude bin at or above the
    # factor-five threshold, and nothing else
    assert mask[:distance_bin, magnitude_bin:].all()
    assert not mask[distance_bin:, :].any()
    assert not mask[:, :magnitude_bin].any()


def planted_frame():
    return pd.DataFrame(
        {
            "distance": np.array([0.75, 0.75, 5.0]),
            "delta_mag": np.array([2.0, 2.0, -3.0]),
            #  all three primaries sit in the brightest band, so the flat index
            #  of a cell is unchanged by the primary-magnitude axis
            "primary_mag": np.array([22.0, 22.0, 22.0]),
            "model": np.array([0.4, 0.6, 0.1]),
            "measured": np.array([0.5, 0.5, 0.2]),
        }
    )


def test_accumulate_sums_into_the_right_cells():
    totals = empty_totals()
    accumulate(totals, planted_frame())
    flat, _ = cell_indices(np.array([0.75]), np.array([2.0]))
    cell = int(flat[0])
    assert totals["n"][cell] == 2
    assert totals["model"][cell] == pytest.approx(1.0)
    assert totals["measured"][cell] == pytest.approx(1.0)
    assert totals["n"].sum() == 3


def build_totals(scale=1.0):
    totals = empty_totals()
    frame = planted_frame()
    frame["model"] = frame["model"] * scale
    accumulate(totals, frame)
    return totals


def test_pooled_forms_per_pair_means_and_the_ratio():
    cases, resolved = stack_cases({0: build_totals(), 1: build_totals()})
    stacked = collapse(resolved)
    assert cases == (0, 1)
    assert stacked.shape == (2, cell_shape()[0] * cell_shape()[1], len(FIELDS))
    grids = pooled(stacked)
    flat, _ = cell_indices(np.array([0.75]), np.array([2.0]))
    cell = int(flat[0])
    assert grids["n"][cell] == 4
    assert grids["mean_model"][cell] == pytest.approx(0.5)
    assert grids["mean_measured"][cell] == pytest.approx(0.5)
    assert grids["model_over_measured"][cell] == pytest.approx(1.0)
    assert np.nansum(grids["response_share"]) == pytest.approx(1.0)


def test_bootstrap_reports_finite_non_negative_errors():
    _, resolved = stack_cases(
        {0: build_totals(1.0), 1: build_totals(1.2), 2: build_totals(0.8)}
    )
    stacked = collapse(resolved)
    errors = bootstrap(stacked, n_boot=200, seed=11)
    flat, _ = cell_indices(np.array([0.75]), np.array([2.0]))
    cell = int(flat[0])
    for name in ("mean_model", "mean_measured", "model_over_measured"):
        assert errors[name].shape == (cell_shape()[0] * cell_shape()[1],)
        assert errors[name][cell] >= 0.0
    # the measured side is identical in every case, so it carries no case scatter
    assert errors["mean_measured"][cell] == pytest.approx(0.0)
    assert errors["mean_model"][cell] > 0.0


def test_region_report_recovers_a_planted_ratio_and_flags_empty_regions():
    _, resolved = stack_cases({0: build_totals(1.0), 1: build_totals(2.0)})
    stacked = collapse(resolved)
    report = region_report(stacked, corner_mask_flat(), n_boot=200, seed=5)
    assert report["pairs"] == 4
    # model sums to 1.0 + 2.0 against a measured sum of 2.0
    assert report["model_over_measured"] == pytest.approx(1.5)
    assert report["mean_model"] == pytest.approx(0.75)
    assert report["mean_measured"] == pytest.approx(0.5)
    assert report["model_over_measured_standard_error"] > 0.0
    empty = region_report(stacked, np.zeros(stacked.shape[1], dtype=bool), 10, 5)
    assert empty["status"] == "no_labelled_pairs"


def test_primary_band_index_places_magnitudes_half_open():
    index = primary_band_index(np.array([10.0, 23.0, 23.5, 24.0, 25.9, 26.0, 99.0]))
    assert list(index) == [0, 1, 1, 2, 3, 4, 4]
    assert len(primary_band_names()) == primary_bands()


def test_collapsing_the_primary_axis_reproduces_the_two_dimensional_grid():
    #  the primary-magnitude axis is a refinement: summing it away must give
    #  back exactly the numbers this diagnostic reported before it existed.
    totals = empty_totals()
    frame = planted_frame()
    frame["primary_mag"] = np.array([22.0, 25.5, 24.5])
    accumulate(totals, frame)
    _, resolved = stack_cases({0: totals})
    assert resolved.shape[1] == primary_bands()
    stacked = collapse(resolved)
    flat, _ = cell_indices(np.array([0.75]), np.array([2.0]))
    cell = int(flat[0])
    #  the two corner pairs are now in different primary bands but still pool
    assert stacked[0, cell, FIELDS.index("n")] == 2
    assert stacked[0, cell, FIELDS.index("model")] == pytest.approx(1.0)
    #  and each band holds exactly one of them
    assert resolved[0, 0, cell, FIELDS.index("model")] == pytest.approx(0.4)
    assert resolved[0, 3, cell, FIELDS.index("model")] == pytest.approx(0.6)
    assert resolved[:, :, :, FIELDS.index("n")].sum() == 3
