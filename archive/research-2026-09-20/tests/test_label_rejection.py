"""Focused tests for the label-rejection support audit."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.diagnose_label_rejection import (
    DELTA_MAG_EDGES,
    DISTANCE_EDGES,
    REJECTION_DELTA_MAG,
    bootstrap_classes,
    bright_neighbour_audit,
    case_summary,
    grid_histogram,
    rejected_corner_mask,
    row_lookup,
)


ARCSEC = 1.0 / 3600.0


def test_row_lookup_maps_identities_and_rejects_negatives():
    lookup = row_lookup(np.array([5, 2, 9], dtype=np.int64))
    assert lookup[5] == 0 and lookup[2] == 1 and lookup[9] == 2
    with pytest.raises(ValueError, match="non-negative"):
        row_lookup(np.array([-1, 3], dtype=np.int64))


def planted_scene():
    """One primary with a bright close neighbour, a faint one and a distant one."""
    return pd.DataFrame(
        {
            # a gap in the identities keeps the lookup sparse, so an anchor
            # naming an identity the scene does not carry can be caught
            "index": np.array([0, 1, 2, 3, 7], dtype=np.int64),
            # primary 0 keeps a bright neighbour at 1", primary 7 keeps none
            "RA": np.array([180.0, 180.0 + ARCSEC, 180.0 + ARCSEC, 180.0 + 5 * ARCSEC, 181.0]),
            "DEC": np.array([-0.5, -0.5, -0.5 + 0.5 * ARCSEC, -0.5, -0.5]),
            "r": np.array([22.0, 19.0, 23.0, 15.0, 22.0]),
        }
    )


def test_bright_neighbour_audit_flags_only_the_close_bright_pair():
    audit = bright_neighbour_audit(planted_scene(), np.array([0, 7], dtype=np.int64))
    primary = audit.loc[audit["input_index"] == 0].iloc[0]
    isolated = audit.loc[audit["input_index"] == 7].iloc[0]
    assert bool(primary["bright_neighbour"]) is True
    # the 19th magnitude object at 1" sets the maximum, not the 15th at 5"
    assert primary["brightest_neighbour_delta_mag"] == pytest.approx(3.0)
    # only the 1" pair and the 1.118" pair are inside the 3" radius
    assert primary["neighbours_within_radius"] == 2
    assert primary["nearest_neighbour_arcsec"] == pytest.approx(1.0, rel=1e-3)
    assert bool(isolated["bright_neighbour"]) is False
    assert isolated["neighbours_within_radius"] == 0
    assert not np.isfinite(isolated["nearest_neighbour_arcsec"])


def test_bright_neighbour_audit_rejects_anchor_outside_the_scene():
    with pytest.raises(RuntimeError, match="absent from the truth scene"):
        bright_neighbour_audit(planted_scene(), np.array([5], dtype=np.int64))
    with pytest.raises(RuntimeError, match="non-finite"):
        scene = planted_scene()
        scene.loc[1, "r"] = np.nan
        bright_neighbour_audit(scene, np.array([0], dtype=np.int64))


def test_grid_histogram_places_pairs_and_drops_the_outside():
    distance = np.array([0.75, 0.75, 12.0])
    delta_mag = np.array([2.0, 2.0, 2.0])
    counts = grid_histogram(distance, delta_mag)
    assert counts.shape == (len(DISTANCE_EDGES) - 1, len(DELTA_MAG_EDGES) - 1)
    # 0.75" is the second separation bin; a two-magnitude difference is the
    # bin just above the factor-five threshold
    assert counts[1, 5] == 2
    assert counts.sum() == 2
    weighted = grid_histogram(distance, delta_mag, weight=np.array([0.3, 0.7, 99.0]))
    assert weighted[1, 5] == pytest.approx(1.0)
    assert weighted.sum() == pytest.approx(1.0)


def test_rejected_corner_mask_is_closed_on_radius_and_open_on_magnitude():
    distance = np.array([3.0, 3.0001, 1.0, 1.0])
    delta_mag = np.array([2.0, 2.0, REJECTION_DELTA_MAG, REJECTION_DELTA_MAG + 1e-9])
    assert list(rejected_corner_mask(distance, delta_mag)) == [True, False, False, True]


def make_primaries(offset=0.0):
    return pd.DataFrame(
        {
            "input_index": np.arange(4, dtype=np.int64),
            "R_blend": np.array([0.1, 0.2, 0.8, 0.9]) + offset,
            "R_blend_rejected_corner": np.array([0.0, 0.0, 0.6, 0.7]),
            "pairs": np.array([10, 12, 16, 18], dtype=np.int64),
            "true_r_magnitude": np.array([22.0, 23.0, 24.0, 25.0]),
            "neighbours_within_radius": np.array([0, 1, 2, 3], dtype=np.int64),
            "bright_neighbour": np.array([False, False, True, True]),
            "class": np.array(["labelled", "labelled", "unlabelled", "unlabelled"]),
        }
    )


def test_case_summary_counts_every_class_and_the_cross_tabulation():
    summary = case_summary(make_primaries())
    assert summary["all_anchor_primaries"]["n"] == 4
    assert summary["labelled"]["n"] == 2
    assert summary["unlabelled"]["n"] == 2
    assert summary["unlabelled__bright_neighbour"]["n"] == 2
    assert summary["unlabelled__no_bright_neighbour"]["n"] == 0
    assert summary["labelled__bright_neighbour"]["n"] == 0
    assert summary["labelled"]["mean_R_blend"] == pytest.approx(0.15)
    assert summary["unlabelled"]["mean_R_blend"] == pytest.approx(0.85)
    # the class means recombine into the cohort mean at the class fractions
    assert 0.5 * 0.15 + 0.5 * 0.85 == pytest.approx(
        summary["all_anchor_primaries"]["mean_R_blend"]
    )
    assert summary["unlabelled"]["fraction_bright_neighbour"] == pytest.approx(1.0)
    assert np.isnan(summary["unlabelled__no_bright_neighbour"]["mean_R_blend"])


def test_bootstrap_classes_separates_complete_from_unevaluable():
    per_case = {
        0: case_summary(make_primaries(0.0)),
        1: case_summary(make_primaries(0.05)),
        2: case_summary(make_primaries(-0.05)),
    }
    summary = bootstrap_classes(per_case, n_boot=200, seed=3)
    entry = summary["unlabelled"]
    assert entry["status"] == "complete"
    assert entry["objects"] == 6
    assert entry["cases"] == 3
    assert entry["fraction_of_anchor"]["mean"] == pytest.approx(0.5)
    assert entry["mean_R_blend"]["mean"] == pytest.approx(0.85)
    assert entry["mean_R_blend"]["standard_error"] >= 0.0
    empty = summary["unlabelled__no_bright_neighbour"]
    assert empty["status"] == "not_evaluable_on_every_case"
    assert empty["cases_without_members"] == [0, 1, 2]
