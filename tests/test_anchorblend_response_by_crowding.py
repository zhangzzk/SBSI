"""Tests for the crowding-binned read of the primary-unsheared scene response."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.diagnose_anchorblend_response_by_crowding import (
    H,
    bin_edges,
    case_responses,
    leg_rows,
    summarize,
)


def primaries(blend, *, case, stratum, weight, m=0.0, intrinsic=0.0,
              in_plus=True, in_minus=True):
    """Rows whose measured legs realise ``m`` exactly on a planted ``blend``.

    A leg measures ``intrinsic + direction * h * R * (1 + m)`` while the model
    predicts ``R``, so the recovered response ratio is ``1 + m`` by
    construction, whatever the intrinsic shape or the weights.
    """
    blend = np.asarray(blend, dtype=float)
    size = len(blend)
    shift = H * blend * (1.0 + m)
    return pd.DataFrame({
        "case": np.full(size, case, dtype=np.int32),
        "stratum": np.full(size, stratum, dtype=np.int8),
        "weight": np.full(size, weight, dtype=float),
        "input_index": np.arange(size) + 1000 * stratum + 1_000_000 * case,
        "R_scene": blend,
        "R_scene_abs": np.abs(blend),
        "pairs": np.full(size, 4),
        "r_input_p": np.full(size, 24.0),
        "e1_plus": intrinsic + shift if in_plus else np.nan,
        "e2_plus": np.zeros(size) if in_plus else np.nan,
        "e1_minus": intrinsic - shift if in_minus else np.nan,
        "e2_minus": np.zeros(size) if in_minus else np.nan,
        "matched": np.full(size, in_plus and in_minus),
    })


def binned(frame, edges):
    out = frame.copy()
    out["bin"] = np.searchsorted(np.asarray(edges), out["R_scene_abs"].to_numpy(), side="right")
    return out


def test_bin_edges_follow_the_weight_not_the_row_count():
    #  ten cheap rows below 1 and ten expensive rows above: by row count the
    #  median sits at 1, but the heavy rows carry four fifths of the population
    #  so the weighted median has to move up into them.
    values = np.concatenate((np.linspace(0.0, 0.9, 10), np.linspace(1.0, 1.9, 10)))
    weights = np.concatenate((np.ones(10), np.full(10, 4.0)))
    unweighted = bin_edges(values, np.ones(20), 4)
    weighted = bin_edges(values, weights, 4)
    #  an edge sits on the last value below it, so ten rows fall on each side
    assert unweighted[2] == pytest.approx(0.9)
    assert (values <= unweighted[2]).sum() == 10
    #  by weight the halfway point is well inside the expensive rows
    assert weighted[2] > 1.2
    assert weights[values <= weighted[2]].sum() == pytest.approx(0.5 * weights.sum(),
                                                                rel=0.05)
    assert weighted[0] == -np.inf and weighted[-1] == np.inf
    assert np.all(np.diff(weighted) > 0)


def test_bin_edges_reject_a_degenerate_column():
    with pytest.raises(ValueError, match="not strictly increasing"):
        bin_edges(np.zeros(100), np.ones(100), 4)
    with pytest.raises(ValueError, match="three bins"):
        bin_edges(np.linspace(0, 1, 100), np.ones(100), 2)


def test_case_responses_recover_a_planted_per_bin_error():
    #  the quiet half is predicted perfectly, the crowded half is 10% off
    frames = []
    for case in (1, 2, 3):
        frames.append(primaries([0.1, 0.2], case=case, stratum=0, weight=1.0, m=0.0))
        frames.append(primaries([1.0, 1.2], case=case, stratum=0, weight=1.0, m=0.10))
    frame = binned(pd.concat(frames, ignore_index=True), [0.5])
    out = case_responses(frame, "matched", 2)
    rows = summarize(out, ["quiet", "loud"], replicates=50, seed=1)
    assert rows[0]["m_percent"] == pytest.approx(0.0, abs=1e-9)
    assert rows[1]["m_percent"] == pytest.approx(10.0)
    assert rows[0]["R_emulator"] == pytest.approx(0.15)
    assert rows[1]["R_sim"] == pytest.approx(1.1 * 1.10)
    assert rows[0]["objects"] == 6 and rows[1]["objects"] == 6


def test_strata_pool_inside_a_case_by_their_inverse_sampling_weight():
    #  identical responses, very different weights: the pooled prediction must
    #  be the weighted mean, so a plain row average would land elsewhere
    frames = []
    for case in (1, 2):
        frames.append(primaries([0.2], case=case, stratum=0, weight=1.0))
        frames.append(primaries([1.0], case=case, stratum=1, weight=9.0))
    frame = binned(pd.concat(frames, ignore_index=True), [])
    out = case_responses(frame, "matched", 1)
    assert out["R_emulator"][0, 0] == pytest.approx((1.0 * 0.2 + 9.0 * 1.0) / 10.0)
    assert out["R_emulator"][0, 0] != pytest.approx(0.6)


def test_the_measured_and_predicted_response_agree_when_the_model_is_right():
    #  the whole point of the instrument: with no planted error, and a large
    #  intrinsic shape that must cancel between the legs, m is exactly zero
    frame = binned(pd.concat(
        [primaries([0.1, 0.5, 1.4], case=case, stratum=0, weight=2.0, intrinsic=0.3)
         for case in range(5)], ignore_index=True), [])
    rows = summarize(case_responses(frame, "matched", 1), ["all"], replicates=50, seed=2)
    assert rows[0]["m_percent"] == pytest.approx(0.0, abs=1e-9)
    assert rows[0]["R_sim_null_21"] == pytest.approx(0.0, abs=1e-12)


def test_unmatched_scores_each_leg_on_its_own_selection():
    kept = primaries([0.4], case=1, stratum=0, weight=1.0)
    plus_only = primaries([0.4], case=1, stratum=0, weight=1.0, in_minus=False)
    plus_only["input_index"] += 77
    frame = pd.concat([kept, plus_only], ignore_index=True)
    assert list(leg_rows(frame, "matched", "plus")) == [True, False]
    assert list(leg_rows(frame, "unmatched", "plus")) == [True, True]
    assert list(leg_rows(frame, "unmatched", "minus")) == [True, False]


def test_share_of_total_blending_is_response_weighted_not_object_weighted():
    #  one crowded object against nine quiet ones: it is a tenth of the
    #  population but carries most of the blending, which is the number that
    #  decides whether a tail failure matters
    quiet = primaries(np.full(9, 0.05), case=1, stratum=0, weight=1.0)
    loud = primaries([4.5], case=1, stratum=0, weight=1.0)
    loud["input_index"] += 99
    frames = []
    for case in (1, 2):
        for part in (quiet, loud):
            copy = part.copy()
            copy["case"] = np.int32(case)
            frames.append(copy)
    frame = binned(pd.concat(frames, ignore_index=True), [1.0])
    rows = summarize(case_responses(frame, "matched", 2), ["quiet", "loud"],
                     replicates=20, seed=3)
    assert rows[0]["objects"] == 18 and rows[1]["objects"] == 2
    assert rows[1]["share_of_total_blending_percent"] == pytest.approx(
        100.0 * 4.5 / (4.5 + 9 * 0.05))


def test_case_responses_refuse_a_cell_a_case_does_not_populate():
    frame = binned(pd.concat([
        primaries([0.1], case=1, stratum=0, weight=1.0),
        primaries([2.0], case=2, stratum=0, weight=1.0),
    ], ignore_index=True), [1.0])
    with pytest.raises(RuntimeError, match="empty"):
        case_responses(frame, "matched", 2)
