import numpy as np
import pandas as pd
import pytest

from scripts.diagnose_selection_boost_response import (
    BOOST_LABELS,
    TRUTH_MAG_LABELS,
    assign_bins,
    bootstrap,
    case_cell_sums,
    combine,
    match_keys,
    pooled_statistics,
    stack_cases,
)


def make_frame(n=400, offset=0.0, slope=0.0, seed=0):
    rng = np.random.default_rng(seed)
    true_mag = rng.uniform(22.0, 27.0, n)
    boost = rng.normal(0.5, 0.8, n)
    measured = rng.normal(0.3, 0.05, n)
    return pd.DataFrame(
        {
            "case": np.zeros(n, dtype=np.int32),
            "input_index": np.arange(n, dtype=np.int64),
            "true_r_magnitude": true_mag,
            "g0_MAG_AUTO": true_mag - boost,
            "R_self_measured": measured,
            "R_self_model": measured + offset + slope * boost,
        }
    )


def test_assign_bins_defines_boost_and_residual():
    frame = assign_bins(make_frame(offset=0.01))
    assert np.allclose(
        frame["boost"], frame["true_r_magnitude"] - frame["g0_MAG_AUTO"]
    )
    assert np.allclose(frame["residual"], 0.01)
    assert set(frame["truth_bin"].unique()) <= set(TRUTH_MAG_LABELS)
    assert set(frame["boost_bin"].unique()) <= set(BOOST_LABELS)


def test_cell_sums_cover_cells_marginals_and_cohort():
    keys = case_cell_sums(assign_bins(make_frame()))
    assert "all" in keys
    assert any(k.startswith("truth=") and "|boost=" in k for k in keys)
    assert any(k.startswith("boost=") for k in keys)
    assert keys["all"]["n"] == 400


def test_combine_recovers_a_flat_offset_with_zero_slope():
    summary = combine([case_cell_sums(assign_bins(make_frame(offset=0.02, seed=s)))
                       for s in range(4)])
    assert summary["all"]["mean_residual"] == pytest.approx(0.02, abs=1e-9)
    assert summary["all"]["residual_boost_slope"] == pytest.approx(0.0, abs=1e-9)


def test_combine_recovers_a_planted_boost_slope():
    summary = combine([case_cell_sums(assign_bins(make_frame(slope=0.05, seed=s)))
                       for s in range(4)])
    assert summary["all"]["residual_boost_slope"] == pytest.approx(0.05, abs=1e-9)
    for key, entry in summary.items():
        if key.startswith("truth=") and "|boost=" not in key and entry["n"] > 50:
            assert entry["residual_boost_slope"] == pytest.approx(0.05, abs=1e-9)


def test_stack_cases_is_dense_and_marks_absent_cells():
    sparse = make_frame(n=60, seed=1)
    sparse = sparse.loc[sparse["true_r_magnitude"] < 24.0]
    per_case = [case_cell_sums(assign_bins(make_frame(seed=2))),
                case_cell_sums(assign_bins(sparse))]
    keys, table = stack_cases(per_case)
    assert table.shape == (2, len(keys), 7)
    assert (table[1, :, 0] == 0).any()


def test_pooled_statistics_flags_absent_keys():
    totals = np.zeros((2, 7))
    totals[0] = [10.0, 3.0, 3.1, 0.1, 5.0, 5.0, 0.05]
    stats = pooled_statistics(totals)
    assert stats["present"].tolist() == [True, False]
    assert stats["mean_residual"][0] == pytest.approx(0.01)


def test_bootstrap_reports_errors_and_draw_counts():
    per_case = [case_cell_sums(assign_bins(make_frame(offset=0.02, seed=s)))
                for s in range(5)]
    errors = bootstrap(per_case, n_boot=64, seed=3)
    assert errors["all"]["bootstrap_draws_with_rows"] == 64
    assert errors["all"]["mean_residual"] >= 0.0


def test_match_keys_rejects_duplicate_keys():
    with pytest.raises(RuntimeError, match="not unique"):
        match_keys([0, 0], [1, 1], [0], [1])
