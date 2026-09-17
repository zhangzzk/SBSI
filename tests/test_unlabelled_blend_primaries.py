import numpy as np
import pandas as pd
import pytest

from scripts.diagnose_unlabelled_blend_primaries import (
    bootstrap,
    case_partition_means,
    pack_keys,
    partition_lookup,
    pool,
)


def lookup_frame():
    return pd.DataFrame(
        {
            "case": np.array([0, 0, 0, 1, 1, 1], dtype=np.int64),
            "input_index": np.array([1, 2, 3, 1, 2, 3], dtype=np.int64),
            "R_blend": np.array([0.1, 0.2, 1.0, 0.1, 0.2, 1.0]),
        }
    )


def test_pack_keys_separates_cases():
    assert pack_keys([0], [1])[0] != pack_keys([1], [1])[0]


def test_partition_marks_only_labelled_primaries():
    labelled = np.unique(pack_keys([0, 0, 1, 1], [1, 2, 1, 2]))
    frame = partition_lookup(lookup_frame(), labelled)
    assert frame["has_label"].tolist() == [True, True, False, True, True, False]


def test_pool_splits_the_cohort_mean_additively():
    labelled = np.unique(pack_keys([0, 0, 1, 1], [1, 2, 1, 2]))
    summary = pool(list(case_partition_means(
        partition_lookup(lookup_frame(), labelled)).values()))
    assert summary["all"]["mean_R_blend"] == pytest.approx(2.6 / 6)
    assert summary["labelled"]["mean_R_blend"] == pytest.approx(0.15)
    assert summary["unlabelled"]["mean_R_blend"] == pytest.approx(1.0)
    assert summary["unlabelled"]["row_fraction"] == pytest.approx(1 / 3)
    total = (
        summary["labelled"]["contribution_to_cohort_mean"]
        + summary["unlabelled"]["contribution_to_cohort_mean"]
    )
    assert total == pytest.approx(summary["all"]["mean_R_blend"])


def test_pool_handles_an_empty_partition():
    labelled = np.unique(pack_keys([0, 0, 0, 1, 1, 1], [1, 2, 3, 1, 2, 3]))
    summary = pool(list(case_partition_means(
        partition_lookup(lookup_frame(), labelled)).values()))
    assert summary["unlabelled"]["n"] == 0
    assert summary["unlabelled"]["mean_R_blend"] is None
    assert summary["labelled"]["mean_R_blend"] == pytest.approx(2.6 / 6)


def test_bootstrap_returns_finite_spreads():
    labelled = np.unique(pack_keys([0, 0, 1, 1], [1, 2, 1, 2]))
    per_case = case_partition_means(partition_lookup(lookup_frame(), labelled))
    errors = bootstrap(per_case, n_boot=32, seed=1)
    assert errors["all"] >= 0.0
    assert errors["unlabelled"] == pytest.approx(0.0)
