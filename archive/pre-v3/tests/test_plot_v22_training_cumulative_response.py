import numpy as np

from scripts.plot_v22_training_cumulative_response import (
    accumulate_axis,
    summarize_axis,
)


def test_axis_accumulation_and_additive_closure() -> None:
    accumulator = {
        "counts": np.zeros((2, 2), dtype=np.int64),
        "label_sum": np.zeros((2, 2), dtype=float),
        "prediction_sum": np.zeros((2, 2), dtype=float),
        "null_sum": np.zeros((2, 2), dtype=float),
        "value_sum": np.zeros((2, 2), dtype=float),
    }
    values = np.array([0.2, 1.2, 0.4, 1.4])
    case_index = np.array([0, 0, 1, 1])
    label = np.array([1.0, 2.0, 3.0, 4.0])
    prediction = np.array([0.5, 1.5, 2.5, 3.5])
    null = np.zeros(4)
    edges = np.array([0.0, 1.0, 2.0])
    accumulate_axis(
        accumulator,
        values,
        edges,
        case_index,
        label,
        prediction,
        null,
    )
    profile = summarize_axis(accumulator, np.array([2, 4]), edges)
    assert profile["total"]["n_pairs"] == 4
    assert profile["total"]["n_primaries"] == 6
    np.testing.assert_allclose(
        profile["total"]["additive_label_per_primary"]["mean"],
        np.mean([3.0 / 2.0, 7.0 / 4.0]),
    )
    np.testing.assert_allclose(
        sum(
            item["additive_label_minus_prediction_per_primary"]["mean"]
            for item in profile["bins"]
        ),
        profile["total"]["additive_label_minus_prediction_per_primary"]["mean"],
    )


def test_axis_accumulator_rejects_out_of_range_values() -> None:
    accumulator = {
        field: np.zeros((2, 2), dtype=(np.int64 if field == "counts" else float))
        for field in (
            "counts", "label_sum", "prediction_sum", "null_sum", "value_sum",
        )
    }
    try:
        accumulate_axis(
            accumulator,
            np.array([-0.1, 0.2]),
            np.array([0.0, 1.0, 2.0]),
            np.array([0, 1]),
            np.ones(2),
            np.ones(2),
            np.zeros(2),
        )
    except RuntimeError as error:
        assert "outside plot edges" in str(error)
    else:
        raise AssertionError("out-of-range values were accepted")
