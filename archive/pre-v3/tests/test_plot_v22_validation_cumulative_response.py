import numpy as np
from sklearn.model_selection import train_test_split

from scripts.plot_v22_validation_cumulative_response import (
    exact_validation_mask,
    row_validation_summary,
)


def test_exact_validation_mask_replays_sklearn_indices():
    mask = exact_validation_mask(31, test_size=0.2, random_state=321)
    indices = np.arange(31, dtype=np.int64)
    train, validation = train_test_split(
        indices, test_size=0.2, random_state=321
    )
    assert mask.dtype == bool
    assert mask.sum() == len(validation) == 7
    assert not mask[train].any()
    assert mask[validation].all()


def test_row_validation_summary_replays_means_and_r2():
    label = np.array([-1.0, 0.0, 1.0, 2.0])
    prediction = np.array([-0.5, 0.0, 0.5, 2.0])
    got = row_validation_summary(
        len(label),
        float(label.sum()),
        float(np.square(label).sum()),
        float(prediction.sum()),
        float(np.square(label - prediction).sum()),
    )
    expected_r2 = 1.0 - np.square(label - prediction).sum() / np.square(
        label - label.mean()
    ).sum()
    assert np.isclose(got["label_mean"], label.mean())
    assert np.isclose(got["prediction_mean"], prediction.mean())
    assert np.isclose(
        got["label_minus_prediction_mean"],
        (label - prediction).mean(),
    )
    assert np.isclose(got["r2"], expected_r2)
