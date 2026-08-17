import numpy as np

from scripts.toy_random_individual_projection import summarize


def test_summarize_uses_draw_level_sem():
    values = np.array([1.0, 2.0, 3.0, 4.0])
    got = summarize(values)
    assert got["mean"] == 2.5
    np.testing.assert_allclose(got["direction_sd"], values.std(ddof=1))
    np.testing.assert_allclose(
        got["direction_sem"], values.std(ddof=1) / np.sqrt(len(values)),
    )
    assert got["n_draw"] == 4


def test_summarize_reports_distribution_quantiles():
    got = summarize(np.arange(1.0, 9.0))
    np.testing.assert_allclose(
        got["percentiles_2p5_50_97p5"], np.percentile(np.arange(1.0, 9.0), [2.5, 50, 97.5]),
    )
