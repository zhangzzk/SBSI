import numpy as np

from _script_loader import load_script_module


TOPDIFF = load_script_module("plot_flow_likelihood_top_differences.py")


def test_sliced_wasserstein_is_zero_for_identical_samples():
    rng = np.random.default_rng(41)
    samples = rng.normal(size=(5, 64, 2)).astype(np.float32)
    score = TOPDIFF.sliced_wasserstein_1(samples, samples, directions=16)
    np.testing.assert_array_equal(score, 0.0)


def test_sliced_wasserstein_detects_larger_translation():
    rng = np.random.default_rng(42)
    baseline = rng.normal(size=(4, 128, 2)).astype(np.float32)
    shifted = baseline.copy()
    shifted[1] += (0.1, 0.0)
    shifted[2] += (0.0, 0.2)
    shifted[3] += (0.3, 0.4)
    score = TOPDIFF.sliced_wasserstein_1(baseline, shifted, directions=32)
    assert score[0] == 0.0
    assert 0.0 < score[1] < score[2] < score[3]


def test_trimmed_sliced_wasserstein_ignores_single_extreme_tail_draw():
    baseline = np.zeros((1, 100, 2), dtype=np.float32)
    shifted = baseline.copy()
    shifted[0, -1] = (1_000.0, -1_000.0)
    untrimmed = TOPDIFF.sliced_wasserstein_1(
        baseline, shifted, directions=16
    )
    trimmed = TOPDIFF.sliced_wasserstein_1(
        baseline, shifted, directions=16, trim_fraction=0.01
    )
    assert untrimmed[0] > 0.0
    assert trimmed[0] == 0.0


def test_display_extent_is_robust_and_bounded():
    samples = np.zeros((2, 32, 2), dtype=np.float32)
    assert TOPDIFF._display_extent(samples, samples) == 1.0
    samples[:, :, 0] = 5.0
    assert TOPDIFF._display_extent(samples, samples) == 2.0
