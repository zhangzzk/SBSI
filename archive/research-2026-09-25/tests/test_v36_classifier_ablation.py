"""Non-degradation decisions must fail closed on point loss or uncertainty."""
import numpy as np

from scripts.summarize_v36_classifier_ablation import gate


def test_worse_point_is_rejected_even_within_margin():
    assert not gate(.01, np.full(100, .01), .05)['passed']


def test_better_point_is_insufficient_with_large_uncertainty():
    assert not gate(-.01, np.linspace(-.2, .2, 1000), .05)['passed']


def test_all_endpoints_must_pass():
    samples = np.tile([-.02, .001], (100, 1))
    assert not gate([-.02, .001], samples, .05)['passed']


def test_unchanged_or_precisely_better_passes():
    assert gate(0, np.zeros(100), .05)['passed']
    assert gate(-.02, np.linspace(-.03, -.01, 100), .05)['passed']


def test_nonfinite_is_rejected():
    assert not gate(float('nan'), np.zeros(100), .05)['passed']
