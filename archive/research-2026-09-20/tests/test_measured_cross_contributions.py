import numpy as np

from scripts.audit_measured_cross_contributions import contributions


def test_block_jackknife_uses_object_weights_and_retains_tails():
    counts = np.arange(1, 65)
    blocks = np.repeat(np.arange(64), counts)
    values = np.repeat(np.linspace(-2, 3, 64), counts)
    report = contributions(values, np.arange(len(values)), blocks)
    expected = np.asarray([values[blocks != i].mean() for i in range(64)])
    np.testing.assert_allclose(report['leave_one_block_out'], expected)
    np.testing.assert_allclose(report['mean'], values.mean())
    np.testing.assert_allclose(report['positive_contribution']+report['negative_contribution'], values.mean())
    np.testing.assert_allclose(report['block_jackknife_se'], np.sqrt(63/64*((expected-expected.mean())**2).sum()))
