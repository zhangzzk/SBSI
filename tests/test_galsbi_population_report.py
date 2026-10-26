"""Aggregate diagnostics must retain covariance and reject incomplete panels."""
import importlib
import json
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def report(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'scripts'))
    return importlib.import_module('report_galsbi_population')


def test_aggregate_mcse_retains_same_observation_noise_covariance(report):
    rng = np.random.default_rng(309)
    values = rng.normal(0, .1, (4000, 4))
    mean, phi = report.aggregate_profile(values)
    assert abs(mean-.005) < .003
    assert abs(phi.mean()) < 1e-12
    # Two perfectly correlated contributions have twice, not sqrt(2), the MCSE.
    assert report.conditional_mcse(2*phi) == pytest.approx(2*report.conditional_mcse(phi), rel=1e-8)
    assert report.conditional_mcse(phi-phi) == 0.


def test_aggregate_profile_rejects_nonfinite_draws(report):
    with pytest.raises(ValueError, match='finite'):
        report.aggregate_profile(np.array([[0., np.nan]]))


def test_missing_population_shards_cannot_complete(report, tmp_path):
    (tmp_path/'plan.json').write_text(json.dumps(dict(observations=20000, batch_observations=128)))
    with pytest.raises(ValueError, match='157 missing'):
        report.combine(tmp_path)
    assert json.loads((tmp_path/'report.json').read_text())['status'] == 'incomplete'


def test_population_retention_does_not_select_magnitude_strata(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'scripts'))
    module = importlib.import_module('check_galsbi_summary_sampler')
    import pandas as pd
    from types import SimpleNamespace
    magnitude = np.array([23., 23.2, 24., 26., 22.])
    draws = np.column_stack([np.zeros(5), np.zeros(5), [4., 4., 4., 4., 2.], 10**(.4*(30-magnitude))])
    flow = SimpleNamespace(device='cpu', target_transform=SimpleNamespace(target_names=['e1', 'e2', 'size', 'flux']),
        sample_with_shape_log_gap=lambda context, batch_size: (draws[:, None].copy(), np.zeros((5, 1))))
    zero = pd.DataFrame(dict(e1_input_p=np.zeros(5), e2_input_p=np.zeros(5)))
    monkeypatch.setattr(module, 'predict_detection_probability', lambda detector, context: np.ones(5))
    monkeypatch.setattr(module, 'transport_with_log_gap', lambda shape, gap, response: (shape, gap, None))
    response = SimpleNamespace(predict=lambda ids, size, flux: (np.zeros(len(ids)), None))
    observed, receipt, ids = module.mock_panel(zero, response, flow, None, seed=17,
        shear=[.02, 0.], retain_population=True)
    assert len(observed) == 3
    np.testing.assert_array_equal(ids, [0, 1, 2])
    assert receipt['n_retained'] == 3
