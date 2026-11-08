"""Interacting walkers must not inflate the number of independent samples."""
import importlib
from pathlib import Path

import numpy as np
import pytest


def test_identical_coupled_walkers_do_not_reduce_mcse(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'scripts'))
    report = importlib.import_module('report_galsbi_ensemble')
    values = np.random.default_rng(202).normal(0, .2, (2, 4000, 1))
    one = report.ensemble_profile(values)
    four = report.ensemble_profile(np.repeat(values, 4, axis=2))
    assert four['relative_mcse'] == pytest.approx(one['relative_mcse'], rel=1e-10)
    assert four['diagnostics']['ess_bulk'] == pytest.approx(one['diagnostics']['ess_bulk'], rel=1e-10)


def test_different_independent_ensembles_flag_disagreement(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'scripts'))
    report = importlib.import_module('report_galsbi_ensemble')
    values = np.random.default_rng(203).normal(0, .2, (2, 4000, 4))
    values[1] += 1.
    result = report.ensemble_profile(values)
    assert result['diagnostics']['rhat'] > 1.1
    assert result['replica_log_mean_ratio'][1]-result['replica_log_mean_ratio'][0] > .9
