"""Bright-cut tests must preserve selection and refuse unbracketed recovery."""
import importlib
import json
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def math(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'scripts'))
    return importlib.import_module('galsbi_bright_math')


def test_bright_cuts_keep_first_rows_and_overlap_without_oversampling(math):
    union, cohorts, complete = math.bright_cohorts([21., 23., 19., 18., 21.5, 17.], [20., 22.], 2)
    assert complete
    np.testing.assert_array_equal(union, [0, 2, 3])
    assert cohorts == {'20.0': [1, 2], '22.0': [0, 1]}
    _, _, complete = math.bright_cohorts([20., 21.], [20., 22.], 1)
    assert not complete  # Strict measured cut; no replacement by a faint row.


def test_selection_factor_recovers_known_gaussian_peak(math):
    grid = np.linspace(-.1, .1, 101)
    truth, width, n = .017, .012, 128
    log_selected = -.5*((grid-truth)/width)**2
    z = np.exp(.8*grid+3*grid**2)*.02
    numerator = log_selected+n*np.log(z)
    result = math.profile_fit(grid, math.selected_log_profile(numerator, z, n, 50))
    assert result['valid']
    assert result['estimate'] == pytest.approx(truth, abs=1e-12)
    assert result['curvature_se'] == pytest.approx(width, abs=1e-12)
    assert result['intervals']['95'][0] < truth < result['intervals']['95'][1]
    assert abs(math.profile_fit(grid, numerator)['estimate']-truth) > .01


def test_boundary_peak_and_missing_selection_cannot_pass(math):
    assert not math.profile_fit([0., .01, .02], [0., 1., 2.])['valid']
    with pytest.raises(ValueError, match='positive selection'):
        math.selected_log_profile([0., 1.], [0., .1], 128, 0)


def test_interval_contains_off_grid_quadratic_peak(math):
    x = np.array([0., .01, .02])
    y = -.5*((x-.007)/.001)**2
    fit = math.profile_fit(x, y)
    assert fit['estimate'] == pytest.approx(.007)
    assert fit['intervals']['68'] == pytest.approx([.006, .008])
    assert fit['intervals']['95'][0] < fit['estimate'] < fit['intervals']['95'][1]


def test_peak_uncertainty_retains_profile_covariance(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'scripts'))
    report = importlib.import_module('report_galsbi_bright')
    x = np.linspace(-.02, .04, 31)
    y = -.5*((x-.0133)/.01)**2
    indices, gradient = report.peak_gradient(x, y)
    # A constant likelihood offset has no effect on a shear fit.
    assert gradient.sum() == pytest.approx(0., abs=1e-12)
    for i, derivative in zip(indices, gradient):
        plus, minus = y.copy(), y.copy()
        plus[i] += 1e-5
        minus[i] -= 1e-5
        difference = (report.profile_fit(x, plus)['estimate']-report.profile_fit(x, minus)['estimate'])/2e-5
        assert difference == pytest.approx(derivative, rel=1e-6)


def test_explicit_population_reference_needs_no_cohorts(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'scripts'))
    run = importlib.import_module('run_galsbi_bright')
    reference = tmp_path/'population'
    reference.mkdir()
    (reference/'observations.parquet').write_bytes(b'rows')
    receipt = dict(preparation_checks_pass=True, observations_sha256=run.digest(reference/'observations.parquet'))
    (reference/'result.json').write_text(json.dumps(receipt))
    assert run.selected_reference(tmp_path/'study', reference)[0] == reference
    with pytest.raises(FileNotFoundError):
        run.selected_reference(tmp_path)
    (reference/'observations.parquet').write_bytes(b'changed')
    with pytest.raises(ValueError):
        run.selected_reference(tmp_path/'study', reference)
