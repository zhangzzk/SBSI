import numpy as np
import pytest

from scripts.compare_fixed_g0_models import (
    case_summary,
    pair_component_scores,
    parse_models,
    response_mc_difference_se,
)


def test_pair_component_scores_recovers_deterministic_quadratic():
    residual = np.array([[1.0, 2.0, 3.0], [-2.0, 1.0, 4.0]])
    grouped = np.repeat(residual[None, :, :], 4, axis=0)
    np.testing.assert_allclose(pair_component_scores(grouped), residual**2 / 4.0)


def test_case_summary_keeps_pooled_and_case_balanced_estimands_distinct():
    values = np.array([0.0, 0.0, 3.0])
    cases = np.array([1, 1, 2])
    result = case_summary(values, cases)
    assert result["pooled_mean"] == pytest.approx(1.0)
    assert result["case_balanced_mean"] == pytest.approx(1.5)
    assert result["case_sem"] == pytest.approx(1.5)
    assert result["n_cases"] == 2


def test_paired_mc_difference_is_zero_for_identical_models():
    residuals = np.arange(4 * 5 * 3, dtype=float).reshape(4, 5, 3)
    result = response_mc_difference_se(residuals, residuals.copy())
    assert result == {"total": 0.0, "shape": 0.0, "radius": 0.0}


def test_parse_models_rejects_duplicates_and_missing_files(tmp_path):
    model = tmp_path / "model.pt"
    model.write_bytes(b"checkpoint")
    assert parse_models([f"first={model}", f"second={model}"]) == {
        "first": model.resolve(),
        "second": model.resolve(),
    }
    with pytest.raises(ValueError, match="duplicate"):
        parse_models([f"same={model}", f"same={model}"])
    with pytest.raises(FileNotFoundError, match="missing"):
        parse_models([f"first={model}", f"second={tmp_path / 'missing.pt'}"])
