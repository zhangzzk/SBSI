import json

import pandas as pd
import pytest

from scripts.plot_anchor_bias_emulator_constgold_calibration import read_inputs


def write_summary(path, n_rows=6, n_cases=2):
    path.write_text(json.dumps({
        "population": {"n_rows": n_rows, "n_cases": n_cases},
        "conditional_transfer": {
            "bin_mean_slope_total_gap_on_predicted_blend_bias": 0.1,
        },
    }))


def test_read_inputs_checks_population_and_sorts_prediction(tmp_path):
    table = pd.DataFrame({
        "bin": [0, 1],
        "n_rows": [3, 3],
        "n_cases": [2, 2],
        "raw_gap_mean": [0.2, -0.1],
        "raw_gap_case_sem": [0.02, 0.03],
        "predicted_bias_mean": [0.1, -0.2],
        "predicted_bias_case_sem": [0.001, 0.001],
        "corrected_gap_mean": [0.1, 0.1],
        "corrected_gap_case_sem": [0.02, 0.03],
    })
    csv_path = tmp_path / "bins.csv"
    json_path = tmp_path / "summary.json"
    table.to_csv(csv_path, index=False)
    write_summary(json_path)
    got, _ = read_inputs(str(csv_path), str(json_path))
    assert got.predicted_bias_mean.tolist() == [-0.2, 0.1]


def test_read_inputs_rejects_row_count_mismatch(tmp_path):
    table = pd.DataFrame({
        "bin": [0], "n_rows": [3], "n_cases": [2],
        "raw_gap_mean": [0.2], "raw_gap_case_sem": [0.02],
        "predicted_bias_mean": [0.1], "predicted_bias_case_sem": [0.001],
        "corrected_gap_mean": [0.1], "corrected_gap_case_sem": [0.02],
    })
    csv_path = tmp_path / "bins.csv"
    json_path = tmp_path / "summary.json"
    table.to_csv(csv_path, index=False)
    write_summary(json_path, n_rows=4)
    with pytest.raises(RuntimeError, match="row count"):
        read_inputs(str(csv_path), str(json_path))
