import numpy as np
import pandas as pd

from scripts.diag_v22_q3_flow_trueflux import (
    case_stats, flux_classes, seeded_stats, summarize,
)


def test_flux_classes_separates_zero_and_positive_quartiles():
    values = np.r_[np.zeros(20), np.arange(1, 81, dtype=float)]
    classes, labels, edges, method = flux_classes(values)
    assert labels == ["zero", "Q1", "Q2", "Q3", "Q4"]
    assert method == "zero_plus_positive_quartiles"
    assert np.all(classes[:20] == 0)
    assert [int(np.sum(classes == index)) for index in range(1, 5)] == [20] * 4
    assert np.allclose(edges, [20.75, 40.5, 60.25])


def test_case_and_seed_uncertainties_use_independent_axes():
    frame = pd.DataFrame({
        "case": [0, 0, 1, 1], "truth": [1.0, 1.0, 3.0, 3.0],
        "R_flow_s1": [2.0, 2.0, 4.0, 4.0],
        "R_flow_s2": [4.0, 4.0, 6.0, 6.0],
    })
    truth = case_stats(frame, "truth")
    got = seeded_stats(frame, ["R_flow_s1", "R_flow_s2"], subtract="truth")
    assert np.isclose(truth["mean"], 2.0)
    assert np.isclose(truth["case_sem"], 1.0)
    assert np.isclose(got["mean"], 2.0)
    assert np.isclose(got["case_sem"], 0.0)
    assert np.isclose(got["seed_sem"], 1.0)
    assert np.isclose(got["quadrature_sem"], 1.0)


def test_summary_reports_radial_proxy_information_and_residual():
    rows = []
    for case in range(2):
        for index in range(50):
            near = 0.0 if index < 10 else float(index - 9)
            mid = float(index + 1)
            far = float(index + 51)
            row = {
                "case": case, "input_index": 100 * case + index,
                "flux_abs_near_0_1": near, "flux_abs_mid_1_3": mid,
                "logflux_abs_near_0_1": np.log10(1 + near),
                "logflux_abs_mid_1_3": np.log10(1 + mid),
                "logflux_abs_far_3_10": np.log10(1 + far),
                "nbr_flux_near": np.log10(1 + near + mid),
                "nbr_flux_far": np.log10(1 + far), "nbr_flux_max": np.log10(1 + far),
                "R_self_truth": 0.5 + 0.01 * near, "R_self_null": 0.0,
            }
            for seed in range(16):
                row[f"R_flow_s{seed}"] = row["R_self_truth"] - 0.02
            rows.append(row)
    frame = pd.DataFrame(rows)
    flow_columns = [f"R_flow_s{seed}" for seed in range(16)]
    result = summarize(frame, flow_columns)
    assert result["n_rows"] == 100
    assert np.isclose(result["global"]["flow_minus_self_truth"]["mean"], -0.02)
    assert len(result["shells"]["near"]["bins"]) == 5
    assert len(result["shells"]["far"]["bins"]) == 4
    assert result["v22_proxy_spearman"]["logflux_abs_far_3_10"]["nbr_flux_far"] > 0.99
