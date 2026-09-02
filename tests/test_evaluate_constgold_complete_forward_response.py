import numpy as np
import pandas as pd

from scripts.evaluate_constgold_complete_forward_response import (
    _experiment_comparison,
    _flow_seed_sem,
    _measured_leg,
)


def test_flow_seed_sem_is_null_for_single_checkpoint():
    assert _flow_seed_sem(np.array([[0.9, 0.0]])) is None
    assert np.allclose(
        _flow_seed_sem(np.array([[0.8, -0.1], [1.0, 0.1]])),
        [0.1, 0.1],
    )


def test_experiment_comparison_records_residual_and_bootstrap():
    measured = {"response_first_column": [0.9, -0.1], "plus_count": 12}
    predicted = {"response_first_column": [0.95, -0.08]}
    result = _experiment_comparison(
        measured=measured,
        predicted=predicted,
        bootstrap={
            "standard_error": [0.01, 0.02],
            "ci95": [[0.03, -0.02], [0.07, 0.06]],
        },
        definition="fixed population",
        model_components=("flow",),
    )

    assert result["measured"] is measured
    assert result["predicted"] is predicted
    assert result["model_components"] == ["flow"]
    assert np.allclose(
        result["predicted_minus_measured"]["response_first_column"],
        [0.05, 0.02],
    )
    assert result["predicted_minus_measured"][
        "case_bootstrap_standard_error"
    ] == [0.01, 0.02]


def test_measured_leg_reports_inner_join_exclusions(tmp_path):
    catalogue = tmp_path / "case0_0.02" / "real0" / "catalogues"
    cross_dir = catalogue / "CrossMatch"
    shape_dir = catalogue / "Shapes"
    cross_dir.mkdir(parents=True)
    shape_dir.mkdir(parents=True)
    pd.DataFrame(
        {"id_detec": [1, 2], "id_input": [10, 20]}
    ).to_feather(cross_dir / "tile180.0_-0.5_rot0_matched.feather")
    pd.DataFrame(
        {
            "NUMBER": [1, 3],
            "NGMIX_G1": [0.1, 0.2],
            "NGMIX_G2": [0.0, 0.0],
            "MAG_AUTO": [25.0, 25.0],
            "FLUX_RADIUS": [5.0, 5.0],
        }
    ).to_feather(
        shape_dir / "shape_catalogue_detect_position_all_tile180.0_-0.5.feather"
    )

    _, _, report, _, _ = _measured_leg(
        tmp_path,
        case=0,
        sign=0.02,
        active_ids=np.array([10]),
        abs_shape_max=0.6,
    )

    assert report["crossmatch_rows_without_shape"] == 1
    assert report["shape_rows_without_crossmatch"] == 1
    assert report["crossmatched_rows_outside_finite_prior_support"] == 0
    assert report["actual_detected_count"] == 1
