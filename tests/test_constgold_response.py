import numpy as np

from sbsi.constgold_response import KeyedLookup, _CaseAccumulator


def test_keyed_lookup_reports_unmatched_rows():
    lookup = KeyedLookup(
        np.array([(40 << 32) | 2, (41 << 32) | 3], dtype=np.int64),
        {"value": np.array([1.25, 2.5])},
    )
    matched, values = lookup.match([40, 40, 41], [2, 9, 3])
    assert matched.tolist() == [True, False, True]
    np.testing.assert_allclose(values["value"][matched], [1.25, 2.5])


def test_case_accumulator_matrix_and_projected_response():
    angles = np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False)
    applied = 0.02 * np.column_stack((np.cos(angles), np.sin(angles)))
    simulation_matrix = np.array([[0.82, 0.03], [-0.01, 0.78]])
    flow_matrix = np.array([[0.74, 0.02], [-0.02, 0.76]])
    simulation_delta = (simulation_matrix @ (2.0 * applied).T).T
    flow_delta = (flow_matrix @ (2.0 * applied).T).T
    accumulator = _CaseAccumulator()
    accumulator.add(simulation_delta, flow_delta, applied, np.full(len(applied), 0.08))
    result = accumulator.summary()
    np.testing.assert_allclose(result["simulation_matrix"], simulation_matrix, atol=1e-14)
    np.testing.assert_allclose(result["flow_matrix"], flow_matrix, atol=1e-14)
    np.testing.assert_allclose(
        result["model_matrix"], flow_matrix + 0.08 * np.eye(2), atol=1e-14
    )
    assert np.isclose(result["simulation_response"], np.trace(simulation_matrix) / 2)
    assert np.isclose(result["flow_response"], np.trace(flow_matrix) / 2)


def test_case_accumulator_does_not_invent_unexcited_matrix_column():
    applied = np.tile([0.02, 0.0], (16, 1))
    simulation_delta = np.tile([0.032, -0.004], (16, 1))
    flow_delta = np.tile([0.030, -0.002], (16, 1))
    accumulator = _CaseAccumulator()
    accumulator.add(simulation_delta, flow_delta, applied, np.full(16, 0.05))
    result = accumulator.summary()
    assert result["applied_shear_matrix_rank"] == 1
    assert result["simulation_matrix"] is None
    assert result["flow_matrix"] is None
    assert result["model_matrix"] is None
    assert np.isclose(result["simulation_response"], 0.8)
    assert np.isclose(result["simulation_cross_response"], -0.1)
