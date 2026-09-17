"""Focused arithmetic tests for the generated-radius flow diagnostic."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from scripts.diagnose_flow_generated_radius_coupling import (
    CaseAccumulator,
    projected_components,
    radius_bin_indices,
    radius_labels,
    summarize_cases,
    summarize_postcut,
)


EDGES = (0.6, 0.6154, 0.6298)


def test_radius_bins_are_left_closed_at_the_fixed_g0_boundary():
    values = np.array([0.5999, 0.6, 0.615399, 0.6154, 0.6298, 2.0])
    expected = [-1, 0, 0, 1, 2, 2]
    assert radius_bin_indices(values, EDGES).tolist() == expected
    assert radius_bin_indices(torch.tensor(values), EDGES).tolist() == expected
    assert radius_labels(EDGES) == (
        "[0.6000,0.6154)",
        "[0.6154,0.6298)",
        ">=0.6298",
    )


def test_projected_components_decompose_the_forward_response():
    gamma = torch.tensor([[0.1, 0.0], [0.0, 0.2]], dtype=torch.float64)
    zero = torch.tensor([[0.02, 0.3], [0.4, -0.1]], dtype=torch.float64)
    sheared = torch.tensor([[0.07, 0.7], [0.6, 0.0]], dtype=torch.float64)
    values = projected_components(zero, sheared, gamma)
    assert values["zero"].tolist() == pytest.approx([0.2, -0.5])
    assert values["sheared"].tolist() == pytest.approx([0.7, 0.0])
    assert values["response"].tolist() == pytest.approx([0.5, 0.5])


def test_projected_components_support_draw_axis():
    gamma = torch.tensor([[0.1, 0.0]], dtype=torch.float64)
    zero = torch.tensor([[[0.0, 1.0], [0.1, 2.0]]], dtype=torch.float64)
    sheared = torch.tensor([[[0.1, 3.0], [0.3, 4.0]]], dtype=torch.float64)
    values = projected_components(zero, sheared, gamma)
    assert values["response"].numpy() == pytest.approx(np.array([[1.0, 2.0]]))


def filled_case(measured: float, model: float) -> CaseAccumulator:
    case = CaseAccumulator(3)
    for extraction, response in (
        ("measured_catalogue_radius", measured),
        ("measured_catalogue_anchor", measured),
        ("catalogue_radius_unconditional", model),
        ("generated_radius", model),
        ("generated_radius_anchor", model),
        ("generated_radius_anchor_permuted_shear", model),
    ):
        case.count[extraction][:] = 10
        case.total[extraction]["zero"][:] = 20
        case.total[extraction]["sheared"][:] = 20 + 10 * response
        case.total[extraction]["response"][:] = 10 * response
    return case


def test_case_summary_uses_paired_equal_weight_case_differences():
    rows = summarize_cases(
        {0: filled_case(0.4, 0.6), 1: filled_case(0.5, 0.6)},
        radius_labels(EDGES),
    )
    row = next(
        value
        for value in rows
        if value["radius_bin"] == "[0.6000,0.6154)"
        and value["extraction"] == "generated_radius"
    )
    assert row["R_self"] == pytest.approx(0.6)
    assert row["model_minus_measured"] == pytest.approx(0.15)
    assert row["model_minus_measured_case_sem"] == pytest.approx(0.05)


def test_postcut_summary_sums_bins_before_equal_weighting_cases():
    first = filled_case(0.4, 0.6)
    second = filled_case(0.5, 0.6)
    first.count["generated_radius"][:] = [10, 20, 30]
    first.total["generated_radius"]["response"][:] = [5, 10, 15]
    first.total["generated_radius"]["zero"][:] = 0
    first.total["generated_radius"]["sheared"][:] = [5, 10, 15]
    rows = summarize_postcut({0: first, 1: second})
    row = next(value for value in rows if value["extraction"] == "generated_radius")
    assert row["R_self"] == pytest.approx(0.55)
    assert row["model_minus_measured"] == pytest.approx(0.1)


def test_anchor_curves_use_anchor_matched_measured_reference():
    first = filled_case(0.4, 0.6)
    second = filled_case(0.5, 0.6)
    for case, measured_anchor in ((first, 0.2), (second, 0.3)):
        case.total["measured_catalogue_anchor"]["response"][:] = 10 * measured_anchor
        case.total["measured_catalogue_anchor"]["sheared"][:] = 20 + 10 * measured_anchor
    rows = summarize_postcut({0: first, 1: second})
    radius = next(value for value in rows if value["extraction"] == "generated_radius")
    anchor = next(
        value for value in rows if value["extraction"] == "generated_radius_anchor"
    )
    assert radius["model_minus_measured"] == pytest.approx(0.15)
    assert anchor["model_minus_measured"] == pytest.approx(0.35)
