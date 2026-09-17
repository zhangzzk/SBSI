"""Focused tests for the ConstGold flow/blend response split."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.diagnose_constgold_response_split import (
    bin_edges,
    fit_terms,
    group_matrix,
    group_responses,
    load_anchor_column,
    load_column,
    magnitude_names,
    stack,
    sufficient_or_empty,
    summarize,
)

H = 0.02


def stats(value, n):
    """Sufficient statistics for ``n`` objects all carrying the same shape."""
    return {"numerator": np.array([value * n, 0.0]), "denominator": float(n)}


def planted_case(flow, blend, measured, n=100):
    """One bin whose pooled response is exactly ``flow``/``blend``/``measured``."""
    entry = {
        "measured": {"plus": stats(measured * H, n), "minus": stats(-measured * H, n)},
        "flow": {"plus": stats(flow * H, n), "minus": stats(-flow * H, n)},
        "blend::a": {"plus": stats(blend * H, n), "minus": stats(-blend * H, n)},
    }
    return entry


def test_bin_edges_are_open_and_increasing():
    edges = bin_edges(np.linspace(-1.0, 3.0, 10_000), 4)
    assert edges[0] == -np.inf and edges[-1] == np.inf
    assert np.all(np.diff(edges) > 0)
    assert edges[2] == pytest.approx(1.0, abs=1e-3)


def test_bin_edges_rejects_a_degenerate_quantile():
    with pytest.raises(ValueError, match="strictly increasing"):
        bin_edges(np.zeros(1000), 4)
    with pytest.raises(ValueError, match="three bins"):
        bin_edges(np.linspace(0, 1, 100), 2)


TERMS = ["measured", "flow", "blend::a"]


def responses_of(per_case, cases, groups, cells):
    numerator, denominator = stack(per_case, cases, cells, TERMS)
    return group_responses(numerator, denominator,
                           group_matrix(groups, cells), H)


def test_group_responses_recover_the_planted_responses_over_cases():
    per_case = {"0": {"b": planted_case(0.48, 0.20, 0.67, n=10)},
                "1": {"b": planted_case(0.48, 0.20, 0.67, n=30)}}
    values, objects = responses_of(per_case, (0, 1), {"b": ["b"]}, ("b",))
    assert values[0, 1] == pytest.approx(0.48)
    assert values[0, 2] == pytest.approx(0.20)
    assert values[0, 0] == pytest.approx(0.67)
    assert objects[0] == pytest.approx(40.0)


def test_a_group_pools_its_cells_by_object_count_not_by_averaging_them():
    #  two cells with the same response but very different occupancy: the
    #  marginal must be that response, and a cell-count average would agree,
    #  so make the responses differ and check the occupancy weighting.
    per_case = {"0": {"lo": planted_case(0.40, 0.0, 0.40, n=900),
                      "hi": planted_case(0.80, 0.0, 0.80, n=100)}}
    cells = ("lo", "hi")
    values, objects = responses_of(per_case, (0,), {"both": list(cells)}, cells)
    assert objects[0] == pytest.approx(1000.0)
    assert values[0, 1] == pytest.approx(0.9 * 0.40 + 0.1 * 0.80)


def test_sufficient_or_empty_reports_a_cell_no_case_populated():
    entry = sufficient_or_empty(np.zeros((0, 2)), np.zeros(0))
    assert entry["denominator"] == 0.0
    assert list(entry["numerator"]) == [0.0, 0.0]


def test_group_responses_refuse_an_empty_group_but_tolerate_a_resample():
    per_case = {"0": {"b": {term: {side: sufficient_or_empty(np.zeros((0, 2)),
                                                             np.zeros(0))
                                   for side in ("plus", "minus")}
                            for term in TERMS}}}
    numerator, denominator = stack(per_case, (0,), ("b",), TERMS)
    matrix = group_matrix({"b": ["b"]}, ("b",))
    with pytest.raises(RuntimeError, match="empty"):
        group_responses(numerator, denominator, matrix, H)
    values, _ = group_responses(numerator, denominator, matrix, H, strict=False)
    assert np.isnan(values).all()


def test_magnitude_names_are_half_open_and_name_the_open_ends():
    assert magnitude_names((23.0, 24.0)) == ["r_p<23", "r_p[23,24)", "r_p>=24"]


def test_fit_terms_recovers_planted_fractional_errors():
    flow = np.full(6, 0.48)
    blend = np.linspace(0.0, 1.2, 6)
    residual = 0.03 * flow - 0.05 * blend
    fit = fit_terms(flow, blend, residual, np.ones(6))
    assert fit["flow_fractional_error"] == pytest.approx(0.03)
    assert fit["blend_fractional_error"] == pytest.approx(-0.05)


def test_fit_terms_separates_a_pure_blend_error_from_a_pure_flow_error():
    flow = np.full(6, 0.48)
    blend = np.linspace(0.0, 1.2, 6)
    blend_only = fit_terms(flow, blend, -0.05 * blend, np.ones(6))
    assert blend_only["blend_fractional_error"] == pytest.approx(-0.05)
    assert blend_only["flow_fractional_error"] == pytest.approx(0.0, abs=1e-12)
    flow_only = fit_terms(flow, blend, 0.02 * flow, np.ones(6))
    assert flow_only["flow_fractional_error"] == pytest.approx(0.02)
    assert flow_only["blend_fractional_error"] == pytest.approx(0.0, abs=1e-12)


def test_fit_terms_weights_bins_by_their_occupancy():
    #  four bins agree on (flow 0, blend +0.1); a fifth contradicts them
    flow = np.full(5, 0.5)
    blend = np.array([0.0, 0.3, 0.6, 1.0, 0.8])
    residual = 0.1 * blend
    residual[4] = -0.5
    populous = fit_terms(flow, blend, residual,
                         np.array([1e6, 1e6, 1e6, 1e6, 1.0]))
    assert populous["blend_fractional_error"] == pytest.approx(0.1, abs=1e-4)
    assert populous["flow_fractional_error"] == pytest.approx(0.0, abs=1e-4)
    #  with the outlier weighted equally it drags the fit far away
    equal = fit_terms(flow, blend, residual, np.ones(5))
    assert abs(equal["blend_fractional_error"] - 0.1) > 0.1


def test_summarize_reports_a_consistent_m_and_recovers_the_planted_errors():
    #  measured = 1.02 * flow + 0.95 * blend, so the fit must find (+0.02, -0.05)
    keys = ("bin00", "bin01", "bin02")
    blends = {"bin00": 0.0, "bin01": 0.4, "bin02": 1.0}
    per_case = {}
    for case in range(4):
        entry = {}
        total_flow = total_blend = total_measured = 0.0
        for key in keys:
            flow, blend = 0.48, blends[key]
            measured = 1.02 * flow + 0.95 * blend
            entry[key] = planted_case(flow, blend, measured)
            total_flow += flow
            total_blend += blend
            total_measured += measured
        entry["all"] = planted_case(total_flow / 3, total_blend / 3,
                                    total_measured / 3, n=300)
        per_case[str(case)] = entry
    groups = {key: [key] for key in keys}
    groups["all"] = list(keys)
    out = summarize(per_case, range(4), keys, groups, keys, ["a"],
                    h=H, replicates=50, seed=3)
    fit = out["fits"]["a"]
    assert fit["flow_fractional_error"]["value"] == pytest.approx(0.02)
    assert fit["blend_fractional_error"]["value"] == pytest.approx(-0.05)
    # identical cases mean a case bootstrap has nothing to resample
    assert fit["blend_fractional_error"]["standard_error"] == pytest.approx(0.0, abs=1e-12)
    # the zero-blend bin is a clean read of the flow error alone
    assert out["bins"]["bin00"]["m_percent::a"]["value"] == pytest.approx(2.0)
    assert out["all"]["R_flow"] == pytest.approx(0.48)


def test_crossed_cells_marginalise_back_to_the_blending_bins():
    #  The point of the magnitude axis is that a blending bin is not a random
    #  slice of the cohort.  Plant a flow error that depends ONLY on primary
    #  magnitude, and make the quiet bin 90% bright while the crowded bin is
    #  90% faint.  Each blending bin then reads a different flow error even
    #  though no flow error depends on blending, and the magnitude marginals
    #  recover the two planted numbers.
    cells, groups = [], {}
    per_case = {}
    plan = {"bin00": {"r_p<24": 900, "r_p>=24": 100},
            "bin01": {"r_p<24": 100, "r_p>=24": 900}}
    error = {"r_p<24": 1.10, "r_p>=24": 1.00}
    for key, bands in plan.items():
        groups[key] = []
        for band, count in bands.items():
            cell = f"{key}|{band}"
            cells.append(cell)
            groups[key].append(cell)
            groups.setdefault(band, []).append(cell)
    groups["all"] = list(cells)
    for case in range(4):
        entry = {}
        for key, bands in plan.items():
            for band, count in bands.items():
                flow = 0.5
                entry[f"{key}|{band}"] = planted_case(
                    flow, 0.0, error[band] * flow, n=count)
        per_case[str(case)] = entry
    out = summarize(per_case, range(4), tuple(cells), groups,
                    tuple(plan), ["a"], h=H, replicates=20, seed=11)
    m = {name: out["groups"][name]["m_percent::a"]["value"] for name in out["groups"]}
    #  the magnitude bands read their own planted errors, cleanly
    assert m["r_p<24"] == pytest.approx(10.0)
    assert m["r_p>=24"] == pytest.approx(0.0, abs=1e-9)
    #  the blending bins read a mixture set by their magnitude composition
    assert m["bin00"] == pytest.approx(9.0)
    assert m["bin01"] == pytest.approx(1.0)
    #  and the cohort reads the cohort mixture, half and half
    assert m["all"] == pytest.approx(5.0)
    assert out["groups"]["bin00"]["objects"] == pytest.approx(4000.0)


def test_load_column_aligns_any_lookup_column_to_the_anchor_order():
    #  the anchor order is the contract: the lookup is keyed by input_index and
    #  may arrive in any order, so alignment must follow the anchor, not the file
    frame = pd.DataFrame({
        "case": [7, 7, 7, 8],
        "input_index": [30, 10, 20, 10],
        "R_blend": [-0.3, 0.1, 0.2, 99.0],
        "R_blend_abs": [0.5, 0.1, 0.4, 99.0],
    })
    anchor = np.array([10, 20, 30])
    signed = load_column(frame, Path("lookup.feather"), 7, anchor, "R_blend")
    absolute = load_column(frame, Path("lookup.feather"), 7, anchor, "R_blend_abs")
    assert list(signed) == pytest.approx([0.1, 0.2, -0.3])
    assert list(absolute) == pytest.approx([0.1, 0.4, 0.5])
    #  the third primary is the point of the column: its signed sum is large and
    #  negative while its absolute sum shows the blending is genuinely there
    assert absolute[2] > abs(signed[2])


def test_load_column_refuses_an_anchor_key_the_lookup_does_not_cover():
    frame = pd.DataFrame({
        "case": [7], "input_index": [10], "R_blend": [0.1], "R_blend_abs": [0.1],
    })
    with pytest.raises(RuntimeError, match="misses 1 anchor keys"):
        load_column(frame, Path("lookup.feather"), 7, np.array([10, 11]), "R_blend_abs")


def test_load_anchor_column_sorts_keys_and_converts_radius_to_arcsec(tmp_path):
    path = tmp_path / "anchor.npz"
    target = np.zeros((3, 4), dtype=float)
    target[:, 2] = [4.0, 3.1, 5.0]
    np.savez(path, input_index=np.array([30, 10, 20]), target=target)
    expected_ids = np.array([10, 20, 30])
    values = load_anchor_column(path, "g0_flux_radius_arcsec", expected_ids)
    np.testing.assert_allclose(values, [0.62, 1.0, 0.8])


def test_load_anchor_column_refuses_identity_mismatch(tmp_path):
    path = tmp_path / "anchor.npz"
    target = np.ones((2, 4), dtype=float)
    np.savez(path, input_index=np.array([1, 2]), target=target)
    with pytest.raises(RuntimeError, match="identities differ"):
        load_anchor_column(
            path,
            "g0_flux_radius_arcsec",
            np.array([1, 3]),
        )
