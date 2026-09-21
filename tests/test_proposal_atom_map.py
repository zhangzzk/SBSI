"""Unit tests for the proposal atom-map plotting helpers.

The whole-catalogue figure is produced by the job itself against the 24m
cached tables; these cover the pure helpers on small synthetic inputs.
"""
import json

import numpy as np
import pytest
import torch

from scripts.plot_proposal_atom_map import (
    PRODUCTION_DELTA,
    TRUTH_COLUMNS,
    build_proxy,
    centred_measurements,
    draw_classes,
    exact_centre_node,
    gather_truth,
    observation_truth,
    panel_limits,
    production_draw,
    score_terms,
    shard_offsets,
    summarise_terms,
    zero_point_row,
)


def _manifest(tmp_path, counts):
    """Write a miniature prior subset with one flow_zero.parquet per shard."""

    import pyarrow as pa
    import pyarrow.parquet as pq

    shards, start = [], 0
    for index, count in enumerate(counts):
        root = tmp_path / f"shard_{index:02d}"
        root.mkdir()
        columns = {
            name: np.arange(start, start + count, dtype=np.float64) + offset / 100.0
            for offset, name in enumerate(TRUTH_COLUMNS)
        }
        pq.write_table(pa.table(columns), root / "flow_zero.parquet")
        shards.append({"root": str(root), "n_rows": int(count)})
        start += count
    return {"n_rows": int(start), "shards": shards}


def test_shard_offsets_are_cumulative_starts():
    manifest = {"n_rows": 9, "shards": [{"n_rows": 4}, {"n_rows": 2}, {"n_rows": 3}]}
    np.testing.assert_array_equal(shard_offsets(manifest), [0, 4, 6, 9])


def test_gather_truth_reads_across_shard_boundaries(tmp_path):
    manifest = _manifest(tmp_path, [4, 2, 3])
    atoms = np.array([0, 3, 4, 5, 8])
    values = gather_truth(manifest, atoms)
    assert values.shape == (5, len(TRUTH_COLUMNS))
    # First column was written as the global atom id, so it recovers the ids.
    np.testing.assert_allclose(values[:, 0], atoms)
    # Later columns carry a per-column offset, proving column order is kept.
    np.testing.assert_allclose(values[:, 1], atoms + 0.01)


def test_gather_truth_preserves_the_requested_order(tmp_path):
    manifest = _manifest(tmp_path, [4, 2, 3])
    atoms = np.array([8, 0, 5, 3])
    np.testing.assert_allclose(gather_truth(manifest, atoms)[:, 0], atoms)


def test_gather_truth_rejects_an_out_of_range_atom(tmp_path):
    manifest = _manifest(tmp_path, [4, 2])
    with pytest.raises(ValueError, match="outside the prior subset"):
        gather_truth(manifest, np.array([6]))


def test_gather_truth_rejects_an_inconsistent_manifest(tmp_path):
    manifest = _manifest(tmp_path, [4, 2])
    manifest["n_rows"] = 7
    with pytest.raises(ValueError, match="do not sum"):
        gather_truth(manifest, np.array([1]))


def _record(row, *, nested_row=None, nodes=9, k=4):
    exact = {
        "top_atoms": [[i + 10 * n for i in range(k)] for n in range(nodes)],
        "top_posterior_weights": [[0.4, 0.3, 0.2, 0.1] for _ in range(nodes)],
    }
    if nested_row is not None:
        exact["row"] = nested_row
    return {"row": row, "exact": exact}


def test_exact_centre_node_reads_the_first_stencil_node():
    heavy, mass = exact_centre_node(_record(409188), 409188)
    np.testing.assert_array_equal(heavy, [0, 1, 2, 3])
    np.testing.assert_allclose(mass, [0.4, 0.3, 0.2, 0.1])


def test_exact_centre_node_accepts_a_record_without_the_nested_row():
    """Only two of the 32 panel records repeat the row under 'exact'."""

    heavy, _ = exact_centre_node(_record(409188, nested_row=None), 409188)
    assert heavy.size == 4


def test_exact_centre_node_cross_checks_the_nested_row_when_present():
    heavy, _ = exact_centre_node(_record(142230, nested_row=142230), 142230)
    assert heavy.size == 4
    with pytest.raises(ValueError, match="disagrees with itself"):
        exact_centre_node(_record(142230, nested_row=99), 142230)


def test_exact_centre_node_rejects_the_wrong_row():
    with pytest.raises(ValueError, match="is for row"):
        exact_centre_node(_record(3563), 142230)


def test_exact_centre_node_rejects_repeated_atoms():
    record = _record(1)
    record["exact"]["top_atoms"][0] = [5, 5, 6, 7]
    with pytest.raises(ValueError, match="repeat"):
        exact_centre_node(record, 1)


def test_exact_centre_node_rejects_a_length_mismatch():
    record = _record(1)
    record["exact"]["top_posterior_weights"][0] = [0.5, 0.5]
    with pytest.raises(ValueError, match="disagree in length"):
        exact_centre_node(record, 1)


def _mock_manifest(tmp_path, case=114, index=3):
    """A miniature case input catalogue in the real column convention."""

    import pandas as pd

    rows = 8
    ratio = np.linspace(0.3, 0.95, rows)
    angle = np.linspace(5.0, 170.0, rows)
    ellipticity = (1.0 - ratio) / (1.0 + ratio)
    table = pd.DataFrame(
        {
            "index_input": np.arange(rows),
            "Re_input": np.linspace(0.2, 2.0, rows),
            "axis_ratio_input": ratio,
            "position_angle_input": angle,
            "r_input": np.linspace(17.0, 26.0, rows),
            "e1_input_rot0": ellipticity * np.cos(2 * np.deg2rad(angle)),
            "e2_input_rot0": ellipticity * np.sin(2 * np.deg2rad(angle)),
        }
    )
    path = tmp_path / "gals_info.feather"
    table.to_feather(path)
    manifest = {
        "per_case": [{"case": case, "sources": {"truth": {"path": str(path)}}}]
    }
    return manifest, table, {"source_case": case, "source_input_index": index}


def test_observation_truth_matches_the_atom_conventions(tmp_path):
    """e1/e2 and circularized Re must be built exactly as flow_zero builds them."""

    manifest, table, row = _mock_manifest(tmp_path, index=5)
    out = observation_truth(manifest, row)
    record = table.iloc[5]
    ratio = float(record["axis_ratio_input"])
    expected_e = (1.0 - ratio) / (1.0 + ratio)
    angle = np.deg2rad(float(record["position_angle_input"]))
    assert out["e1"] == pytest.approx(expected_e * np.cos(2 * angle))
    assert out["e2"] == pytest.approx(expected_e * np.sin(2 * angle))
    assert out["circularized_Re"] == pytest.approx(
        float(record["Re_input"]) * np.sqrt(ratio)
    )
    assert out["r"] == pytest.approx(float(record["r_input"]))
    json.dumps(out, allow_nan=False)


def test_observation_truth_rejects_an_unknown_case(tmp_path):
    manifest, _, row = _mock_manifest(tmp_path)
    row["source_case"] = 999
    with pytest.raises(ValueError, match="absent from the image-mock manifest"):
        observation_truth(manifest, row)


def test_observation_truth_rejects_an_out_of_range_index(tmp_path):
    manifest, _, row = _mock_manifest(tmp_path, index=99)
    with pytest.raises(ValueError, match="outside"):
        observation_truth(manifest, row)


def test_observation_truth_rejects_a_catalogue_not_indexed_by_input_index(tmp_path):
    import pandas as pd

    manifest, table, row = _mock_manifest(tmp_path)
    shuffled = table.iloc[::-1].reset_index(drop=True)
    path = tmp_path / "shuffled.feather"
    shuffled.to_feather(path)
    manifest["per_case"][0]["sources"]["truth"]["path"] = str(path)
    with pytest.raises(ValueError, match="not indexed by source_input_index"):
        observation_truth(manifest, row)


FLOOR = 4.1666666666666667e-09


def test_draw_classes_attribute_a_draw_to_the_larger_mixture_component():
    """q = 0.1*uniform + 0.9*softmax; the flat part contributes exactly floor."""
    # 3*FLOOR: ranking supplies 2*FLOOR against the flat FLOOR, so ranked.
    # 1.5*FLOOR: ranking supplies 0.5*FLOOR, less than the flat FLOOR.
    probability = np.array([0.7, FLOOR, 3.0 * FLOOR, 1.5 * FLOOR])
    ranked, flat = draw_classes(probability, np.array([0, 1, 2, 3]), FLOOR)
    np.testing.assert_array_equal(ranked, [True, False, True, False])
    np.testing.assert_array_equal(flat, [False, True, False, True])
    # The two classes partition the draw.
    assert (ranked ^ flat).all()


def test_draw_classes_do_not_call_a_near_floor_atom_ranked():
    """The old q>floor rule called this ranked; the ranking barely touched it."""
    probability = np.array([FLOOR * 1.0001])
    ranked, flat = draw_classes(probability, np.array([0]), FLOOR)
    assert not ranked.any()
    assert flat.all()


def test_draw_classes_put_a_floor_atom_in_the_uniform_class():
    probability = np.array([FLOOR * (1.0 + 1.0e-12), FLOOR])
    _, flat = draw_classes(probability, np.array([0, 1]), FLOOR)
    assert flat.all()


def test_draw_classes_rejects_a_non_positive_floor():
    with pytest.raises(ValueError, match="must be positive"):
        draw_classes(np.array([0.5]), np.array([0]), 0.0)


def test_score_terms_reproduce_the_documented_proxy_score():
    values = np.array([[0.0, 0.0], [1.0, 2.0]])
    dispersion = np.array([[0.5, 2.0], [1.0, 1.0]])
    detection = np.array([0.25, 1.0])
    observed = np.array([0.5, 1.0])
    terms = score_terms(values, dispersion, detection, observed,
                        np.array([0, 1]))
    expected = (
        np.log(detection)
        - np.log(dispersion).sum(axis=1)
        - 0.5 * (((observed[None, :] - values) / dispersion) ** 2).sum(axis=1)
    )
    np.testing.assert_allclose(terms["score"], expected, rtol=0, atol=1e-12)
    np.testing.assert_allclose(
        terms["score"],
        terms["log_detection"] + terms["log_dispersion"] + terms["quadratic"],
        rtol=0, atol=1e-12,
    )


def test_score_terms_reward_a_narrow_atom_that_fits_no_better():
    """A tighter sigma raises the density even at the same standardized miss."""
    values = np.array([[0.0], [0.0]])
    dispersion = np.array([[1.0], [0.01]])
    detection = np.array([1.0, 1.0])
    # Each atom is one sigma away, so the quadratic term is identical.
    narrow = score_terms(values, dispersion, detection, np.array([0.0]),
                         np.array([0, 1]))
    wide = score_terms(np.array([[1.0], [0.01]]), dispersion, detection,
                       np.array([0.0]), np.array([0, 1]))
    np.testing.assert_allclose(wide["quadratic"][0], wide["quadratic"][1])
    assert wide["score"][1] > wide["score"][0]
    assert narrow["log_dispersion"][1] > narrow["log_dispersion"][0]


def test_score_terms_rejects_a_non_positive_dispersion():
    with pytest.raises(ValueError, match="dispersion must be positive"):
        score_terms(np.array([[0.0]]), np.array([[0.0]]), np.array([1.0]),
                    np.array([0.0]), np.array([0]))


def test_summarise_terms_survives_an_undetected_atom():
    """log(Pdet)=-inf for an undetected atom must not poison the summary."""
    values = np.array([[0.0], [0.0]])
    dispersion = np.array([[1.0], [1.0]])
    detection = np.array([0.0, 1.0])
    terms = score_terms(values, dispersion, detection, np.array([0.0]),
                        np.array([0, 1]))
    summary = summarise_terms(terms, "mixed")
    assert summary["n"] == 2
    assert summary["n_finite"] == 1
    assert np.isfinite(summary["median_score"])


def test_zero_point_row_finds_the_zero_shear_point():
    """probability.npy is point-major; the proposal wants the zero-shear row."""

    manifest = {
        "n_rows": 12,
        "points": [[0.0, 0.0], [0.0139, 0.00028], [0.0149, 0.00028]],
    }
    assert zero_point_row(manifest, 12) == 0
    manifest["points"] = [[0.0139, 0.00028], [0.0, 0.0]]
    assert zero_point_row(manifest, 12) == 1


def test_zero_point_row_rejects_a_cache_without_zero_shear():
    with pytest.raises(ValueError, match="no zero-shear point"):
        zero_point_row({"n_rows": 4, "points": [[0.01, 0.0]]}, 4)


def test_zero_point_row_rejects_an_atom_count_mismatch():
    with pytest.raises(ValueError, match="disagree on atom count"):
        zero_point_row({"n_rows": 4, "points": [[0.0, 0.0]]}, 5)


def test_centred_measurements_differences_shape_and_logs_flux():
    observed = np.array([0.1, -0.2, 8.0, 1000.0])
    block = np.array([[0.3, 0.1, 9.0, 100.0], [0.1, -0.2, 8.0, 1000.0]])
    out = centred_measurements(block, observed)
    np.testing.assert_allclose(out[:, 0], [0.2, 0.0])
    np.testing.assert_allclose(out[:, 1], [0.3, 0.0])
    np.testing.assert_allclose(out[:, 2], [1.0, 0.0])
    # A tenth of the observed flux is exactly one decade below it.
    np.testing.assert_allclose(out[:, 3], [-1.0, 0.0])


def test_centred_measurements_rejects_wrong_width():
    with pytest.raises(ValueError, match="four measured coordinates"):
        centred_measurements(np.zeros((3, 2)), np.array([0.0, 0.0, 0.0, 1.0]))


def test_panel_limits_always_contain_the_mass_carrying_atoms():
    background = np.random.default_rng(0).normal(size=5000)
    primary = np.array([-40.0, 37.0])
    low, high = panel_limits(primary, background)
    assert low < primary.min() and high > primary.max()


def test_panel_limits_survive_a_degenerate_background():
    low, high = panel_limits(np.array([1.0]), np.zeros(10))
    assert low < high


def test_build_proxy_matches_an_explicit_gaussian_mixture():
    values = np.array([[0.0, 1.0, 3.0, 9.0], [2.0, -1.0, 4.0, 8.0]])
    dispersion = np.array([[1.0, 2.0, 0.5, 1.5], [0.5, 1.5, 2.0, 0.25]])
    detection = np.array([0.5, 1.0])
    observation = np.array([[0.3, 0.8, 3.5, 8.5]])

    mixture = build_proxy(values, dispersion, detection).mixture(
        observation, delta=PRODUCTION_DELTA
    ).numpy()[0]

    score = (
        np.log(detection)
        - np.log(dispersion).sum(axis=1)
        - 0.5 * np.square((observation[0] - values) / dispersion).sum(axis=1)
    )
    local = np.exp(score - score.max())
    local /= local.sum()
    expected = PRODUCTION_DELTA * np.full(2, 0.5) + (1.0 - PRODUCTION_DELTA) * local
    np.testing.assert_allclose(mixture, expected, rtol=1e-12, atol=0)




def test_report_payload_is_json_serializable_without_nan():
    payload = dict(
        mass=float(0.3),
        proposal_probability=float(4.1666666666666667e-09),
        drawn=bool(True),
    )
    json.dumps(payload, allow_nan=False)


def _mixture(n=1024, floor_share=PRODUCTION_DELTA, seed=3):
    """A defensive mixture: a flat floor everywhere plus a ranked tail."""

    rng = np.random.default_rng(seed)
    floor = floor_share / n
    ranked = rng.exponential(1.0, size=n)
    ranked /= ranked.sum() / (1.0 - floor_share)
    return floor + ranked, floor


def test_production_draw_is_deterministic_and_keyed_on_the_object():
    q, _ = _mixture()
    empty = np.array([], dtype=np.int64)
    first = production_draw(q, seed=8701, object_id=142230, n_draws=512,
                            exact_stratum=empty)
    again = production_draw(q, seed=8701, object_id=142230, n_draws=512,
                            exact_stratum=empty)
    other = production_draw(q, seed=8701, object_id=409188, n_draws=512,
                            exact_stratum=empty)
    np.testing.assert_array_equal(first["all_draws"], again["all_draws"])
    assert not np.array_equal(first["all_draws"], other["all_draws"])


def test_production_draw_nests_a_smaller_budget_inside_a_larger_one():
    """Fixed-width uniform records are what make the draw ladder free.

    ``draw_uniforms_batch`` takes a whole row of uniforms per draw and reads one
    column, so doubling the budget appends draws rather than re-rolling them.
    """

    q, _ = _mixture()
    empty = np.array([], dtype=np.int64)
    short = production_draw(q, seed=8701, object_id=7, n_draws=256,
                            exact_stratum=empty)
    long = production_draw(q, seed=8701, object_id=7, n_draws=1024,
                           exact_stratum=empty)
    np.testing.assert_array_equal(short["all_draws"], long["all_draws"][:256])


def test_production_draw_samples_with_replacement():
    """The estimator this reproduces draws with replacement, so a dominant atom
    is drawn repeatedly; priority sampling could only ever take it once."""

    n = 1024
    q = np.full(n, PRODUCTION_DELTA / n)
    q[0] += 1.0 - PRODUCTION_DELTA
    out = production_draw(q, seed=8701, object_id=1, n_draws=4096,
                          exact_stratum=np.array([], dtype=np.int64))
    hits = out["multiplicity"][out["drawn"] == 0]
    assert hits.size == 1
    # ~90% of 4096 draws land on atom 0; the exact count is the sampler's.
    assert hits[0] > 3000
    assert out["all_draws"].size == 4096


def test_production_draw_wastes_the_draws_that_land_in_the_exact_stratum():
    """A draw inside the exactly summed stratum carries no contribution, so it
    is spent but unusable -- it is not re-rolled and the atom is not returned."""

    q, _ = _mixture()
    stratum = np.argsort(q)[::-1][:64]
    out = production_draw(q, seed=8701, object_id=1, n_draws=4096,
                          exact_stratum=stratum)
    assert out["n_wasted"] + out["n_usable"] == 4096
    assert out["n_wasted"] > 0
    assert not np.isin(out["drawn"], stratum).any()
    assert int(out["multiplicity"].sum()) == out["n_usable"]
    np.testing.assert_allclose(out["exact_mass"], q[stratum].sum(), rtol=1e-12)


def test_production_draw_frequencies_follow_the_proposal():
    """Inverse-CDF sampling, so an atom is hit in proportion to its own q."""

    q, _ = _mixture(n=256, seed=5)
    out = production_draw(q, seed=8701, object_id=11, n_draws=200000,
                          exact_stratum=np.array([], dtype=np.int64))
    frequency = np.zeros(q.size)
    frequency[out["drawn"]] = out["multiplicity"] / 200000.0
    # Three sigma on a binomial share of 200k draws, summed over 256 atoms.
    tolerance = 3.0 * np.sqrt(q * (1.0 - q) / 200000.0)
    assert (np.abs(frequency - q) <= tolerance + 1e-12).mean() > 0.98


def test_production_draw_coverage_is_the_chance_of_being_seen_at_all():
    q, _ = _mixture(n=256, seed=5)
    out = production_draw(q, seed=8701, object_id=11, n_draws=1000,
                          exact_stratum=np.array([], dtype=np.int64))
    np.testing.assert_allclose(out["coverage"], 1.0 - (1.0 - q) ** 1000,
                               rtol=1e-9, atol=1e-12)


def test_production_draw_rejects_an_empty_budget_or_catalogue():
    q, _ = _mixture(n=16)
    empty = np.array([], dtype=np.int64)
    with pytest.raises(ValueError, match="n_draws must be positive"):
        production_draw(q, seed=1, object_id=1, n_draws=0, exact_stratum=empty)
    with pytest.raises(ValueError, match="one proposal probability per atom"):
        production_draw(np.empty(0), seed=1, object_id=1, n_draws=8,
                        exact_stratum=empty)

