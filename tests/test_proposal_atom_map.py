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
    common_metric_scale,
    common_metric_score,
    mixture_from_score,
    draw_classes,
    slot_accounting,
    score_terms,
    summarise_terms,
    exact_centre_node,
    gather_truth,
    observation_truth,
    panel_limits,
    priority_race,
    shard_offsets,
    stratified_race,
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


def _mixture(probabilities):
    return torch.as_tensor(np.asarray(probabilities, dtype=np.float64))


def test_priority_race_is_deterministic_and_returns_distinct_atoms():
    weights = np.full(64, 1.0 / 64.0)
    first, _ = priority_race(_mixture(weights), seed=8701, object_id=142230, n_select=16)
    again, _ = priority_race(_mixture(weights), seed=8701, object_id=142230, n_select=16)
    np.testing.assert_array_equal(first, again)
    assert first.size == 16
    assert np.unique(first).size == 16


def test_priority_race_depends_on_the_object_id():
    weights = np.full(4096, 1.0 / 4096.0)
    a, _ = priority_race(_mixture(weights), seed=8701, object_id=1, n_select=64)
    b, _ = priority_race(_mixture(weights), seed=8701, object_id=2, n_select=64)
    assert not np.array_equal(np.sort(a), np.sort(b))


def test_priority_race_reproduces_the_production_key_rule():
    """key = q/u on a generator seeded by (seed, object_id); keep the top n."""

    from sbsi.catalogue_sampling import _priority_row_seed

    weights = np.linspace(1.0, 2.0, 256)
    weights /= weights.sum()
    generator = torch.Generator()
    generator.manual_seed(_priority_row_seed(8701, 77))
    uniform = torch.rand(256, generator=generator, dtype=torch.float64).clamp_min(
        float(np.finfo(np.float64).tiny)
    )
    expected = torch.topk(_mixture(weights) / uniform, 33, sorted=True).indices[:32]
    np.testing.assert_array_equal(
        priority_race(_mixture(weights), seed=8701, object_id=77, n_select=32)[0],
        expected.numpy(),
    )


def test_priority_race_favours_the_heavy_atoms():
    weights = np.full(2048, 1.0e-6)
    weights[:32] = 1.0
    weights /= weights.sum()
    chosen, _ = priority_race(_mixture(weights), seed=8701, object_id=5, n_select=64)
    # Every one of the 32 heavy atoms should win a place among 64 draws.
    assert set(range(32)).issubset(set(chosen.tolist()))


def test_priority_race_rejects_more_draws_than_atoms():
    with pytest.raises(ValueError, match="more atoms than draws"):
        priority_race(_mixture(np.full(8, 0.125)), seed=1, object_id=1, n_select=8)


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


def test_race_on_a_collapsed_mixture_still_reaches_the_starved_atoms():
    """The defensive floor is what lets a starved heavy atom be drawn at all."""

    n_atoms = 200000
    weights = np.full(n_atoms, PRODUCTION_DELTA / n_atoms)
    weights[0] += 1.0 - PRODUCTION_DELTA
    chosen, _ = priority_race(
        _mixture(weights), seed=8701, object_id=142230, n_select=16384
    )
    assert 0 in chosen.tolist()
    # Without replacement, the race cannot spend more than one draw on the
    # dominant atom -- the rest necessarily land on floor atoms.
    assert chosen.size == 16384


def test_report_payload_is_json_serializable_without_nan():
    payload = dict(
        mass=float(0.3),
        proposal_probability=float(4.1666666666666667e-09),
        drawn=bool(True),
    )
    json.dumps(payload, allow_nan=False)


def test_slot_accounting_shows_a_concentrated_component_wasting_its_mass():
    """A mass spike caps at one slot; the same mass spread converts linearly."""
    delta, n = 0.1, 1000
    floor = delta / n
    spike = np.full(n, floor)
    spike[0] += 0.9
    threshold = 1.0e-3
    book = slot_accounting(spike, floor, threshold, delta)
    # The flat 10% converts linearly over the 999 atoms that do not saturate;
    # the spike's own flat share is inside its capped single slot, not extra.
    np.testing.assert_allclose(book["flat_expected_draws"],
                               (floor / threshold) * (n - 1))
    # The ranked 0.9 sits on one atom, so it buys one slot, not 0.9/tau = 900.
    np.testing.assert_allclose(book["ranked_draws_if_unconcentrated"], 900.0)
    np.testing.assert_allclose(book["ranked_expected_draws"], 1.0, atol=1e-9)
    assert book["n_saturated"] == 1
    # The decomposition is exact: the two parts add to the realised draws.
    np.testing.assert_allclose(
        book["flat_expected_draws"] + book["ranked_expected_draws"],
        book["expected_draws"], rtol=1e-12,
    )


def test_slot_accounting_wastes_nothing_when_no_atom_saturates():
    delta, n = 0.1, 1000
    floor = delta / n
    flat = np.full(n, 1.0 / n)
    book = slot_accounting(flat, floor, 1.0e-2, delta)
    # No atom reaches tau, so realised draws equal total mass over tau.
    np.testing.assert_allclose(book["expected_draws"], 1.0 / 1.0e-2)
    assert book["n_saturated"] == 0


def test_slot_accounting_rejects_a_non_positive_threshold():
    with pytest.raises(ValueError, match="threshold must be positive"):
        slot_accounting(np.array([0.5]), 0.1, 0.0, 0.1)


def _stratified(probability, excluded, floor, n_ranked, n_uniform, seed=8701):
    return stratified_race(
        np.asarray(probability, dtype=np.float64),
        np.asarray(excluded, dtype=np.int64),
        floor, seed=seed, object_id=142230,
        n_ranked=n_ranked, n_uniform=n_uniform,
    )


def _tailed(n, floor, seed=7):
    """A mixture where every atom keeps some softmax preference above floor."""

    rng = np.random.default_rng(seed)
    return floor + rng.exponential(1e-4, size=n)


def test_stratified_race_spends_exactly_the_slots_it_is_given():
    """The point of the design: the counts are fixed, not bought with mass."""

    n, floor = 4096, 0.1 / 4096
    out = _stratified(_tailed(n, floor), np.arange(8), floor, 512, 512)
    assert out["ranked_indices"].size == 512
    assert out["uniform_indices"].size == 512
    assert out["drawn"].size == 1024 - out["n_overlap"]
    assert out["n_ranked_slots"] == 512 and out["n_uniform_slots"] == 512


def test_stratified_race_never_draws_an_atom_the_exact_stratum_holds():
    n, floor = 2048, 0.1 / 2048
    excluded = np.arange(64)
    out = _stratified(_tailed(n, floor), excluded, floor, 128, 128)
    assert not np.isin(out["drawn"], excluded).any()
    assert out["inclusion"][excluded].max() == 0.0
    assert out["n_eligible"] == n - 64


def test_stratified_race_gives_every_eligible_atom_the_uniform_inclusion():
    """Equal weights make a priority race a simple random sample."""

    n, floor = 1024, 0.1 / 1024
    out = _stratified(np.full(n, floor), np.arange(4), floor, 32, 256)
    eligible = np.ones(n, dtype=bool)
    eligible[:4] = False
    # No atom carries any preference, so the ranked stratum has nothing to
    # race and every slot ends up uniform.
    assert out["n_ranked_slots"] == 0
    assert out["n_uniform_slots"] == 32 + 256
    assert out["uniform_inclusion"] == pytest.approx((32 + 256) / (n - 4))
    np.testing.assert_allclose(out["inclusion"][eligible], out["uniform_inclusion"])


def test_stratified_race_hands_unspendable_ranked_slots_to_the_uniform_side():
    """A budget must never shrink because the ranking ran out of atoms."""

    n, floor = 2048, 0.1 / 2048
    q = np.full(n, floor)
    q[:100] += 1e-3
    out = _stratified(q, np.arange(10), floor, 512, 512)
    # 90 preferred atoms survive the exact stratum; all are taken outright.
    assert out["n_ranked_slots"] == 90
    assert out["n_ranked_slots_requested"] == 512
    assert out["n_uniform_slots"] == 512 + (512 - 90)
    np.testing.assert_allclose(out["ranked_inclusion"][10:100], 1.0)


def test_stratified_race_reaches_deeper_into_the_tail_than_a_mixed_race():
    """More ranked slots must lower the ranked threshold, not raise it."""

    n, floor = 8192, 0.1 / 8192
    q = _tailed(n, floor)
    narrow = _stratified(q, np.arange(16), floor, 256, 256)
    wide = _stratified(q, np.arange(16), floor, 2048, 256)
    assert wide["ranked_threshold"] < narrow["ranked_threshold"]
    assert (wide["ranked_inclusion"] >= narrow["ranked_inclusion"] - 1e-12).all()


def test_stratified_race_credits_a_double_winner_to_the_ranking():
    n, floor = 512, 0.1 / 512
    out = _stratified(_tailed(n, floor), np.array([], dtype=np.int64),
                      floor, 400, 400)
    assert out["n_overlap"] > 0
    ranked_members = np.isin(out["drawn"], out["ranked_indices"])
    np.testing.assert_array_equal(out["from_ranked"], ranked_members)


def test_stratified_race_combines_the_two_inclusions_independently():
    n, floor = 2048, 0.1 / 2048
    out = _stratified(_tailed(n, floor), np.arange(4), floor, 256, 256)
    expected = 1.0 - (1.0 - out["ranked_inclusion"][10]) * (1.0 - out["uniform_inclusion"])
    assert out["inclusion"][10] == pytest.approx(expected)
    assert (out["inclusion"] <= 1.0).all()


def test_stratified_race_rejects_a_budget_larger_than_the_eligible_set():
    n, floor = 256, 0.1 / 256
    with pytest.raises(ValueError, match="more eligible atoms"):
        _stratified(_tailed(n, floor), np.arange(200), floor, 128, 8)


def test_stratified_race_rejects_a_non_positive_floor():
    with pytest.raises(ValueError, match="defensive floor"):
        _stratified(np.full(256, 1.0 / 256), np.arange(4), 0.0, 16, 16)


def _catalogue(n=64, seed=11):
    """A small catalogue whose atoms span five decades in predicted flux."""

    rng = np.random.default_rng(seed)
    flux = 10.0 ** rng.uniform(0.0, 5.0, size=n)
    values = np.column_stack([
        rng.normal(0.0, 0.2, size=n),
        rng.normal(0.0, 0.2, size=n),
        rng.uniform(1.0, 8.0, size=n),
        flux,
    ])
    dispersion = np.column_stack([
        rng.uniform(0.05, 0.5, size=n),
        rng.uniform(0.05, 0.5, size=n),
        rng.uniform(0.5, 5.0, size=n),
        0.3 * np.log(10.0) * flux,
    ])
    detection = rng.uniform(0.2, 1.0, size=n)
    return values, dispersion, detection


def test_common_metric_scale_is_positive_and_per_coordinate():
    values, dispersion, _ = _catalogue()
    scale = common_metric_scale(values, dispersion)
    assert scale.shape == (4,)
    assert (scale > 0.0).all()
    # The three linear coordinates are the catalogue's median scatter.
    assert scale[:3] == pytest.approx(np.median(dispersion[:, :3], axis=0))
    # Flux is a fractional scatter in dex, so a catalogue built with a
    # constant 0.3 dex spread reports 0.3 however bright its atoms are.
    assert scale[3] == pytest.approx(0.3)


def test_common_metric_scale_ignores_the_observation_entirely():
    values, dispersion, _ = _catalogue()
    first = common_metric_scale(values, dispersion)
    second = common_metric_scale(values, dispersion)
    assert first == pytest.approx(second)
    assert common_metric_scale.__code__.co_argcount == 2


def test_common_metric_scale_rejects_a_non_positive_dispersion():
    values, dispersion, _ = _catalogue()
    dispersion[3, 1] = 0.0
    with pytest.raises(ValueError, match="dispersion must be positive"):
        common_metric_scale(values, dispersion)


def test_common_metric_score_prefers_the_closer_atom_not_the_vaguer_one():
    """The defect the shared metric exists to remove.

    Two atoms and one observation.  The first is closer in *every* coordinate
    but has narrow predicted errors, so it sits four of its own sigma away.
    The second is further away everywhere with errors wide enough that it
    never leaves one sigma.  Production rates the honest atom 22 nats worse
    for being closer, because the dispersion term cannot pay back what the
    quadratic term charges it.  The shared metric must reverse that.
    """

    observed = np.array([0.0, 0.0, 4.0, 1.2e5])
    values = np.array([
        [0.20, 0.20, 6.0, 4.0e4],
        [0.30, 0.30, 7.0, 2.0e4],
    ])
    dispersion = np.array([
        [0.05, 0.05, 0.5, 2.0e4],
        [0.50, 0.50, 5.0, 2.0e5],
    ])
    detection = np.array([1.0, 1.0])

    terms = score_terms(values, dispersion, detection, observed, np.arange(2))
    # The first atom really is nearer the observation in all four coordinates.
    assert (np.abs(terms["residual"][0]) < np.abs(terms["residual"][1])).all()
    # And production prefers the other one anyway.
    assert terms["score"][1] > terms["score"][0] + 20.0

    scale = np.array([0.2, 0.2, 2.0, 0.3])
    shared = common_metric_score(values, dispersion, detection, observed, scale)
    assert shared[0] > shared[1]


def test_common_metric_score_measures_flux_in_dex():
    """A factor-ten miss costs the same whatever the observation's brightness."""

    scale = np.array([0.2, 0.2, 2.0, 0.5])
    dispersion = np.ones((1, 4))
    detection = np.ones(1)
    penalties = []
    for bright in (1.0e2, 1.0e5):
        values = np.array([[0.0, 0.0, 0.0, bright / 10.0]])
        observed = np.array([0.0, 0.0, 0.0, bright])
        penalties.append(float(
            common_metric_score(values, dispersion, detection, observed, scale)[0]
        ))
    assert penalties[0] == pytest.approx(penalties[1])
    assert penalties[0] == pytest.approx(-0.5 * (1.0 / 0.5) ** 2)


def test_common_metric_score_sends_undetectable_atoms_to_minus_infinity():
    values, dispersion, detection = _catalogue(n=8)
    detection[2] = 0.0
    observed = np.array([0.0, 0.0, 4.0, 1.0e3])
    scale = common_metric_scale(values, dispersion)
    score = common_metric_score(values, dispersion, detection, observed, scale)
    assert score[2] == -np.inf
    assert np.isfinite(np.delete(score, 2)).all()


def test_common_metric_score_rejects_a_non_positive_observed_flux():
    values, dispersion, detection = _catalogue(n=8)
    scale = common_metric_scale(values, dispersion)
    with pytest.raises(ValueError, match="observed flux must be positive"):
        common_metric_score(values, dispersion, detection,
                            np.array([0.0, 0.0, 1.0, 0.0]), scale)


def test_mixture_from_score_matches_the_production_composition():
    """Same mixture as `WholeCatalogueProxy.mixture`, only the score differs."""

    values, dispersion, detection = _catalogue(n=32)
    observed = np.array([0.0, 0.0, 4.0, 1.0e3])
    proxy = build_proxy(values, dispersion, detection)
    reference = proxy.mixture(observed[None, :], delta=PRODUCTION_DELTA,
                              temperature=1.0)[0].numpy()
    terms = score_terms(values, dispersion, detection, observed, np.arange(32))
    rebuilt = mixture_from_score(terms["score"], delta=PRODUCTION_DELTA,
                                 temperature=1.0, n_atoms=32).numpy()
    assert rebuilt == pytest.approx(reference, rel=1e-10, abs=1e-15)


def test_mixture_from_score_keeps_the_defensive_floor_on_every_atom():
    values, dispersion, detection = _catalogue(n=32)
    score = np.full(32, -np.inf)
    score[7] = 0.0
    mixture = mixture_from_score(score, delta=PRODUCTION_DELTA,
                                 temperature=1.0, n_atoms=32).numpy()
    assert mixture.sum() == pytest.approx(1.0)
    assert (mixture >= PRODUCTION_DELTA / 32).all()
    assert mixture[7] == pytest.approx(1.0 - PRODUCTION_DELTA
                                       + PRODUCTION_DELTA / 32)


def test_mixture_from_score_rejects_a_non_positive_temperature():
    with pytest.raises(ValueError, match="temperature must be positive"):
        mixture_from_score(np.zeros(4), delta=PRODUCTION_DELTA,
                           temperature=0.0, n_atoms=4)


def test_common_metric_scale_survives_a_vanishing_predicted_flux():
    """Atoms with a denormal flux must not overflow the fractional scatter."""

    values, dispersion, _ = _catalogue(n=64)
    reference = common_metric_scale(values, dispersion)
    values[5, 3] = 1e-300
    dispersion[5, 3] = 1.0
    with np.errstate(over="raise", divide="raise"):
        scale = common_metric_scale(values, dispersion)
    assert np.isfinite(scale).all()
    # One atom out of 64 cannot move a median.
    assert scale == pytest.approx(reference, rel=1e-6)
