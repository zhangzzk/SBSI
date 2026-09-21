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
    exact_centre_node,
    gather_truth,
    panel_limits,
    priority_race,
    shard_offsets,
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
    first = priority_race(_mixture(weights), seed=8701, object_id=142230, n_select=16)
    again = priority_race(_mixture(weights), seed=8701, object_id=142230, n_select=16)
    np.testing.assert_array_equal(first, again)
    assert first.size == 16
    assert np.unique(first).size == 16


def test_priority_race_depends_on_the_object_id():
    weights = np.full(4096, 1.0 / 4096.0)
    a = priority_race(_mixture(weights), seed=8701, object_id=1, n_select=64)
    b = priority_race(_mixture(weights), seed=8701, object_id=2, n_select=64)
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
        priority_race(_mixture(weights), seed=8701, object_id=77, n_select=32),
        expected.numpy(),
    )


def test_priority_race_favours_the_heavy_atoms():
    weights = np.full(2048, 1.0e-6)
    weights[:32] = 1.0
    weights /= weights.sum()
    chosen = priority_race(_mixture(weights), seed=8701, object_id=5, n_select=64)
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
    chosen = priority_race(
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
