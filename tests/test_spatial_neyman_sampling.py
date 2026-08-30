import numpy as np
import pytest
from scipy.special import logsumexp

from sbsi.catalogue_sampling import (
    SPATIAL_CELL_COUNT,
    SPATIAL_PILOT_SEED,
    SPATIAL_PRODUCTION_SEED,
    TAIL_PROPOSAL_RECIPE,
    DefensiveLocalProposal,
    ProposalCandidates,
    ProposalCoordinateTable,
    SpatialCellPartition,
    _file_sha256,
    _spatial_rng,
    allocate_spatial_neyman_cycles,
    pack_spatial_production_prefix,
    reduce_spatial_mis,
    spatial_pilot_variance,
)


def _proposal(
    n_atoms=512,
    *,
    values=None,
    dispersion=None,
    detection=None,
):
    if values is None:
        values = np.zeros((n_atoms, 4), dtype=np.float64)
    if dispersion is None:
        parity = np.where(np.arange(n_atoms) % 2, 1.3, 0.7)
        dispersion = np.repeat(parity[:, None], 4, axis=1)
    if detection is None:
        detection = np.ones(n_atoms, dtype=np.float64)
    coordinates = ProposalCoordinateTable(
        values=np.asarray(values, dtype=np.float64),
        target_names=("x0", "x1", "x2", "x3"),
        center=np.zeros(4),
        scale=np.ones(4),
        dispersion=np.asarray(dispersion, dtype=np.float64),
        statistic="mean",
    )
    return DefensiveLocalProposal(
        coordinates,
        np.ones(n_atoms, dtype=np.float64),
        local_base_weights=np.asarray(detection, dtype=np.float64),
        build_coordinate_tree=False,
    )


def _candidates(rows):
    indices = np.asarray(rows, dtype=np.int64)
    return ProposalCandidates(
        indices=indices,
        distances=np.zeros(indices.shape, dtype=np.float64),
        radius=np.ones(len(indices), dtype=np.float64),
    )


@pytest.fixture(scope="module")
def sampled_uniform_proposal():
    proposal = _proposal()
    partition = proposal.spatial_cell_partition()
    observed = np.array([[0.2, -0.1, 0.3, 0.0]], dtype=np.float64)
    candidates = _candidates([[0]])
    pilot, production = proposal.draw_spatial_mis_reservoirs(
        candidates,
        observed,
        object_ids=np.array([41]),
        object_chunk=1,
    )
    return proposal, partition, observed, candidates, pilot, production


def test_partition_ties_binary_ids_hashes_cache_and_artifact(
    sampled_uniform_proposal, tmp_path
):
    proposal, partition, _, _, _, _ = sampled_uniform_proposal
    np.testing.assert_array_equal(partition.counts, 2)
    np.testing.assert_array_equal(
        partition.cell_id, np.arange(512, dtype=np.uint16) // 2
    )
    assert proposal.spatial_cell_partition() is partition
    assert partition.cell_id.dtype == np.uint16
    assert not partition.cell_id.flags.writeable
    assert len(partition.assignment_sha256) == 64
    assert len(partition.counts_sha256) == 64
    assert len(partition.detected_mass_sha256) == 64
    assert len(partition.prior_mass_sha256) == 64

    first = tmp_path / "first"
    second = tmp_path / "second"
    first_hashes = partition.save_artifact(first)
    second_hashes = partition.save_artifact(second)
    assert first_hashes == second_hashes
    assert (first / "partition.npz").read_bytes() == (
        second / "partition.npz"
    ).read_bytes()
    assert (first / "manifest.json").read_bytes() == partition.manifest_bytes
    assert first_hashes["partition_sha256"] == partition.partition_sha256
    assert first_hashes["partition_cache_sha256"] == _file_sha256(
        first / "partition.npz"
    )
    assert (
        first_hashes["partition_manifest_sha256"]
        == partition.partition_manifest_sha256
    )
    loaded = SpatialCellPartition.load_artifact(first, proposal=proposal)
    np.testing.assert_array_equal(loaded.cell_id, partition.cell_id)
    assert loaded.partition_sha256 == partition.partition_sha256

    changed_values = proposal.coordinates.values.copy()
    changed_values[0, 0] = 1.0
    changed = _proposal(values=changed_values)
    with pytest.raises(ValueError, match="coordinates_sha256"):
        SpatialCellPartition.load(first / "partition.npz", proposal=changed)

    (first / "manifest.json").write_bytes(partition.manifest_bytes + b" ")
    with pytest.raises(ValueError, match="manifest does not match"):
        SpatialCellPartition.load_artifact(first, proposal=proposal)


def test_partition_zero_detected_mass_nodes_use_deterministic_count_splits():
    n_atoms = 65_536
    detection = np.ones(n_atoms, dtype=np.float64)
    detection[: n_atoms // 2] = 0.0
    proposal = _proposal(n_atoms, detection=detection)
    partition = proposal.spatial_cell_partition()
    assert len(partition.cell_id) == n_atoms
    assert np.count_nonzero(partition.detected_mass == 0.0) > 0
    assert partition.counts.min() > 0
    assert partition.counts.max() <= 262_144
    np.testing.assert_array_equal(
        np.bincount(partition.cell_id, minlength=SPATIAL_CELL_COUNT),
        partition.counts,
    )
    repeated = _proposal(n_atoms, detection=detection).spatial_cell_partition()
    np.testing.assert_array_equal(repeated.cell_id, partition.cell_id)
    assert repeated.partition_sha256 == partition.partition_sha256


def test_cell_conditioned_dm3_matches_dense_q_and_excludes_exact_stratum(
    sampled_uniform_proposal,
):
    proposal, partition, observed, _, pilot, production = sampled_uniform_proposal
    proxy = proposal._tilted_proxy("cpu")
    proxy_logs = tuple(
        value.cpu().numpy()
        for value in proxy.component_log_scores(observed[0], TAIL_PROPOSAL_RECIPE)
    )
    for reservoir in (pilot, production):
        assert reservoir.partition_sha256 == partition.partition_sha256
        assert reservoir.atom_chunk == 262_144
        assert reservoir.object_chunk == 1
        assert not np.isin(reservoir.indices, [0]).any()
        for cell in range(SPATIAL_CELL_COUNT):
            positions = np.flatnonzero(partition.cell_id == cell)
            keep = proposal.active_indices[positions] != 0
            positions = positions[keep]
            prior = proposal.prior_weights[proposal.active_indices[positions]]
            prior = prior / prior.sum()
            components = []
            for log_score in proxy_logs:
                selected = log_score[positions]
                components.append(np.exp(selected - logsumexp(selected)))
            components.append(prior)
            q_h = np.einsum(
                "c,ca->a",
                np.asarray(TAIL_PROPOSAL_RECIPE.component_weights),
                np.stack(components),
            )
            slots = np.flatnonzero(reservoir.cell_id == cell)
            active_position = np.searchsorted(
                proposal.active_indices, reservoir.indices[0, slots]
            )
            local = np.searchsorted(positions, active_position)
            np.testing.assert_allclose(
                reservoir.probability[0, slots],
                q_h[local],
                rtol=2e-14,
                atol=1e-15,
            )


def test_rng_domains_normalizer_reuse_and_component_counts(
    sampled_uniform_proposal,
):
    _, _, _, _, pilot, production = sampled_uniform_proposal
    assert pilot.seed == SPATIAL_PILOT_SEED
    assert production.seed == SPATIAL_PRODUCTION_SEED
    assert pilot.domain == "pilot"
    assert production.domain == "production"
    assert pilot.record_sha256 != production.record_sha256
    np.testing.assert_array_equal(
        pilot.component_log_normalizer, production.component_log_normalizer
    )
    np.testing.assert_array_equal(pilot.cell_source_counts[0], [8, 6, 2])
    np.testing.assert_array_equal(production.cell_source_counts[0], [256, 192, 64])

    pilot_uniform = _spatial_rng(SPATIAL_PILOT_SEED, 41, 7, "pilot").random(16)
    repeated = _spatial_rng(SPATIAL_PILOT_SEED, 41, 7, "pilot").random(16)
    production_uniform = _spatial_rng(
        SPATIAL_PRODUCTION_SEED, 41, 7, "production"
    ).random(16)
    np.testing.assert_array_equal(pilot_uniform, repeated)
    assert not np.array_equal(pilot_uniform, production_uniform)


def test_sampler_rejects_equal_seeds_and_empty_post_exclusion_cell(
    sampled_uniform_proposal,
):
    proposal, partition, observed, _, _, _ = sampled_uniform_proposal
    with pytest.raises(ValueError, match="seeds must be distinct"):
        proposal.draw_spatial_mis_reservoirs(
            _candidates([[0]]),
            observed,
            pilot_seed=3,
            production_seed=3,
            partition=partition,
        )
    with pytest.raises(RuntimeError, match="cell 0 is empty"):
        proposal.draw_spatial_mis_reservoirs(
            _candidates([[0, 1]]),
            observed,
            partition=partition,
            object_ids=np.array([41]),
            object_chunk=1,
        )


def test_zero_proxy_normalizer_falls_back_exactly_to_cell_prior():
    n_atoms = 512
    with np.errstate(over="ignore", invalid="ignore"):
        proposal = _proposal(
            n_atoms,
            values=np.full((n_atoms, 4), 1.0e200),
            dispersion=np.ones((n_atoms, 4)),
        )
        empty = _candidates(np.empty((1, 0), dtype=np.int64))
        pilot, production = proposal.draw_spatial_mis_reservoirs(
            empty,
            np.zeros((1, 4)),
            object_ids=np.array([9]),
            object_chunk=1,
        )
    assert pilot.proxy_fallback.all()
    assert production.proxy_fallback.all()
    np.testing.assert_allclose(pilot.probability, 0.5, atol=1e-15)
    np.testing.assert_allclose(production.probability, 0.5, atol=1e-15)


def test_pilot_fixed_source_variance_uses_declared_alpha(
    sampled_uniform_proposal,
):
    _, _, _, _, pilot, _ = sampled_uniform_proposal
    values = np.empty_like(pilot.probability)
    per_cell = 16
    schedule = pilot.component_id[:per_cell]
    patterns = (
        np.arange(8, dtype=np.float64),
        2.0 * np.arange(6, dtype=np.float64),
        np.array([-1.0, 1.0]),
    )
    for cell in range(SPATIAL_CELL_COUNT):
        block = values[0, cell * per_cell : (cell + 1) * per_cell]
        for component, pattern in enumerate(patterns):
            block[schedule == component] = pattern
    variance = spatial_pilot_variance(values, pilot)
    expected = sum(
        weight * np.var(pattern, ddof=1)
        for weight, pattern in zip(
            TAIL_PROPOSAL_RECIPE.component_weights, patterns
        )
    )
    np.testing.assert_allclose(variance, expected, rtol=2e-15)


def test_exact_greedy_cycle_allocation_and_all_zero_fallback():
    all_zero = allocate_spatial_neyman_cycles(np.zeros((1, SPATIAL_CELL_COUNT)))
    np.testing.assert_array_equal(all_zero.cycles, 8)
    assert all_zero.all_zero_fallback[0]
    assert int(all_zero.total_cycles[0]) == 2048

    equal = allocate_spatial_neyman_cycles(np.ones((1, SPATIAL_CELL_COUNT)))
    np.testing.assert_array_equal(equal.cycles, 8)
    assert not equal.all_zero_fallback[0]

    sparse = np.zeros((1, SPATIAL_CELL_COUNT), dtype=np.float64)
    sparse[0, 255] = 1.0
    allocated = allocate_spatial_neyman_cycles(sparse)
    assert int(allocated.cycles[0, 255]) == 64
    assert int(allocated.cycles.min()) >= 2
    assert int(allocated.cycles.max()) <= 64
    assert int(allocated.cycles.sum()) == 2048
    assert not allocated.all_zero_fallback[0]


def test_whole_cycle_prefix_identity_exclusion_and_sum_of_cell_means(
    sampled_uniform_proposal,
):
    _, _, _, _, pilot, production = sampled_uniform_proposal
    # This toy target is q_h times the surviving prior mass in each cell, so
    # Y=c/q_h is constant within a cell.  Cell zero lost one of its two atoms.
    pilot_values = np.where(
        pilot.cell_id[None, :] == 0, 1.0 / 512.0, 1.0 / 256.0
    )
    allocation = allocate_spatial_neyman_cycles(
        spatial_pilot_variance(pilot_values, pilot)
    )
    assert allocation.all_zero_fallback[0]
    draw = pack_spatial_production_prefix(production, allocation)
    assert draw.indices.shape == (1, 16_384)
    for cell in range(SPATIAL_CELL_COUNT):
        packed = slice(cell * 64, (cell + 1) * 64)
        bank = slice(cell * 512, cell * 512 + 64)
        np.testing.assert_array_equal(
            draw.indices[0, packed], production.indices[0, bank]
        )
        np.testing.assert_array_equal(
            draw.component_id[0, packed], production.component_id[bank]
        )
    values = np.where(draw.cell_id == 0, 1.0 / 512.0, 1.0 / 256.0)
    reduced = reduce_spatial_mis(values, draw)
    np.testing.assert_allclose(reduced.estimate, 511.0 / 512.0, atol=1e-15)
    np.testing.assert_array_equal(reduced.standard_error, 0.0)


def test_reduction_reports_fixed_source_standard_error(
    sampled_uniform_proposal,
):
    _, _, _, _, _, production = sampled_uniform_proposal
    allocation = allocate_spatial_neyman_cycles(
        np.zeros((1, SPATIAL_CELL_COUNT))
    )
    draw = pack_spatial_production_prefix(production, allocation)
    values = np.empty_like(draw.probability)
    expected_cell_mean = 0.0
    expected_cell_variance = 0.0
    alpha = np.asarray(TAIL_PROPOSAL_RECIPE.component_weights)
    for cell in range(SPATIAL_CELL_COUNT):
        block = slice(cell * 64, (cell + 1) * 64)
        sources = draw.component_id[0, block]
        for component in range(len(alpha)):
            count = np.count_nonzero(sources == component)
            pattern = np.arange(count, dtype=np.float64) - (count - 1) / 2.0
            values[0, block][sources == component] = pattern
            if cell == 0:
                expected_cell_mean += alpha[component] * pattern.mean()
                expected_cell_variance += (
                    alpha[component] ** 2
                    * np.var(pattern, ddof=1)
                    / count
                )
    reduced = reduce_spatial_mis(values, draw)
    np.testing.assert_allclose(
        reduced.estimate,
        SPATIAL_CELL_COUNT * expected_cell_mean,
        atol=2e-13,
    )
    np.testing.assert_allclose(
        reduced.standard_error,
        np.sqrt(SPATIAL_CELL_COUNT * expected_cell_variance),
        rtol=2e-15,
    )
    np.testing.assert_allclose(
        reduced.cell_variance, expected_cell_variance, rtol=2e-15
    )
