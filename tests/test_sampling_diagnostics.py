import json

import numpy as np
from scipy.special import logsumexp

from sbsi.sampling_diagnostics import (
    ExactProposalTargetComparison,
    ImportanceSamplingDiagnostic,
    defensive_proposal_probabilities,
    normalized_importance_weights,
    optimize_defensive_epsilon,
    plot_exact_proposal_target,
    plot_importance_sampling_diagnostic,
    save_exact_proposal_target,
    save_importance_sampling_diagnostic,
    select_example_rows,
    simulate_defensive_evidence_errors,
    summarize_defensive_proposal,
    summarize_exact_proposal_target,
    summarize_importance_sampling,
)


def _diagnostic():
    log_weight = np.log(
        np.asarray(
            [
                [1, 1, 1, 1, 1, 1, 1, 1],
                [8, 1, 1, 1, 1, 1, 1, 1],
                [40, 1, 1, 1, 1, 1, 1, 1],
            ],
            dtype=np.float64,
        )
    )
    return ImportanceSamplingDiagnostic(
        object_ids=np.asarray([11, 22, 33]),
        center=(0.01, -0.02),
        conditional_log_likelihood=np.asarray(
            [
                [-4, -3, -2, -1, -4, -3, -2, -1],
                [-5, -4, -3, -2, -5, -4, -3, -2],
                [-6, -5, -4, -3, -6, -5, -4, -3],
            ],
            dtype=np.float64,
        ),
        log_importance_weight=log_weight,
        atom_indices=np.asarray(
            [
                [1, 2, 3, 4, 1, 2, 3, 4],
                [5, 6, 7, 8, 5, 6, 7, 8],
                [9, 10, 11, 12, 9, 10, 11, 12],
            ]
        ),
        proposal_probability=np.full((3, 8), 0.1),
        local_member=np.asarray(
            [
                [True, True, True, True, True, True, True, True],
                [False, True, True, True, False, True, True, True],
                [False, True, True, True, False, True, True, True],
            ]
        ),
        global_component=np.asarray(
            [
                [False, False, False, False, False, False, False, False],
                [True, False, False, False, True, False, False, False],
                [True, False, False, False, True, False, False, False],
            ]
        ),
        population_log_normalization=0.25,
        n_candidates=8,
        prefilter_candidates=16,
        epsilon=0.1,
        proposal_seed=8701,
    )


def test_sampling_summary_uses_nested_prefixes_and_ordinary_evidence():
    diagnostic = _diagnostic()
    weight = normalized_importance_weights(diagnostic.log_importance_weight[1, :4])
    np.testing.assert_allclose(weight.sum(), 1.0)
    rows = summarize_importance_sampling(diagnostic, (4, 8))
    entry = next(
        row for row in rows if row["object_id"] == 22 and row["n_draws"] == 4
    )
    expected = logsumexp(diagnostic.log_importance_weight[1, :4]) - np.log(4) - 0.25
    assert entry["log_evidence"] == expected
    assert entry["unique_atoms"] == 4
    assert entry["outside_local_evidence_fraction"] == weight[0]
    assert entry["global_draw_evidence_fraction"] == weight[0]


def test_sampling_examples_are_distinct_and_span_ess():
    rows, labels = select_example_rows(_diagnostic())
    assert len(rows) == len(set(rows)) == 3
    assert labels == ("typical ESS", "low ESS (p10)", "minimum ESS")
    assert rows[-1] == 2


def test_sampling_diagnostic_saves_arrays_summary_and_figures(tmp_path):
    diagnostic = _diagnostic().take([0, 2])
    labels = ("typical ESS", "minimum ESS")
    output = tmp_path / "diagnostic"
    summary = save_importance_sampling_diagnostic(
        diagnostic,
        output,
        ladder=(4, 8),
        labels=labels,
        pool_summary={"n_objects": 3},
        metadata={"reference": "test"},
    )
    payload = json.loads(summary.read_text())
    assert payload["object_ids"] == [11, 33]
    assert payload["pool_summary"]["n_objects"] == 3
    with np.load(output / "samples.npz") as arrays:
        assert arrays["log_importance_weight"].shape == (2, 8)
    paths = plot_importance_sampling_diagnostic(
        diagnostic,
        output,
        ladder=(4, 8),
        labels=labels,
        bins=8,
    )
    assert len(paths) == 4
    assert all(path.stat().st_size > 0 for path in paths)


def test_exact_proposal_target_reports_mismatch_and_saves_figure(tmp_path):
    target = np.asarray([0.40, 0.30, 0.20, 0.08, 0.02])
    proposal = np.asarray([0.10, 0.10, 0.10, 0.30, 0.40])
    comparison = ExactProposalTargetComparison(
        object_id=514716,
        center=(0.01, 0.0),
        atom_indices=np.arange(5),
        conditional_log_likelihood=np.asarray([-2.0, -1.0, 0.0, 1.0, 2.0]),
        log_target=np.log(target),
        proposal_probability=proposal,
        candidate_member=np.asarray([True, True, False, False, False]),
        candidate_indices=np.asarray([0, 1]),
        population_log_normalization=0.5,
        epsilon=0.1,
        n_candidates=2,
        prefilter_candidates=4,
    )
    summary = summarize_exact_proposal_target(comparison)
    np.testing.assert_allclose(summary["candidate_target_mass"], 0.7)
    np.testing.assert_allclose(summary["candidate_proposal_mass"], 0.2)
    np.testing.assert_allclose(
        summary["asymptotic_ess_fraction"],
        1.0 / np.sum(np.square(target) / proposal),
    )
    output = tmp_path / "exact"
    result = save_exact_proposal_target(
        comparison, output, metadata={"reference": "test"}
    )
    assert json.loads(result.read_text())["object_id"] == 514716
    paths = plot_exact_proposal_target(comparison, output, bins=8)
    assert all(path.stat().st_size > 0 for path in paths)


def test_defensive_proposal_sweep_metrics_and_paired_mc_are_exact():
    target = np.asarray([0.40, 0.30, 0.20, 0.10])
    prior = np.full(4, 0.25)
    candidates = np.asarray([0, 1])
    proposal, local, mass = defensive_proposal_probabilities(
        target, prior, candidates, 0.4
    )
    np.testing.assert_allclose(local, [4 / 7, 3 / 7])
    np.testing.assert_allclose(mass, 0.7)
    np.testing.assert_allclose(proposal.sum(), 1.0)
    summary = summarize_defensive_proposal(target, prior, candidates, 0.4)
    np.testing.assert_allclose(
        summary["asymptotic_ess_fraction"],
        1.0 / np.sum(np.square(target) / proposal),
    )
    epsilon, ess_fraction = optimize_defensive_epsilon(
        target, prior, candidates
    )
    assert 0 < epsilon <= 1
    assert ess_fraction >= summary["asymptotic_ess_fraction"]

    component = np.asarray([[0.1, 0.9, 0.1, 0.9], [0.9, 0.1, 0.9, 0.1]])
    global_positions = np.asarray([[0, 1, 2, 3], [3, 2, 1, 0]])
    local_uniform = np.asarray([[0.1, 0.9, 0.2, 0.8], [0.8, 0.2, 0.9, 0.1]])
    rows = simulate_defensive_evidence_errors(
        target,
        prior,
        candidates,
        0.4,
        component_uniform=component,
        global_positions=global_positions,
        local_uniform=local_uniform,
        ladder=(2, 4),
    )
    assert [row["n_draws"] for row in rows] == [2, 4]
    assert all(row["n_replicates"] == 2 for row in rows)
