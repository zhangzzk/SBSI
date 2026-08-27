import json

import numpy as np
from scipy.special import logsumexp

from sbsi.sampling_diagnostics import (
    ImportanceSamplingDiagnostic,
    normalized_importance_weights,
    plot_importance_sampling_diagnostic,
    save_importance_sampling_diagnostic,
    select_example_rows,
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
