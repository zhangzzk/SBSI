"""The bright/faint hybrid: a bright-only recompute plus a carried remainder.

cont.342 needs three evaluations of the score and information to converge.
cont.344 measured that recomputing only the 7.91% of objects brighter than
`mag_auto = 22` and carrying the rest by their own linear response reproduces
the converged answer to 0.14 sigma for 1.16 pass-equivalents of compute.

Two things have to hold for that to be a pipeline rather than a spreadsheet.
A bright-only pass evaluates a *non-contiguous* subset, so each object must
still draw exactly the atoms it would have drawn inside a full pass -- if the
subset perturbs the draw the comparison against the exact chain is no longer
paired.  And the carry itself must be applied once, from a genuinely evaluated
centre, not compounded pass over pass.
"""

import json
from hashlib import sha256
from pathlib import Path

import numpy as np
import pytest

from sbsi.catalogue_closure import generate_mock_catalogue
from sbsi.catalogue_null import estimate_one_step_adaptive_section5
from sbsi.selection_normalization import (
    ExactPopulationNormalization,
    load_population_normalization,
    log_mass_derivatives,
)

from _script_loader import load_script_module
from test_catalogue_null import _proposal, _torch_two_shape_likelihood

COMBINER = load_script_module("combine_hybrid_pass.py")


def _moments(mock, likelihood, **extra):
    return estimate_one_step_adaptive_section5(
        likelihood,
        mock,
        _proposal(likelihood),
        center=(0.0, 0.0),
        h=0.005,
        draw_ladder=(32, 64),
        n_candidates=4,
        epsilon=0.2,
        proposal_seed=806,
        min_ess=1e9,
        max_weight_fraction=1.0,
        object_chunk=8,
        atom_chunk=16,
        **extra,
    ).moments


def test_a_scattered_subset_draws_exactly_what_the_full_pass_drew():
    """The load-bearing property of `--object-subset`.

    Per-object seeds used to come from `offset + position in the chunk`, which
    is only the object's own identity when the partition is contiguous.  A
    bright-only pass is not, so the ids are now carried explicitly.
    """

    likelihood = _torch_two_shape_likelihood()
    mock = generate_mock_catalogue(
        likelihood,
        n_detected=24,
        g1=0.0,
        g2=0.0,
        scene_seed=803,
        detection_seed=804,
        flow_seed=805,
    )
    full = _moments(mock, likelihood)

    # Deliberately scattered, crossing the 8-object chunk boundaries, so a
    # position-derived seed could not accidentally agree.
    rows = np.array([1, 3, 9, 10, 17, 23], dtype=np.int64)
    subset = type(mock)(
        mock.measurements.iloc[rows].reset_index(drop=True),
        mock.truth.iloc[rows].reset_index(drop=True),
    )
    partial = _moments(subset, likelihood, object_ids=rows)

    np.testing.assert_array_equal(partial.draw_counts, full.draw_counts[rows])
    np.testing.assert_array_equal(partial.unique_counts, full.unique_counts[rows])
    np.testing.assert_allclose(partial.score, full.score[rows], rtol=1e-12, atol=0)
    np.testing.assert_allclose(
        partial.information, full.information[rows], rtol=1e-12, atol=0
    )


def test_a_contiguous_offset_and_explicit_ids_agree():
    """The new path must reproduce the old one exactly where they overlap."""

    likelihood = _torch_two_shape_likelihood()
    mock = generate_mock_catalogue(
        likelihood,
        n_detected=16,
        g1=0.0,
        g2=0.0,
        scene_seed=803,
        detection_seed=804,
        flow_seed=805,
    )
    by_offset = _moments(mock, likelihood, object_id_offset=40)
    by_ids = _moments(mock, likelihood, object_ids=40 + np.arange(16))
    np.testing.assert_array_equal(by_offset.score, by_ids.score)
    np.testing.assert_array_equal(by_offset.information, by_ids.information)


def test_ids_and_offset_together_are_refused():
    likelihood = _torch_two_shape_likelihood()
    mock = generate_mock_catalogue(
        likelihood,
        n_detected=8,
        g1=0.0,
        g2=0.0,
        scene_seed=803,
        detection_seed=804,
        flow_seed=805,
    )
    with pytest.raises(ValueError, match="exclusive"):
        _moments(mock, likelihood, object_id_offset=1, object_ids=np.arange(8))


def test_one_id_per_observation_is_required():
    likelihood = _torch_two_shape_likelihood()
    mock = generate_mock_catalogue(
        likelihood,
        n_detected=8,
        g1=0.0,
        g2=0.0,
        scene_seed=803,
        detection_seed=804,
        flow_seed=805,
    )
    with pytest.raises(ValueError, match="one id per observation"):
        _moments(mock, likelihood, object_ids=np.arange(5))


def _write_run(directory, center, rows, score, information, normalization=None):
    directory.mkdir(parents=True, exist_ok=True)
    selection = None
    if normalization is not None:
        normalization = Path(normalization)
        selection = {
            "normalization_cache": {
                "path": str(normalization.resolve()),
                "sha256": sha256(normalization.read_bytes()).hexdigest(),
            }
        }
    (directory / "result.json").write_text(
        json.dumps(
            {
                "result": {"center": list(map(float, center))},
                "pipeline_release": "v1.2-infer",
                "injected_shear": [0.02, 0.0],
                "selection": selection,
            }
        )
    )
    np.savez_compressed(
        directory / "one_step_moments.npz",
        score=np.asarray(score, dtype=np.float64),
        information=np.asarray(information, dtype=np.float64),
        object_rows=np.asarray(rows, dtype=np.int64),
    )
    return directory


def _write_normalization(path, center, log_mass, h=0.001):
    center = np.asarray(center, dtype=np.float64)
    points = {}
    for d1 in (-h, 0.0, h):
        for d2 in (-h, 0.0, h):
            point = center + (d1, d2)
            points[tuple(point)] = float(np.exp(log_mass(*point)))
    ExactPopulationNormalization(points, h, {}, {}).save(path)
    return path


def _quadratic_truth(n, seed=0):
    """A population whose log-likelihood is exactly quadratic in g.

    For such a population the linear carry is not an approximation, so the
    hybrid must return the same answer whatever fraction is recomputed.  That
    isolates the carry algebra from the question of whether real galaxies obey
    it, which is what the production run measures.
    """

    rng = np.random.default_rng(seed)
    information = np.zeros((n, 2, 2))
    root = rng.normal(size=(n, 2, 2))
    information[:] = np.einsum("nij,nkj->nik", root, root) + np.eye(2)
    peak = rng.normal(scale=0.02, size=(n, 2))
    return information, peak


def _score_at(information, peak, center):
    return np.einsum("nij,nj->ni", information, peak - np.asarray(center))


def test_the_carry_is_exact_on_an_exactly_quadratic_population(tmp_path):
    n = 60
    information, peak = _quadratic_truth(n)
    c0 = np.array([0.0, 0.0])
    c1 = np.array([0.013, -0.004])
    rows = np.arange(n)
    base = _write_run(
        tmp_path / "base", c0, rows, _score_at(information, peak, c0), information
    )
    bright = rows[:7]
    recomputed = _write_run(
        tmp_path / "bright",
        c1,
        bright,
        _score_at(information[bright], peak[bright], c1),
        information[bright],
    )
    out = tmp_path / "hybrid.json"
    COMBINER.main(
        ["--base", str(base), "--recomputed", str(recomputed), "--output", str(out)]
    )
    payload = json.loads(out.read_text())

    exact = c1 + np.linalg.solve(
        information.sum(0), _score_at(information, peak, c1).sum(0)
    )
    np.testing.assert_allclose(payload["hybrid"]["estimate"], exact, atol=1e-12)
    # And on this population the carry alone is already exact, so recomputing
    # changes nothing -- the recompute is pure cost, correctly reported.
    np.testing.assert_allclose(
        payload["hybrid_minus_carry_only"], [0.0, 0.0], atol=1e-12
    )
    assert payload["n_recomputed"] == 7
    assert payload["pass_equivalents"] == pytest.approx(1.0 + 7 / 60)


def test_fresh_population_normalization_is_applied_to_every_carried_row(tmp_path):
    n = 40
    rows = np.arange(n)
    bright = rows[:6]
    c0 = np.array([0.0, 0.0])
    c1 = np.array([0.02, -0.003])
    numerator_information = np.tile(np.diag([20.0, 15.0]), (n, 1, 1))
    peak = np.tile(np.array([0.012, 0.001]), (n, 1))

    def log_mass(g1, g2):
        return 10.0 * g1**3 - 0.7 * g1 * g2 + 0.4 * g2**2

    cache0 = _write_normalization(tmp_path / "norm0.json", c0, log_mass)
    cache1 = _write_normalization(tmp_path / "norm1.json", c1, log_mass)
    b0, h0 = log_mass_derivatives(load_population_normalization(cache0), c0)[1:]
    b1, h1 = log_mass_derivatives(load_population_normalization(cache1), c1)[1:]

    def total_moments(center, gradient, hessian, selected):
        numerator_score = _score_at(
            numerator_information[selected], peak[selected], center
        )
        return (
            numerator_score - gradient,
            numerator_information[selected] + hessian,
        )

    score0, information0 = total_moments(c0, b0, h0, rows)
    score1, information1 = total_moments(c1, b1, h1, bright)
    base = _write_run(
        tmp_path / "base", c0, rows, score0, information0, cache0
    )
    recomputed = _write_run(
        tmp_path / "bright", c1, bright, score1, information1, cache1
    )
    out = tmp_path / "hybrid.json"
    COMBINER.main(
        ["--base", str(base), "--recomputed", str(recomputed), "--output", str(out)]
    )
    payload = json.loads(out.read_text())
    expected_score, expected_information = total_moments(c1, b1, h1, rows)
    expected = c1 + np.linalg.solve(
        expected_information.sum(axis=0), expected_score.sum(axis=0)
    )
    np.testing.assert_allclose(payload["hybrid"]["estimate"], expected, atol=1e-11)
    assert payload["normalization_correction"]["n_carried"] == n - len(bright)
    assert np.linalg.norm(
        payload["normalization_correction"]["per_carried_row_information"]
    ) > 0


def test_bfgs_score_root_survives_nonpositive_observed_information(tmp_path):
    n = 30
    rows = np.arange(n)
    c0 = np.array([0.0, 0.0])
    c1 = np.array([0.025, -0.002])
    information = np.tile(np.diag([7.0, 11.0]), (n, 1, 1))
    peak = np.tile(np.array([0.018, 0.003]), (n, 1))
    base = _write_run(
        tmp_path / "base",
        c0,
        rows,
        _score_at(information, peak, c0),
        information,
    )
    recomputed = _write_run(
        tmp_path / "all",
        c1,
        rows,
        _score_at(information, peak, c1),
        -information,
    )
    out = tmp_path / "hybrid.json"
    COMBINER.main(
        [
            "--base",
            str(base),
            "--recomputed",
            str(recomputed),
            "--score-root-bfgs",
            "--output",
            str(out),
        ]
    )
    payload = json.loads(out.read_text())
    np.testing.assert_allclose(payload["hybrid"]["estimate"], peak[0], atol=1e-12)
    assert min(payload["hybrid"]["observed_information_eigenvalues"]) < 0
    assert min(payload["hybrid"]["solver_information_eigenvalues"]) > 0


def test_recomputing_everything_reduces_to_the_plain_solve(tmp_path):
    n = 30
    rng = np.random.default_rng(3)
    root = rng.normal(size=(n, 2, 2))
    information = np.einsum("nij,nkj->nik", root, root) + 4 * np.eye(2)
    rows = np.arange(n)
    base = _write_run(
        tmp_path / "base", [0.0, 0.0], rows, rng.normal(size=(n, 2)), information
    )
    # Scores at the new centre that are NOT the linear carry of the base ones,
    # so agreement can only come from the carry being fully overwritten.
    fresh = rng.normal(size=(n, 2))
    recomputed = _write_run(tmp_path / "all", [0.01, 0.0], rows, fresh, information)
    out = tmp_path / "hybrid.json"
    COMBINER.main(
        ["--base", str(base), "--recomputed", str(recomputed), "--output", str(out)]
    )
    payload = json.loads(out.read_text())
    expected = np.array([0.01, 0.0]) + np.linalg.solve(
        information.sum(0), fresh.sum(0)
    )
    np.testing.assert_allclose(payload["hybrid"]["estimate"], expected, atol=1e-12)


def test_partitions_are_reassembled_by_row_not_by_order(tmp_path):
    """Two GPUs return two files; the second may hold the lower rows."""

    n = 40
    information, peak = _quadratic_truth(n, seed=5)
    c0, c1 = np.array([0.0, 0.0]), np.array([0.02, 0.01])
    upper, lower = np.arange(20, 40), np.arange(0, 20)
    base = [
        _write_run(
            tmp_path / "b1", c0, upper, _score_at(information[upper], peak[upper], c0),
            information[upper],
        ),
        _write_run(
            tmp_path / "b0", c0, lower, _score_at(information[lower], peak[lower], c0),
            information[lower],
        ),
    ]
    bright = np.array([31, 2, 18, 37])
    recomputed = [
        _write_run(
            tmp_path / "r0", c1, bright[:2],
            _score_at(information[bright[:2]], peak[bright[:2]], c1),
            information[bright[:2]],
        ),
        _write_run(
            tmp_path / "r1", c1, bright[2:],
            _score_at(information[bright[2:]], peak[bright[2:]], c1),
            information[bright[2:]],
        ),
    ]
    out = tmp_path / "hybrid.json"
    COMBINER.main(
        ["--base", *map(str, base), "--recomputed", *map(str, recomputed),
         "--output", str(out)]
    )
    payload = json.loads(out.read_text())
    exact = c1 + np.linalg.solve(
        information.sum(0), _score_at(information, peak, c1).sum(0)
    )
    np.testing.assert_allclose(payload["hybrid"]["estimate"], exact, atol=1e-12)
    assert payload["n_base"] == 40


@pytest.mark.parametrize(
    "mutate, message",
    [
        ("foreign_row", "absent from the base pass"),
        ("overlap", "overlap"),
        ("centre", "different centres"),
        ("no_rows", "does not record which mock rows"),
    ],
)
def test_a_mismatched_assembly_is_refused(tmp_path, mutate, message):
    n = 12
    information, peak = _quadratic_truth(n, seed=9)
    c0, c1 = np.array([0.0, 0.0]), np.array([0.01, 0.0])
    rows = np.arange(n)
    base = _write_run(
        tmp_path / "base", c0, rows, _score_at(information, peak, c0), information
    )
    bright = np.array([1, 4])
    recomputed = _write_run(
        tmp_path / "bright", c1, bright,
        _score_at(information[bright], peak[bright], c1), information[bright],
    )
    argv = ["--base", str(base), "--recomputed", str(recomputed),
            "--output", str(tmp_path / "out.json")]

    if mutate == "foreign_row":
        np.savez_compressed(
            recomputed / "one_step_moments.npz",
            score=_score_at(information[bright], peak[bright], c1),
            information=information[bright],
            object_rows=np.array([1, 999]),
        )
    elif mutate == "overlap":
        second = _write_run(
            tmp_path / "base2", c0, rows[:4],
            _score_at(information[:4], peak[:4], c0), information[:4],
        )
        argv = ["--base", str(base), str(second),
                "--recomputed", str(recomputed),
                "--output", str(tmp_path / "out.json")]
    elif mutate == "centre":
        second = _write_run(
            tmp_path / "bright2", np.array([0.05, 0.0]), np.array([7]),
            _score_at(information[7:8], peak[7:8], c1), information[7:8],
        )
        argv = ["--base", str(base), "--recomputed", str(recomputed), str(second),
                "--output", str(tmp_path / "out.json")]
    elif mutate == "no_rows":
        np.savez_compressed(
            recomputed / "one_step_moments.npz",
            score=_score_at(information[bright], peak[bright], c1),
            information=information[bright],
        )

    with pytest.raises(RuntimeError, match=message):
        COMBINER.main(argv)


def test_a_pass_from_a_different_scene_is_refused(tmp_path):
    """The hybrid spans passes, so a stale directory is easy to pick up."""

    n = 10
    information, peak = _quadratic_truth(n, seed=11)
    c0, c1 = np.array([0.0, 0.0]), np.array([0.01, 0.0])
    rows = np.arange(n)
    base = _write_run(
        tmp_path / "base", c0, rows, _score_at(information, peak, c0), information
    )
    payload = json.loads((base / "result.json").read_text())
    payload["scene_sha256"] = {"scene": "a" * 64}
    (base / "result.json").write_text(json.dumps(payload))

    bright = np.array([2, 5])
    recomputed = _write_run(
        tmp_path / "bright", c1, bright,
        _score_at(information[bright], peak[bright], c1), information[bright],
    )
    payload = json.loads((recomputed / "result.json").read_text())
    payload["scene_sha256"] = {"scene": "b" * 64}
    (recomputed / "result.json").write_text(json.dumps(payload))

    with pytest.raises(RuntimeError, match="does not share the identity"):
        COMBINER.main(
            ["--base", str(base), "--recomputed", str(recomputed),
             "--output", str(tmp_path / "out.json")]
        )


def test_the_information_share_is_read_at_one_centre(tmp_path):
    """A share is only a share if numerator and denominator share a centre.

    The information grows as the centre approaches the peak, so dividing the
    recomputed information at the new centre by the total at the base centre
    produced values above one on the first production chain.  The share is now
    read entirely at the base centre and the growth is reported separately.
    """

    n = 50
    information, peak = _quadratic_truth(n, seed=13)
    c0, c1 = np.array([0.0, 0.0]), np.array([0.015, 0.0])
    rows = np.arange(n)
    base = _write_run(
        tmp_path / "base", c0, rows, _score_at(information, peak, c0), information
    )
    bright = rows[:10]
    # Recomputed information deliberately five times the base value, which is
    # what the cross-centre ratio used to report as the "share".
    recomputed = _write_run(
        tmp_path / "bright", c1, bright,
        _score_at(information[bright], peak[bright], c1), 5.0 * information[bright],
    )
    out = tmp_path / "hybrid.json"
    COMBINER.main(
        ["--base", str(base), "--recomputed", str(recomputed), "--output", str(out)]
    )
    payload = json.loads(out.read_text())
    expected = information[bright][:, 0, 0].sum() / information[:, 0, 0].sum()
    assert payload["recomputed_information_share"] == pytest.approx(expected)
    assert 0.0 < payload["recomputed_information_share"] <= 1.0
    assert payload["recomputed_information_growth"] == pytest.approx(5.0)
