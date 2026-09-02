#!/usr/bin/env python
"""Combine a bright-only recompute with analytically carried faint moments.

The Newton chain of cont.342 needs three evaluations of the score and the
information to converge, and a full evaluation costs the same every time.  But
the two halves of the catalogue behave differently between passes.  A bright
object's score is genuinely re-evaluated: it carries most of the information
and most of the pass-to-pass change.  A faint object's score moves almost
exactly as its own quadratic expansion says it should,

    s(g) = s(g0) - I(g0) (g - g0),      I(g) = I(g0),

which is free.  So a pass after the first evaluates only the bright objects and
carries the rest.

The carry is always taken from the **first** pass, never from the previous
carried pass, so the linear extrapolation is applied once from a genuinely
evaluated point rather than compounded.  ``--base`` therefore stays pointed at
pass 0 for every later pass.

This is an approximation and the only one in the chain.  It is quantified by
`--reference`: the same solve with *every* object carried, which is what the
hybrid would return if the recompute did nothing.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from sbsi.selection_normalization import (
    load_population_normalization,
    log_mass_derivatives,
)


def _sha256(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _summarize(
    center,
    score,
    information,
    *,
    solver_information=None,
    solver_method="observed_newton",
):
    center = np.asarray(center, dtype=np.float64)
    score_sum = score.sum(axis=0, dtype=np.float64)
    observed_information = information.sum(axis=0, dtype=np.float64)
    observed_information = 0.5 * (observed_information + observed_information.T)
    observed_eigenvalues = np.linalg.eigvalsh(observed_information)
    information_sum = (
        observed_information
        if solver_information is None
        else np.asarray(solver_information, dtype=np.float64)
    )
    information_sum = 0.5 * (information_sum + information_sum.T)
    eigenvalues = np.linalg.eigvalsh(information_sum)
    if not np.isfinite(eigenvalues).all() or eigenvalues[0] <= 0:
        raise RuntimeError(f"non-positive combined information: {eigenvalues.tolist()}")
    step = np.linalg.solve(information_sum, score_sum)
    residual = score - np.einsum("nij,j->ni", information, step)
    influence = np.linalg.solve(information_sum / len(score), residual.T).T
    robust_covariance = np.cov(influence, rowvar=False, ddof=1) / len(score)
    return {
        "center": center.tolist(),
        "estimate": (center + step).tolist(),
        "step": step.tolist(),
        "score_sum": score_sum.tolist(),
        "observed_information_sum": observed_information.tolist(),
        "observed_information_eigenvalues": observed_eigenvalues.tolist(),
        "solver_method": solver_method,
        "solver_information_sum": information_sum.tolist(),
        "solver_information_eigenvalues": eigenvalues.tolist(),
        "robust_standard_error": np.sqrt(np.diag(robust_covariance)).tolist(),
        "model_standard_error": np.sqrt(
            np.diag(np.linalg.inv(information_sum))
        ).tolist(),
        "quadratic_log_likelihood_gain": 0.5 * float(score_sum @ step),
        "n_objects": int(len(score)),
    }


def _bfgs_information(base_center, base_score, base_information, center, score, previous):
    if previous is None:
        previous_center = np.asarray(base_center, dtype=np.float64)
        previous_score = np.asarray(base_score, dtype=np.float64)
        information = np.asarray(base_information, dtype=np.float64)
        source = "pass0_observed_information"
    else:
        payload = json.loads(Path(previous).read_text())
        summary = payload["hybrid"]
        previous_center = np.asarray(summary["center"], dtype=np.float64)
        previous_score = np.asarray(summary["score_sum"], dtype=np.float64)
        information = np.asarray(summary["solver_information_sum"], dtype=np.float64)
        source = str(Path(previous).resolve())
    information = 0.5 * (information + information.T)
    step = np.asarray(center, dtype=np.float64) - previous_center
    score_change = previous_score - np.asarray(score, dtype=np.float64)
    curvature = float(score_change @ step)
    denominator = float(step @ information @ step)
    if not np.isfinite(curvature) or curvature <= 0:
        raise RuntimeError(
            "BFGS score-root update violates positive curvature: "
            f"-(delta score).delta center={curvature}"
        )
    if not np.isfinite(denominator) or denominator <= 0:
        raise RuntimeError(
            f"BFGS source information is not positive along the step: {denominator}"
        )
    projected = information @ step
    updated = (
        information
        - np.outer(projected, projected) / denominator
        + np.outer(score_change, score_change) / curvature
    )
    updated = 0.5 * (updated + updated.T)
    eigenvalues = np.linalg.eigvalsh(updated)
    if not np.isfinite(eigenvalues).all() or eigenvalues[0] <= 0:
        raise RuntimeError(f"BFGS information is non-positive: {eigenvalues.tolist()}")
    return updated, {
        "method": "positive_definite_bfgs_score_root",
        "source": source,
        "previous_center": previous_center.tolist(),
        "previous_score_sum": previous_score.tolist(),
        "center_step": step.tolist(),
        "score_change": score_change.tolist(),
        "curvature": curvature,
        "source_directional_information": denominator,
        "eigenvalues": eigenvalues.tolist(),
    }


# A pass is only comparable to another pass if it saw the same scene, the same
# mock, the same model and the same code.  The partition combiner already
# enforces this across partitions; the hybrid additionally spans passes, where
# a stale directory is a much easier mistake to make.
IDENTITY_NAMES = (
    "scene_sha256",
    "model_sha256",
    "model_cache_sha256",
    "proposal_cache_sha256",
    "mock_input_sha256",
    "injected_shear",
    "likelihood_component_sha256",
    "implementation_sha256",
)


def _identity(result: dict) -> dict:
    return {name: result[name] for name in IDENTITY_NAMES if name in result}


def _require_one_identity(results, labels):
    reference = _identity(results[0])
    for result, label in zip(results[1:], labels[1:]):
        other = _identity(result)
        differing = sorted(
            name
            for name in set(reference) | set(other)
            if reference.get(name) != other.get(name)
        )
        if differing:
            raise RuntimeError(
                f"{label} does not share the identity of {labels[0]}: {differing}"
            )
    return reference


def _load(directory: Path):
    result = json.loads((directory / "result.json").read_text())
    moments = np.load(directory / "one_step_moments.npz")
    if "object_rows" not in moments:
        raise RuntimeError(
            f"{directory} predates --object-subset and does not record which "
            "mock rows its moments belong to; rerun it"
        )
    return (
        np.asarray(result["result"]["center"], dtype=np.float64),
        np.asarray(moments["score"], dtype=np.float64),
        np.asarray(moments["information"], dtype=np.float64),
        np.asarray(moments["object_rows"], dtype=np.int64),
        result,
    )


def _normalization_terms(results, labels, center):
    reports = [
        (result.get("selection") or {}).get("normalization_cache")
        for result in results
    ]
    if all(report is None for report in reports):
        return np.zeros(2), np.zeros((2, 2)), None
    if any(report is None for report in reports):
        raise RuntimeError("normalization cache is absent from only some partitions")
    reference = reports[0]
    for report, label in zip(reports[1:], labels[1:]):
        if report.get("sha256") != reference.get("sha256"):
            raise RuntimeError(
                f"{label} uses a different normalization cache from {labels[0]}"
            )
    path = Path(reference["path"])
    if _sha256(path) != reference.get("sha256"):
        raise RuntimeError(f"normalization cache does not match its recorded hash: {path}")
    normalization = load_population_normalization(path)
    _, gradient, hessian = log_mass_derivatives(normalization, center)
    return gradient, hessian, {
        "path": str(path.resolve()),
        "sha256": reference["sha256"],
        "gradient": gradient.tolist(),
        "hessian": hessian.tolist(),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base",
        nargs="+",
        required=True,
        help="pass-0 run directories, one per partition; the carry source",
    )
    parser.add_argument(
        "--recomputed",
        nargs="+",
        required=True,
        help="bright-only run directories at the new centre, one per partition",
    )
    parser.add_argument(
        "--score-root-bfgs",
        action="store_true",
        help="use a positive-definite BFGS score-root update instead of the local observed Hessian",
    )
    parser.add_argument(
        "--previous",
        help="previous hybrid JSON carrying the prior BFGS information and score",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    base_centers, base_score, base_information, base_rows, base_results = [], [], [], [], []
    for name in args.base:
        c, s, i, r, res = _load(Path(name))
        base_centers.append(c)
        base_score.append(s)
        base_information.append(i)
        base_rows.append(r)
        base_results.append(res)
    if not all(np.array_equal(c, base_centers[0]) for c in base_centers):
        raise RuntimeError("base partitions were evaluated at different centres")
    c0 = base_centers[0]
    rows0 = np.concatenate(base_rows)
    s0 = np.concatenate(base_score)
    i0 = np.concatenate(base_information)
    if np.unique(rows0).size != rows0.size:
        raise RuntimeError("base partitions overlap")
    order = np.argsort(rows0)
    rows0, s0, i0 = rows0[order], s0[order], i0[order]

    new_centers, new_score, new_information, new_rows, new_results = [], [], [], [], []
    for name in args.recomputed:
        c, s, i, r, res = _load(Path(name))
        new_centers.append(c)
        new_score.append(s)
        new_information.append(i)
        new_rows.append(r)
        new_results.append(res)
    if not all(np.array_equal(c, new_centers[0]) for c in new_centers):
        raise RuntimeError("recomputed partitions were evaluated at different centres")
    c1 = new_centers[0]
    rows1 = np.concatenate(new_rows)
    s1 = np.concatenate(new_score)
    i1 = np.concatenate(new_information)
    if np.unique(rows1).size != rows1.size:
        raise RuntimeError("recomputed partitions overlap")
    _require_one_identity(
        base_results + new_results,
        [*args.base, *args.recomputed],
    )
    membership = np.isin(rows1, rows0)
    if not membership.all():
        missing = rows1[~membership]
        raise RuntimeError(
            f"{missing.size} recomputed rows are absent from the base pass, "
            f"first {missing[:5].tolist()}; the two passes must cover the same mock"
        )

    # Carry every object from the base centre, then overwrite the ones that
    # were genuinely re-evaluated.
    shift = c1 - c0
    carried_score = s0 - np.einsum("nij,j->ni", i0, shift)
    carried_information = i0.copy()
    base_gradient, base_hessian, base_normalization = _normalization_terms(
        base_results, args.base, c0
    )
    new_gradient, new_hessian, new_normalization = _normalization_terms(
        new_results, args.recomputed, c1
    )
    if (base_normalization is None) != (new_normalization is None):
        raise RuntimeError("normalization is present at only one hybrid centre")
    # The ordinary carry linearizes the complete per-object likelihood, so it
    # also carries the old -log B expansion.  B is a population term shared by
    # every observation: once its fresh exact stencil exists, replace that old
    # expansion for every carried row.  Recomputed rows are overwritten below
    # and already contain the fresh normalization.
    normalization_score_correction = (
        base_gradient + base_hessian @ shift - new_gradient
    )
    normalization_information_correction = new_hessian - base_hessian
    carried_score += normalization_score_correction
    carried_information += normalization_information_correction
    position = np.searchsorted(rows0, rows1)
    score = carried_score.copy()
    information = carried_information.copy()
    score[position] = s1
    information[position] = i1
    solver_information = None
    solver_method = "observed_newton"
    solver_report = None
    if args.score_root_bfgs:
        solver_information, solver_report = _bfgs_information(
            c0,
            s0.sum(axis=0, dtype=np.float64),
            i0.sum(axis=0, dtype=np.float64),
            c1,
            score.sum(axis=0, dtype=np.float64),
            args.previous,
        )
        solver_method = "positive_definite_bfgs_score_root"
    elif args.previous is not None:
        raise RuntimeError("--previous requires --score-root-bfgs")
    reference = _summarize(
        c1,
        carried_score,
        carried_information,
        solver_information=solver_information,
        solver_method=solver_method,
    )
    hybrid = _summarize(
        c1,
        score,
        information,
        solver_information=solver_information,
        solver_method=solver_method,
    )

    recomputed_fraction = float(rows1.size) / float(rows0.size)
    payload = {
        "method": "hybrid_bright_recompute_faint_linear_carry",
        "base_center": c0.tolist(),
        "recompute_center": c1.tolist(),
        "n_base": int(rows0.size),
        "n_recomputed": int(rows1.size),
        "recomputed_fraction": recomputed_fraction,
        "pass_equivalents": 1.0 + recomputed_fraction,
        # Share of the information the recompute covers, both matrices read at
        # the BASE centre.  Comparing the recomputed information at the new
        # centre against the base total would mix two centres and can exceed
        # one, because the information itself grows as the centre approaches
        # the peak -- which is a real effect, reported separately below.
        "recomputed_information_share": float(
            i0[position][:, 0, 0].sum() / i0[:, 0, 0].sum()
        ),
        "recomputed_information_growth": float(
            i1[:, 0, 0].sum() / i0[position][:, 0, 0].sum()
        ),
        "hybrid": hybrid,
        "carry_only_reference": reference,
        "hybrid_minus_carry_only": (
            np.asarray(hybrid["estimate"]) - np.asarray(reference["estimate"])
        ).tolist(),
        "normalization_correction": {
            "base": base_normalization,
            "recomputed": new_normalization,
            "per_carried_row_score": normalization_score_correction.tolist(),
            "per_carried_row_information": normalization_information_correction.tolist(),
            "n_carried": int(rows0.size - rows1.size),
        },
        "solver": solver_report,
        "base_directories": [str(Path(name).resolve()) for name in args.base],
        "recomputed_directories": [
            str(Path(name).resolve()) for name in args.recomputed
        ],
        "base_moment_sha256": [
            _sha256(Path(name) / "one_step_moments.npz") for name in args.base
        ],
        "recomputed_moment_sha256": [
            _sha256(Path(name) / "one_step_moments.npz") for name in args.recomputed
        ],
        "pipeline_release": base_results[0].get("pipeline_release"),
        "recompute_pipeline_release": new_results[0].get("pipeline_release"),
        "injected_shear": base_results[0].get("injected_shear"),
        "identity": _identity(base_results[0]),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
