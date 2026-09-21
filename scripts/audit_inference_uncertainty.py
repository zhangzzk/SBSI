#!/usr/bin/env python
"""Audit saved one-step moments, conditional on fixed models and expansion point.

Whole-case resampling preserves within-case dependence. It does not repeat
training, regenerate the prior, move the expansion point, or redraw quadrature.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd


def file_hash(path):
    digest = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def case_sums(values, inverse, n_cases):
    out = np.zeros((n_cases, *values.shape[1:]), dtype=np.float64)
    np.add.at(out, inverse, values)
    return out


def summarize(center, score, information, groups):
    score = np.asarray(score, dtype=np.float64)
    information = np.asarray(information, dtype=np.float64)
    groups = np.asarray(groups)
    if score.ndim != 2 or score.shape[1] != 2 or information.shape != (len(score), 2, 2):
        raise ValueError("require aligned two-component scores and information")
    if groups.shape != (len(score),) or len(score) < 2:
        raise ValueError("require aligned grouping labels and at least two rows")
    if not np.isfinite(score).all() or not np.isfinite(information).all():
        raise ValueError("nonfinite saved moments")
    cases, inverse = np.unique(groups, return_inverse=True)
    if len(cases) < 2:
        raise ValueError("case uncertainty requires at least two cases")
    h = information.sum(axis=0)
    h = (h + h.T) / 2
    if np.linalg.eigvalsh(h)[0] <= 0:
        raise ValueError("nonpositive total information")
    step = np.linalg.solve(h, score.sum(axis=0))
    residual = score - np.einsum("nij,j->ni", information, step)
    # Contributions sum to zero; their scale is contribution to the catalogue
    # estimate, not the conventional N-times-larger influence function.
    contribution = np.linalg.solve(h, residual.T).T
    row_cov = np.cov(contribution, rowvar=False, ddof=1) * len(score)
    grouped = case_sums(contribution, inverse, len(cases))
    case_cov = np.cov(grouped, rowvar=False, ddof=1) * len(cases)
    return {
        "estimate": np.asarray(center) + step,
        "step": step,
        "information_sum": h,
        "row_covariance": row_cov,
        "case_covariance": case_cov,
        "row_se": np.sqrt(np.diag(row_cov)),
        "case_se": np.sqrt(np.diag(case_cov)),
        "case_labels": cases,
        "case_n": np.bincount(inverse),
        "case_score": case_sums(score, inverse, len(cases)),
        "case_information": case_sums(information, inverse, len(cases)),
        "case_contribution": grouped,
        "contribution": contribution,
    }


def resample_cases(center, grouped_score, grouped_information, *, replicates, seed):
    g = len(grouped_score)
    rng = np.random.default_rng(seed)
    weights = rng.multinomial(g, np.full(g, 1.0 / g), size=replicates)
    scores = weights @ grouped_score
    information = (weights @ grouped_information.reshape(g, 4)).reshape(-1, 2, 2)
    information = (information + information.transpose(0, 2, 1)) / 2
    valid = np.isfinite(information).all(axis=(1, 2)) & (np.linalg.eigvalsh(information)[:, 0] > 0)
    estimate = np.asarray(center) + np.linalg.solve(information[valid], scores[valid, :, None])[..., 0]
    return {
        "replicates": replicates, "seed": seed,
        "invalid_information_replicates": int((~valid).sum()),
        "valid_replicates": int(valid.sum()),
        "standard_error": np.std(estimate, axis=0, ddof=1),
        "percentile_95_interval": np.quantile(estimate, [0.025, 0.975], axis=0).T,
        "mean_estimate": estimate.mean(axis=0),
    }


def public_summary(s):
    return {k: s[k] for k in ("estimate", "row_se", "case_se", "row_covariance", "case_covariance")}


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=2026091401)
    parser.add_argument("--classifier-protocol", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists() or args.replicates < 2:
        raise ValueError("require a new output and at least two replicates")
    args.output.mkdir(parents=True)
    run = args.run.resolve()
    result = json.loads((run / "combined/result.json").read_text())
    moment_path = run / "combined" / result["one_step_moments"]["path"]
    if file_hash(moment_path) != result["one_step_moments"]["sha256"]:
        raise ValueError("combined moment hash mismatch")
    truth = pd.read_parquet(run / "input/truth.parquet")
    observed = pd.read_parquet(run / "input/measurements.parquet")
    for name in ("truth.parquet", "measurements.parquet"):
        if file_hash(run / "input" / name) != result["identity"]["mock_input_sha256"][name]:
            raise ValueError("input hash mismatch")
    if truth[["source_case", "source_input_index"]].duplicated().any():
        raise ValueError("duplicate source identities")
    rows = []
    weight_diagnostics = {k: [] for k in ("weight_ess", "weight_max_fraction", "weight_relative_error", "weight_pareto_k")}
    for part in result["partitions"]:
        p = Path(part["root"]) / "one_step_moments.npz"
        if file_hash(p) != part["moments_sha256"]:
            raise ValueError("partition moment hash mismatch")
        with np.load(p) as z:
            rows.append(z["object_rows"])
            for k in weight_diagnostics:
                weight_diagnostics[k].append(z[k][-1])
    np.testing.assert_array_equal(np.concatenate(rows), np.arange(len(truth)))
    n = len(truth)
    if n != result["n_observations"] or len(observed) != n:
        raise ValueError("source coverage differs from combined result")
    groups = truth.source_case.to_numpy()
    center = np.asarray(result["summary"]["center"])
    with np.load(moment_path) as z:
        score, info = z["score"], z["information"]
        s = summarize(center, score, info, groups)
        np.testing.assert_allclose(s["estimate"], result["summary"]["estimate"], rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(s["row_se"], result["summary"]["robust_standard_error"], rtol=1e-12)
        ladder = {}
        previous = None
        for i, rung in enumerate(result["pipeline_config"]["estimator"]["draw_ladder"]):
            v = summarize(center, z["ladder_score"][i], z["ladder_information"][i], groups)
            entry = public_summary(v)
            if previous is not None:
                diff = v["contribution"] - previous["contribution"]
                cdiff = v["case_contribution"] - previous["case_contribution"]
                entry["paired_change_from_previous"] = {
                    "estimate_difference": v["estimate"] - previous["estimate"],
                    "row_se": np.sqrt(np.diag(np.cov(diff, rowvar=False, ddof=1) * n)),
                    "case_se": np.sqrt(np.diag(np.cov(cdiff, rowvar=False, ddof=1) * len(cdiff))),
                    "meaning": "Data-sampling uncertainty of the paired rung difference; not independent-seed integration uncertainty.",
                }
            ladder[str(rung)] = entry
            previous = v
    print("ALIGNED_AND_REPRODUCED", n, "cases", len(s["case_labels"]), flush=True)
    boot = resample_cases(center, s["case_score"], s["case_information"], replicates=args.replicates, seed=args.seed)
    g = len(s["case_labels"])
    leave_h = s["information_sum"] - s["case_information"]
    valid = np.linalg.eigvalsh(leave_h)[:, 0] > 0
    leave = center + np.linalg.solve(leave_h[valid], (score.sum(0) - s["case_score"][valid])[..., None])[..., 0]
    leave_centered = leave - leave.mean(axis=0)
    jack_cov = (g - 1) / g * leave_centered.T @ leave_centered
    protocol = json.loads(args.classifier_protocol.read_text())
    subgroups = {}
    assigned = []
    for name in ("train", "tune"):
        case_ids = protocol["splits"][name]
        assigned.extend(case_ids)
        mask = np.isin(groups, case_ids)
        v = summarize(center, score[mask], info[mask], groups[mask])
        subgroups[name] = {"n": int(mask.sum()), "cases": len(v["case_labels"]), **public_summary(v)}
    mask = ~np.isin(groups, assigned)
    v = summarize(center, score[mask], info[mask], groups[mask])
    subgroups["outside_classifier_train_tune"] = {"n": int(mask.sum()), "case_labels": v["case_labels"], **public_summary(v)}
    influence_square = np.square(s["contribution"])
    tail = {}
    for fraction in (0.01, 0.001, 0.0001):
        top = max(1, int(np.ceil(fraction * n)))
        tail[str(fraction)] = np.sort(influence_square, axis=0)[-top:].sum(0) / influence_square.sum(0)
    weights = {k: np.concatenate(v) for k, v in weight_diagnostics.items()}
    for k, v in weights.items():
        if v.shape != (n,):
            raise ValueError(f"diagnostic shape mismatch: {k} {v.shape}")
    diagnostic = {
        "ess_below_32_fraction": float(np.mean(weights["weight_ess"] < 32)),
        "max_weight_above_half_fraction": float(np.mean(weights["weight_max_fraction"] > 0.5)),
        "percentiles": {k: {"finite_fraction": float(np.isfinite(v).mean()), "values": np.quantile(v[np.isfinite(v)], [0, .01, .5, .9, .99, 1])} for k, v in weights.items()},
    }
    e = observed[["measured_ngmix_g1", "measured_ngmix_g2"]].to_numpy()
    # A measured-mean comparator; not a lower bound on likelihood uncertainty.
    mean_contribution = (e - e.mean(0)) / n
    _, inverse = np.unique(groups, return_inverse=True)
    case_mean_contribution = case_sums(mean_contribution, inverse, g)
    report = {
        "run": str(run), "n": n, "n_cases": g, "center": center,
        "conditioned_on": ["fixed trained likelihood", "fixed finite prior", "fixed selection quadrature", "fixed proposal random streams", "fixed data-derived expansion point"],
        "not_estimated": ["model-training uncertainty", "finite-prior uncertainty", "independent-seed integration error", "nonlinear effect of recomputing the starting point", "iteration and finite-difference truncation bias", "population-domain mismatch"],
        "full_sample": public_summary(s), "case_to_row_se_ratio": s["case_se"] / s["row_se"],
        "case_bootstrap": boot,
        "case_jackknife": {"invalid_information_deletions": int((~valid).sum()), "standard_error": np.sqrt(np.diag(jack_cov)), "estimate_min": leave.min(0), "estimate_max": leave.max(0)},
        "ladder": ladder, "classifier_split_diagnostics": subgroups,
        "influence_variance_share_in_largest_fraction": tail,
        "individual_information_nonpositive_fraction": float(np.mean(np.linalg.eigvalsh(info)[:, 0] <= 0)),
        "mean_information": info.mean(0), "score_covariance": np.cov(score, rowvar=False),
        "final_rung_weight_diagnostics": diagnostic,
        "measured_shape_mean_comparator": {"mean": e.mean(0), "row_se": e.std(0, ddof=1) / np.sqrt(n), "case_se": np.sqrt(np.diag(np.cov(case_mean_contribution, rowvar=False) * g)), "meaning": "Different estimator; not a lower bound on likelihood uncertainty."},
        "measurement_checks": {"all_finite": bool(np.isfinite(observed.to_numpy()).all()), "shape_norm_ge_one": int(np.count_nonzero(np.linalg.norm(e, axis=1) >= 1)), "min_radius_pixels": float(observed.measured_flux_radius.min()), "max_mag_auto": float(observed.measured_mag_auto.max())},
        "hashes": {"script": file_hash(__file__), "combined_result": file_hash(run / "combined/result.json"), "classifier_protocol": file_hash(args.classifier_protocol)},
    }
    (args.output / "result.json").write_text(json.dumps(report, indent=2, default=json_default, allow_nan=False) + "\n")
    pd.DataFrame({"case": s["case_labels"], "n": s["case_n"], "influence_g1": s["case_contribution"][:, 0], "influence_g2": s["case_contribution"][:, 1]}).to_csv(args.output / "case_contributions.csv", index=False)
    print(json.dumps({"row_se": s["row_se"], "case_se": s["case_se"], "bootstrap_se": boot["standard_error"]}, default=json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
