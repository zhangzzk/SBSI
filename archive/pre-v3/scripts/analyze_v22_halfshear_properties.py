"""Profile V2.2 half-shear self-response closure over primary and blend properties.

This is the training-side counterpart to the established constgold localization.  It uses the
forward half-shear estimator and compares only the flow SELF response,

    flow residual = R_flow / R_self - 1,
    m_self        = R_self / R_flow - 1.

No BlendEMU response is added.  The same frozen constgold ``R_blend`` edges are loaded from a
recorded JSON artifact so the scene profiles are directly comparable without deriving a boundary
from the half-shear residual.  Absolute response quantities require the exact 16-seed ensemble.

Uncertainties keep two sources separate: checkpoint-seed SEM and case-blocked sampling SEM of the
half-shear truth.  The model prediction is held fixed on the evaluated population, matching the
established constgold localization convention.  A generalized least-squares constant-profile test
uses their full covariance matrices.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.feather as pf
from scipy.stats import chi2

from sbs_shear import domain as sbs_domain


KEYS = ["case", "input_index"]
EXPECTED_SEEDS = [501, 502, 503, *range(505, 518)]
MAG_EDGES = np.asarray([18, 22, 23, 24, 24.5, 25, 25.5, 26], dtype=float)
SIZE_EDGES = np.asarray([0.50, 0.55, 0.60, 0.70, 0.80, 1.00, 1.20, 1.50], dtype=float)
TRUE_SN_EDGES = np.asarray([0, 5, 7.5, 10, 15, 20, 30, 50, 100, 1e6], dtype=float)
MEASURED_SN_EDGES = np.asarray([0, 10, 20, 30, 50, 100, 200, 1e6], dtype=float)


@dataclass(frozen=True)
class Axis:
    name: str
    values: np.ndarray
    ids: np.ndarray
    labels: list[str]
    lower: list[float | None]
    upper: list[float | None]


def strict_float(value: Any) -> float | None:
    """Return JSON-safe finite floats, preserving missing values as null."""
    x = float(value)
    return x if np.isfinite(x) else None


def edge_axis(name: str, values: np.ndarray, edges: np.ndarray, fmt: str) -> Axis:
    ids = np.digitize(values, edges, right=False) - 1
    ids[(ids < 0) | (ids >= len(edges) - 1)] = -1
    labels = [f"[{fmt.format(edges[i])},{fmt.format(edges[i + 1])})"
              for i in range(len(edges) - 1)]
    return Axis(name, values, ids.astype(np.int64), labels,
                edges[:-1].astype(float).tolist(), edges[1:].astype(float).tolist())


def blend_axis(
    values: np.ndarray, positive_edges: np.ndarray, epsilon: float,
    name: str = "blendness constgold-frozen",
) -> Axis:
    """Use the frozen constgold low class and positive-response quartile boundaries."""
    if len(positive_edges) != 5 or not np.all(np.diff(positive_edges) > 0):
        raise ValueError(f"expected five strictly increasing positive edges, got {positive_edges}")
    if not np.isclose(positive_edges[0], epsilon, rtol=0, atol=1e-6):
        raise ValueError(f"first positive edge {positive_edges[0]} does not match epsilon {epsilon}")
    ids = np.full(len(values), -1, dtype=np.int64)
    ids[values < epsilon] = 0
    for q in range(4):
        use = (values >= positive_edges[q]) & (values < positive_edges[q + 1])
        ids[use] = q + 1
    labels = [f"Rbl<{epsilon:.2f}"] + [
        f"q{q + 1} [{positive_edges[q]:.3f},{positive_edges[q + 1]:.3f})"
        for q in range(4)
    ]
    return Axis(
        name, values, ids, labels,
        [None, *positive_edges[:-1].astype(float).tolist()],
        [float(epsilon), *positive_edges[1:].astype(float).tolist()],
    )


def case_truth_covariance(
    ids: np.ndarray, cases: np.ndarray, truth: np.ndarray, nbin: int,
) -> tuple[np.ndarray, int]:
    """Covariance of the population-mean truth vector, blocking on whole simulation cases."""
    unique = np.unique(cases)
    case_idx = np.searchsorted(unique, cases)
    valid = ids >= 0
    flat = ids[valid] * len(unique) + case_idx[valid]
    size = nbin * len(unique)
    sums = np.bincount(flat, weights=truth[valid], minlength=size).reshape(nbin, len(unique))
    counts = np.bincount(flat, minlength=size).reshape(nbin, len(unique))
    means = np.divide(sums, counts, out=np.full_like(sums, np.nan), where=counts > 0).T
    complete = np.isfinite(means).all(axis=1)
    ncomplete = int(complete.sum())
    if ncomplete < 3:
        raise RuntimeError(f"only {ncomplete} complete cases across {nbin} bins")
    cov = np.atleast_2d(np.cov(means[complete], rowvar=False, ddof=1)) / ncomplete
    return cov, ncomplete


def covariance_of_mean(samples: np.ndarray) -> np.ndarray:
    """Covariance of a sample mean for rows=independent replicates, columns=bins."""
    if samples.shape[0] < 2:
        raise ValueError("at least two replicates are required")
    return np.atleast_2d(np.cov(samples, rowvar=False, ddof=1)) / samples.shape[0]


def constant_shape_test(values: np.ndarray, covariance: np.ndarray) -> dict[str, float | int | None]:
    """Generalized least-squares test of one common level across all bins."""
    n = len(values)
    if n < 2:
        return {"constant": strict_float(values[0]), "chi2": None, "dof": 0, "p_value": None}
    inv = np.linalg.pinv(covariance, rcond=1e-12, hermitian=True)
    one = np.ones(n)
    denom = float(one @ inv @ one)
    if not np.isfinite(denom) or denom <= 0:
        raise RuntimeError("constant-profile covariance is singular in the common-level direction")
    level = float((one @ inv @ values) / denom)
    delta = values - level
    statistic = float(delta @ inv @ delta)
    dof = n - 1
    return {
        "constant": level,
        "chi2": statistic,
        "dof": dof,
        "p_value": float(chi2.sf(statistic, dof)),
    }


def profile_axis(
    axis: Axis,
    window: np.ndarray,
    cases: np.ndarray,
    truth: np.ndarray,
    flow: np.ndarray,
    min_count: int,
) -> dict[str, Any]:
    """Compute one profile and its seed/case covariance decomposition."""
    original_ids = np.where(window, axis.ids, -1)
    kept_bins = [
        b for b in range(len(axis.labels)) if int((original_ids == b).sum()) >= min_count
    ]
    omitted_bins = [
        {"bin": b + 1, "label": axis.labels[b], "N": int((original_ids == b).sum())}
        for b in range(len(axis.labels)) if b not in kept_bins
    ]
    if len(kept_bins) < 2 and axis.name != "global":
        raise RuntimeError(
            f"{axis.name}: fewer than two bins meet minimum count {min_count}: {omitted_bins}"
        )

    # The retained original bin numbers need not be contiguous (for example, the first true-S/N
    # bin can be empty).  Remap them before bincount-based case covariance construction.
    ids = np.full(len(original_ids), -1, dtype=np.int64)
    for j, b in enumerate(kept_bins):
        ids[original_ids == b] = j
    counts = np.asarray([(ids == j).sum() for j in range(len(kept_bins))], dtype=np.int64)
    truth_mean = np.asarray([truth[ids == j].mean() for j in range(len(kept_bins))])
    flow_mean = np.asarray([[flow[ids == j, s].mean() for j in range(len(kept_bins))]
                            for s in range(flow.shape[1])])
    additive = flow_mean - truth_mean[None, :]
    flow_relative = 100.0 * (flow_mean / truth_mean[None, :] - 1.0)
    m_self = 100.0 * (truth_mean[None, :] / flow_mean - 1.0)

    truth_cov, ncomplete = case_truth_covariance(ids, cases, truth, len(kept_bins))
    seed_cov_add = covariance_of_mean(additive)
    seed_cov_rel = covariance_of_mean(flow_relative)
    seed_cov_m = covariance_of_mean(m_self)
    mean_flow = flow_mean.mean(axis=0)
    deriv_rel = -100.0 * mean_flow / truth_mean ** 2
    deriv_m = 100.0 / mean_flow
    sim_cov_add = truth_cov
    sim_cov_rel = np.multiply.outer(deriv_rel, deriv_rel) * truth_cov
    sim_cov_m = np.multiply.outer(deriv_m, deriv_m) * truth_cov
    total_cov_add = seed_cov_add + sim_cov_add
    total_cov_rel = seed_cov_rel + sim_cov_rel
    total_cov_m = seed_cov_m + sim_cov_m

    bins = []
    for j, b in enumerate(kept_bins):
        bins.append({
            "bin": b + 1,
            "label": axis.labels[b],
            "lower": strict_float(axis.lower[b]) if axis.lower[b] is not None else None,
            "upper": strict_float(axis.upper[b]) if axis.upper[b] is not None else None,
            "N": int(counts[j]),
            "R_self": float(truth_mean[j]),
            "R_flow": float(mean_flow[j]),
            "flow_minus_self": float(additive[:, j].mean()),
            "flow_minus_self_seed_sem": float(np.sqrt(seed_cov_add[j, j])),
            "flow_minus_self_sim_sem": float(np.sqrt(sim_cov_add[j, j])),
            "flow_minus_self_total_sem": float(np.sqrt(total_cov_add[j, j])),
            "flow_relative_pct": float(flow_relative[:, j].mean()),
            "flow_relative_seed_sem_pct": float(np.sqrt(seed_cov_rel[j, j])),
            "flow_relative_sim_sem_pct": float(np.sqrt(sim_cov_rel[j, j])),
            "flow_relative_total_sem_pct": float(np.sqrt(total_cov_rel[j, j])),
            "m_self_pct": float(m_self[:, j].mean()),
            "m_self_seed_sem_pct": float(np.sqrt(seed_cov_m[j, j])),
            "m_self_sim_sem_pct": float(np.sqrt(sim_cov_m[j, j])),
            "m_self_total_sem_pct": float(np.sqrt(total_cov_m[j, j])),
        })

    rel_mean = flow_relative.mean(axis=0)
    return {
        "axis": axis.name,
        "assigned_rows": int((ids >= 0).sum()),
        "unassigned_rows": int(window.sum() - (ids >= 0).sum()),
        "outside_axis_rows": int((window & (axis.ids < 0)).sum()),
        "omitted_bins_below_min_count": omitted_bins,
        "complete_cases_for_covariance": ncomplete,
        "flow_relative_shape_test": constant_shape_test(rel_mean, total_cov_rel),
        "flow_relative_range_pct": float(rel_mean.max() - rel_mean.min()),
        "bins": bins,
    }


def load_blend_edges(path: str) -> np.ndarray:
    with open(path) as handle:
        data = json.load(handle)
    try:
        values = data["constgold"]["sixshell"]["profile_by_R_blend"]["positive_quantile_edges"]
    except (KeyError, TypeError) as exc:
        raise RuntimeError(f"cannot find frozen blend edges in {path}") from exc
    return np.asarray(values, dtype=float)


def load_response_target_edges(path: str) -> np.ndarray:
    with np.load(path, allow_pickle=False) as data:
        if "edges_crowd" not in data:
            raise RuntimeError(f"response target {path} has no edges_crowd")
        edges = np.asarray(data["edges_crowd"], dtype=float)
    if len(edges) != 6 or not np.all(np.diff(edges) > 0):
        raise RuntimeError(f"response target has invalid five-bin crowd edges: {edges}")
    return edges


def load_constgold_reference(path: str) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    if not path:
        return out
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            out.setdefault(row["axis"], []).append({
                "label": row["bin"],
                "flow_relative_pct": float(row["flow_residual_pct"]),
                "flow_relative_total_sem_pct": float(row["flow_total_sem_pct"]),
                "N": int(row["N"]),
            })
    return out


def write_csv(path: str, profiles: dict[str, Any]) -> None:
    fields = [
        "window", "axis", "bin", "label", "lower", "upper", "N", "R_self", "R_flow",
        "flow_minus_self", "flow_minus_self_seed_sem", "flow_minus_self_sim_sem",
        "flow_minus_self_total_sem", "flow_relative_pct", "flow_relative_seed_sem_pct",
        "flow_relative_sim_sem_pct", "flow_relative_total_sem_pct", "m_self_pct",
        "m_self_seed_sem_pct", "m_self_sim_sem_pct", "m_self_total_sem_pct",
    ]
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for window, axes in profiles.items():
            for axis_name, profile in axes.items():
                for row in profile["bins"]:
                    writer.writerow({"window": window, "axis": axis_name, **row})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selfresp", required=True)
    ap.add_argument("--rblend-base", required=True)
    ap.add_argument("--blend-edges-json", required=True)
    ap.add_argument("--response-target", required=True)
    ap.add_argument("--constgold-profile", default="")
    ap.add_argument("--output", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--min-count", type=int, default=2000)
    ap.add_argument("--min-rblend-match", type=float, default=0.9999)
    args = ap.parse_args()

    seed_cols = [f"R_flow_s{seed}" for seed in EXPECTED_SEEDS]
    columns = [*KEYS, "r_sim_self", "SN", "r_input_p", "Re_input_p", *seed_cols]
    table = pf.read_table(args.selfresp, columns=columns).to_pandas()
    missing = [column for column in columns if column not in table]
    if missing:
        raise RuntimeError(f"self-response table is missing columns: {missing}")
    if table.duplicated(KEYS).any():
        raise RuntimeError("self-response keys are not unique")

    mag = table["r_input_p"].to_numpy(float)
    re_ = table["Re_input_p"].to_numpy(float)
    case = table["case"].to_numpy(np.int64)
    truth = table["r_sim_self"].to_numpy(float)
    flow = table[seed_cols].to_numpy(float)
    finite = np.isfinite(truth) & np.isfinite(flow).all(axis=1) & np.isfinite(mag) & np.isfinite(re_)
    domain = finite & (mag < 25.8) & (re_ > 0.5) & (case >= 40) & (case <= 199)
    table = table.loc[domain].reset_index(drop=True)

    lookup = pf.read_table(args.rblend_base, columns=[*KEYS, "r_blend"]).to_pandas()
    if lookup.duplicated(KEYS).any():
        raise RuntimeError("r_blend lookup keys are not unique")
    spine = table[KEYS].copy()
    joined = spine.merge(lookup, on=KEYS, how="left", sort=False, validate="one_to_one")
    if len(joined) != len(table):
        raise RuntimeError("r_blend join changed row count")
    if not all(np.array_equal(joined[key].to_numpy(), table[key].to_numpy()) for key in KEYS):
        raise RuntimeError("r_blend join changed row order")
    matched = joined["r_blend"].notna().to_numpy()
    match_fraction = float(matched.mean())
    if match_fraction < args.min_rblend_match:
        raise RuntimeError(
            f"r_blend match fraction {match_fraction:.8%} is below {args.min_rblend_match:.8%}"
        )
    # Explicit inner-join semantics: unmatched rows are dropped and never zero-filled.
    table = table.loc[matched].reset_index(drop=True)
    rblend = joined.loc[matched, "r_blend"].to_numpy(float)
    if not np.isfinite(rblend).all():
        raise RuntimeError("non-finite r_blend values after the strict join")

    mag = table["r_input_p"].to_numpy(float)
    re_ = table["Re_input_p"].to_numpy(float)
    measured_sn = table["SN"].to_numpy(float)
    true_sn = sbs_domain.sn_true(mag, re_)
    case = table["case"].to_numpy(np.int64)
    truth = table["r_sim_self"].to_numpy(float)
    flow = table[seed_cols].to_numpy(float)
    blend_edges = load_blend_edges(args.blend_edges_json)
    target_blend_edges = load_response_target_edges(args.response_target)
    axes = [
        edge_axis("primary magnitude", mag, MAG_EDGES, "{:.1f}"),
        edge_axis("primary size", re_, SIZE_EDGES, "{:.2f}"),
        edge_axis("primary true S/N", true_sn, TRUE_SN_EDGES, "{:.1f}"),
        edge_axis("primary measured S/N", measured_sn, MEASURED_SN_EDGES, "{:.1f}"),
        edge_axis("blendness target-grid bins", rblend, target_blend_edges, "{:.3f}"),
        blend_axis(rblend, blend_edges, epsilon=0.02),
    ]
    windows = {
        "all_c40_199": np.ones(len(table), dtype=bool),
        "target_overlap_c40_99": (case >= 40) & (case <= 99),
        "target_heldout_c100_199": (case >= 100) & (case <= 199),
    }
    profiles: dict[str, Any] = {}
    for window_name, window in windows.items():
        profiles[window_name] = {
            axis.name: profile_axis(axis, window, case, truth, flow, args.min_count)
            for axis in axes
        }

    global_axis = Axis(
        "global", np.zeros(len(table)), np.zeros(len(table), dtype=np.int64),
        ["all"], [None], [None],
    )
    global_profiles = {
        name: profile_axis(global_axis, window, case, truth, flow, args.min_count)
        for name, window in windows.items()
    }

    result = {
        "status": "16-seed half-shear self-response diagnostic; constgold is not used for training",
        "definition": {
            "flow_relative_pct": "100 * (R_flow / R_self - 1); negative means missing flow response",
            "m_self_pct": "100 * (R_self / R_flow - 1)",
            "response_estimator": "forward half-shear, primary random-shear projection",
        },
        "provenance": {
            "selfresp": os.path.abspath(args.selfresp),
            "rblend_base": os.path.abspath(args.rblend_base),
            "blend_edges_json": os.path.abspath(args.blend_edges_json),
            "response_target": os.path.abspath(args.response_target),
            "constgold_profile": os.path.abspath(args.constgold_profile) if args.constgold_profile else None,
            "seeds": EXPECTED_SEEDS,
            "input_domain_rows": int(len(matched)),
            "matched_rows": int(matched.sum()),
            "dropped_unmatched_rblend": int((~matched).sum()),
            "rblend_match_fraction": match_fraction,
            "positive_R_blend_edges": blend_edges.tolist(),
            "response_target_R_blend_edges": target_blend_edges.tolist(),
        },
        "global": global_profiles,
        "profiles": profiles,
        "constgold_reference": load_constgold_reference(args.constgold_profile),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    write_csv(args.csv, profiles)

    print(f"rows after V2.2 domain={len(matched):,}; r_blend matched={matched.sum():,} "
          f"({match_fraction:.6%}); dropped={int((~matched).sum()):,}")
    for name, profile in global_profiles.items():
        row = profile["bins"][0]
        print(f"{name}: R_flow/R_self-1={row['flow_relative_pct']:+.4f}% "
              f"+/-{row['flow_relative_total_sem_pct']:.4f}% total; "
              f"m_self={row['m_self_pct']:+.4f}%")
    for axis in axes:
        shape = profiles["all_c40_199"][axis.name]["flow_relative_shape_test"]
        spread = profiles["all_c40_199"][axis.name]["flow_relative_range_pct"]
        print(f"{axis.name}: range={spread:.4f} points; "
              f"constant chi2/dof={shape['chi2']:.3f}/{shape['dof']} p={shape['p_value']:.4g}")
    print(f"saved {args.output} and {args.csv}")
    print("ANALYZE_V22_HALFSHEAR_PROPERTIES_DONE")


if __name__ == "__main__":
    main()
