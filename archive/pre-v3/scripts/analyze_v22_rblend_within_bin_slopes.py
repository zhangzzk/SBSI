"""Measure the half-shear R_self slope inside each V2.2 r_blend target bin.

The response target is a 6 (magnitude) x 6 (size) x 5 (r_blend) grid.  A marginal slope inside one
crowding bin can therefore be manufactured by a changing primary-property mixture.  This script
reports both:

* the ordinary pooled OLS slope dR_self/dr_blend; and
* the primary-cell-controlled slope after removing separate means in all 36 magnitude x size cells.

Uncertainty on both slopes is a leave-one-simulation-case-out jackknife.  As a distribution-robust
cross-check, r_blend is also split into quintiles *within every magnitude x size cell* and the
top-minus-bottom R_self change is reported with a case-blocked SEM.  These are descriptive slopes,
not a claim that R_self is globally linear in r_blend.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.feather as pf


KEYS = ["case", "input_index"]


def slope_from_sums(n: np.ndarray, sx: np.ndarray, sy: np.ndarray,
                    sxx: np.ndarray, sxy: np.ndarray) -> float:
    """Within-group OLS slope from sufficient statistics; arrays index control groups."""
    valid = n > 1
    if not np.any(valid):
        return np.nan
    den = np.sum(sxx[valid] - sx[valid] ** 2 / n[valid])
    num = np.sum(sxy[valid] - sx[valid] * sy[valid] / n[valid])
    return float(num / den) if den > 0 else np.nan


def jackknife_sem(estimates: np.ndarray) -> float:
    values = np.asarray(estimates, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 3:
        return np.nan
    mean = values.mean()
    return float(np.sqrt((len(values) - 1) / len(values) * np.sum((values - mean) ** 2)))


def grouped_slope_with_case_jackknife(
    x: np.ndarray, y: np.ndarray, cases: np.ndarray, groups: np.ndarray, ngroup: int,
) -> tuple[float, float, int]:
    """Slope controlling group intercepts, with leave-one-case-out uncertainty."""
    unique_cases = np.unique(cases)
    case_idx = np.searchsorted(unique_cases, cases)
    flat = case_idx * ngroup + groups
    size = len(unique_cases) * ngroup

    def accumulate(weights: np.ndarray | None = None) -> np.ndarray:
        return np.bincount(flat, weights=weights, minlength=size).reshape(len(unique_cases), ngroup)

    n_case = accumulate()
    sx_case = accumulate(x)
    sy_case = accumulate(y)
    sxx_case = accumulate(x * x)
    sxy_case = accumulate(x * y)
    totals = [array.sum(axis=0) for array in (n_case, sx_case, sy_case, sxx_case, sxy_case)]
    estimate = slope_from_sums(*totals)
    leave_one = np.asarray([
        slope_from_sums(*[total - array[k] for total, array in zip(
            totals, (n_case, sx_case, sy_case, sxx_case, sxy_case)
        )])
        for k in range(len(unique_cases))
    ])
    return estimate, jackknife_sem(leave_one), len(unique_cases)


def assign_edge_bins(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    ids = np.digitize(values, edges, right=False) - 1
    ids[(ids < 0) | (ids >= len(edges) - 1)] = -1
    return ids.astype(np.int64)


def within_primary_cell_quintiles(
    x: np.ndarray, mask: np.ndarray, primary_cell: np.ndarray, n_primary_cells: int,
) -> np.ndarray:
    """Assign x quintiles independently inside each primary magnitude-size cell."""
    ids = np.full(len(x), -1, dtype=np.int8)
    for cell in range(n_primary_cells):
        use = mask & (primary_cell == cell)
        if use.sum() < 10:
            continue
        boundaries = np.quantile(x[use], [0.2, 0.4, 0.6, 0.8])
        ids[use] = np.searchsorted(boundaries, x[use], side="right").astype(np.int8)
    return ids


def case_blocked_quintile_profile(
    x: np.ndarray, y: np.ndarray, cases: np.ndarray, quintile: np.ndarray,
) -> tuple[list[dict[str, Any]], dict[str, float | int]]:
    unique_cases = np.unique(cases)
    case_idx = np.searchsorted(unique_cases, cases)
    profile: list[dict[str, Any]] = []
    case_x = np.full((len(unique_cases), 5), np.nan)
    case_y = np.full((len(unique_cases), 5), np.nan)
    for q in range(5):
        use = quintile == q
        for values, output in ((x, case_x), (y, case_y)):
            sums = np.bincount(case_idx[use], weights=values[use], minlength=len(unique_cases))
            counts = np.bincount(case_idx[use], minlength=len(unique_cases))
            output[:, q] = np.divide(
                sums, counts, out=np.full(len(unique_cases), np.nan), where=counts > 0,
            )
        have = np.isfinite(case_y[:, q])
        profile.append({
            "quintile": q + 1,
            "N": int(use.sum()),
            "R_blend_mean": float(x[use].mean()),
            "R_self_mean": float(y[use].mean()),
            "R_self_case_sem": float(case_y[have, q].std(ddof=1) / np.sqrt(have.sum())),
        })
    complete = np.isfinite(case_x[:, [0, 4]]).all(axis=1) & np.isfinite(case_y[:, [0, 4]]).all(axis=1)
    delta_y_case = case_y[complete, 4] - case_y[complete, 0]
    delta_x_case = case_x[complete, 4] - case_x[complete, 0]
    secant_case = delta_y_case / delta_x_case
    delta_y = profile[4]["R_self_mean"] - profile[0]["R_self_mean"]
    delta_x = profile[4]["R_blend_mean"] - profile[0]["R_blend_mean"]
    return profile, {
        "complete_cases": int(complete.sum()),
        "q5_minus_q1_R_self": float(delta_y),
        "q5_minus_q1_R_self_case_sem": float(
            delta_y_case.std(ddof=1) / np.sqrt(len(delta_y_case))
        ),
        "q5_minus_q1_R_blend": float(delta_x),
        "q5_q1_secant_slope": float(delta_y / delta_x),
        "q5_q1_secant_slope_case_sem": float(
            secant_case.std(ddof=1) / np.sqrt(len(secant_case))
        ),
    }


def analyze_window(
    x: np.ndarray, y: np.ndarray, cases: np.ndarray, crowd_bin: np.ndarray,
    primary_cell: np.ndarray, crowd_edges: np.ndarray, window: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for crowd in range(5):
        use = window & (crowd_bin == crowd)
        xb, yb, cb = x[use], y[use], cases[use]
        groups = primary_cell[use]
        pooled, pooled_sem, ncase = grouped_slope_with_case_jackknife(
            xb, yb, cb, np.zeros(len(xb), dtype=np.int64), 1,
        )
        controlled, controlled_sem, controlled_ncase = grouped_slope_with_case_jackknife(
            xb, yb, cb, groups, 36,
        )
        if ncase != controlled_ncase:
            raise RuntimeError("pooled and controlled slopes see different case sets")
        quintile = within_primary_cell_quintiles(x, use, primary_cell, 36)
        q_profile, q_contrast = case_blocked_quintile_profile(
            x[use], y[use], cases[use], quintile[use],
        )
        rows.append({
            "bin": crowd + 1,
            "lower": float(crowd_edges[crowd]),
            "upper": float(crowd_edges[crowd + 1]),
            "N": int(use.sum()),
            "n_cases": ncase,
            "R_blend_mean": float(xb.mean()),
            "R_blend_median": float(np.median(xb)),
            "R_self_mean": float(yb.mean()),
            "pooled_ols_slope": pooled,
            "pooled_ols_case_jackknife_sem": pooled_sem,
            "primary_cell_controlled_ols_slope": controlled,
            "primary_cell_controlled_ols_case_jackknife_sem": controlled_sem,
            "within_primary_cell_quintiles": q_profile,
            **q_contrast,
        })
    return rows


def write_csv(path: str, windows: dict[str, list[dict[str, Any]]]) -> None:
    fields = [
        "window", "bin", "lower", "upper", "N", "n_cases", "R_blend_mean",
        "R_blend_median", "R_self_mean", "pooled_ols_slope",
        "pooled_ols_case_jackknife_sem", "primary_cell_controlled_ols_slope",
        "primary_cell_controlled_ols_case_jackknife_sem", "q5_minus_q1_R_blend",
        "q5_minus_q1_R_self", "q5_minus_q1_R_self_case_sem", "q5_q1_secant_slope",
        "q5_q1_secant_slope_case_sem",
    ]
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for window, rows in windows.items():
            for row in rows:
                writer.writerow({"window": window, **{field: row[field] for field in fields[1:]}})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selfresp", required=True)
    ap.add_argument("--rblend-base", required=True)
    ap.add_argument("--response-target", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--min-rblend-match", type=float, default=0.9999)
    args = ap.parse_args()

    columns = [*KEYS, "r_sim_self", "r_input_p", "Re_input_p"]
    table = pf.read_table(args.selfresp, columns=columns).to_pandas()
    if table.duplicated(KEYS).any():
        raise RuntimeError("self-response keys are not unique")
    mag = table["r_input_p"].to_numpy(float)
    re_ = table["Re_input_p"].to_numpy(float)
    truth = table["r_sim_self"].to_numpy(float)
    case = table["case"].to_numpy(np.int64)
    domain = (np.isfinite(truth) & np.isfinite(mag) & np.isfinite(re_)
              & (mag < 25.8) & (re_ > 0.5) & (case >= 40) & (case <= 199))
    table = table.loc[domain].reset_index(drop=True)

    lookup = pf.read_table(args.rblend_base, columns=[*KEYS, "r_blend"]).to_pandas()
    if lookup.duplicated(KEYS).any():
        raise RuntimeError("r_blend lookup keys are not unique")
    joined = table[KEYS].merge(lookup, on=KEYS, how="left", sort=False, validate="one_to_one")
    if len(joined) != len(table) or not all(
        np.array_equal(joined[key].to_numpy(), table[key].to_numpy()) for key in KEYS
    ):
        raise RuntimeError("r_blend join changed the self-response spine")
    matched = joined["r_blend"].notna().to_numpy()
    match_fraction = float(matched.mean())
    if match_fraction < args.min_rblend_match:
        raise RuntimeError(f"r_blend match fraction {match_fraction:.8%} is too low")
    table = table.loc[matched].reset_index(drop=True)
    rblend = joined.loc[matched, "r_blend"].to_numpy(float)

    with np.load(args.response_target, allow_pickle=False) as target:
        mag_edges = np.asarray(target["edges_flux"], dtype=float)
        size_edges = np.asarray(target["edges_size"], dtype=float)
        crowd_edges = np.asarray(target["edges_crowd"], dtype=float)
    if tuple(map(len, (mag_edges, size_edges, crowd_edges))) != (7, 7, 6):
        raise RuntimeError("response target is not the expected 6x6x5 grid")

    mag = table["r_input_p"].to_numpy(float)
    re_ = table["Re_input_p"].to_numpy(float)
    truth = table["r_sim_self"].to_numpy(float)
    case = table["case"].to_numpy(np.int64)
    mag_bin = assign_edge_bins(mag, mag_edges)
    size_bin = assign_edge_bins(re_, size_edges)
    crowd_bin = assign_edge_bins(rblend, crowd_edges)
    primary_cell = mag_bin * 6 + size_bin
    finite_grid = (mag_bin >= 0) & (size_bin >= 0) & (crowd_bin >= 0) & np.isfinite(rblend)
    if finite_grid.mean() < 0.9999:
        raise RuntimeError(f"only {finite_grid.mean():.8%} of joined rows enter the response grid")

    windows_mask = {
        "all_c40_199": finite_grid,
        "target_overlap_c40_99": finite_grid & (case <= 99),
        "target_heldout_c100_199": finite_grid & (case >= 100),
    }
    windows = {
        name: analyze_window(
            rblend, truth, case, crowd_bin, primary_cell, crowd_edges, window,
        )
        for name, window in windows_mask.items()
    }
    result = {
        "status": "half-shear R_self slopes; descriptive, case-blocked, no constgold",
        "definitions": {
            "pooled_ols_slope": "OLS dR_self/dr_blend with one intercept",
            "primary_cell_controlled_ols_slope": (
                "OLS dR_self/dr_blend after separate intercepts in the 36 target mag-size cells"
            ),
            "q5_minus_q1_R_self": (
                "top-minus-bottom r_blend quintile R_self; quintiles assigned separately in each "
                "target mag-size cell"
            ),
            "slope_uncertainty": "leave-one-simulation-case-out jackknife SEM",
            "quintile_uncertainty": "SEM of per-case q5-q1 contrasts",
        },
        "provenance": {
            "selfresp": os.path.abspath(args.selfresp),
            "rblend_base": os.path.abspath(args.rblend_base),
            "response_target": os.path.abspath(args.response_target),
            "input_domain_rows": int(len(matched)),
            "matched_rows": int(matched.sum()),
            "dropped_unmatched_rblend": int((~matched).sum()),
            "rblend_match_fraction": match_fraction,
            "mag_edges": mag_edges.tolist(),
            "size_edges": size_edges.tolist(),
            "rblend_edges": crowd_edges.tolist(),
        },
        "windows": windows,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    write_csv(args.csv, windows)

    print(f"rows matched={matched.sum():,}/{len(matched):,} ({match_fraction:.6%}); "
          f"grid rows={finite_grid.sum():,}")
    for row in windows["all_c40_199"]:
        print(
            f"bin{row['bin']} [{row['lower']:.6g},{row['upper']:.6g}) N={row['N']:,} "
            f"Rself={row['R_self_mean']:.5f} "
            f"slope_controlled={row['primary_cell_controlled_ols_slope']:+.5f}"
            f"+/-{row['primary_cell_controlled_ols_case_jackknife_sem']:.5f} "
            f"q5-q1={row['q5_minus_q1_R_self']:+.5f}"
            f"+/-{row['q5_minus_q1_R_self_case_sem']:.5f}"
        )
    print(f"saved {args.output} and {args.csv}")
    print("ANALYZE_V22_RBLEND_WITHIN_BIN_SLOPES_DONE")


if __name__ == "__main__":
    main()
