"""Case-block stability gate for proposed conditional-r_blend response targets.

Edges are frozen from the full cases 0--99 target before this audit.  The direct SNC response is
then reconstructed once and accumulated by simulation case and response cell.  Cases, not millions
of galaxies, are the independent units for the stability uncertainty.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.feather as pf
import pyarrow.ipc as ipc

from sbs_shear.measurement_model import add_measurement_target_features, raw_columns_for_measurement_targets
from scripts.compute_response_target_blend import _selection_cuts, load_snc_lookup
from sbs_shear.preprocessing import source_select_selection


def holm_rejections(p_values: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    rejected = np.zeros(len(p), dtype=bool)
    for rank, index in enumerate(order):
        if p[index] <= alpha / (len(p) - rank):
            rejected[index] = True
        else:
            break
    return rejected


def assign_conditional_cells(
    mag: np.ndarray, size: np.ndarray, crowd: np.ndarray,
    mag_edges: np.ndarray, size_edges: np.ndarray, crowd_edges: np.ndarray,
) -> np.ndarray:
    nf, ns, nc = len(mag_edges) - 1, len(size_edges) - 1, crowd_edges.shape[-1] - 1
    fi = np.clip(np.digitize(mag, mag_edges) - 1, 0, nf - 1)
    si = np.clip(np.digitize(size, size_edges) - 1, 0, ns - 1)
    row_edges = crowd_edges[fi, si]
    ci = np.sum(crowd[:, None] >= row_edges[:, 1:-1], axis=1)
    return ((fi * ns + si) * nc + ci).astype(np.int64)


def load_response_rows(args: argparse.Namespace) -> pd.DataFrame:
    snc = load_snc_lookup(args.snc_lookup, args.snc_cols)
    need = set(raw_columns_for_measurement_targets(args.target_cols))
    need |= {"gamma1_input_p", "gamma2_input_p", "detected", "r_input_p", "Re_input_p",
             "distance", "neighbored", "input_index", "case", "r_blend"}
    parts = []
    with ipc.open_file(args.catalogue) as reader:
        available = set(reader.schema.names)
        missing = sorted(need - available)
        if missing:
            raise KeyError(f"target catalogue lacks columns {missing}")
        columns = sorted(need)
        for batch_index in range(reader.num_record_batches):
            frame = pa.Table.from_batches([reader.get_batch(batch_index)]).select(columns).to_pandas()
            frame = frame[(frame["case"] >= 0) & (frame["case"] <= 99)]
            if len(frame) == 0:
                continue
            frame = source_select_selection(frame, cuts=_selection_cuts(args))
            frame = frame[frame["detected"].astype(bool)].reset_index(drop=True)
            if len(frame):
                parts.append(frame)
    frame = pd.concat(parts, ignore_index=True)
    g1 = frame["gamma1_input_p"].to_numpy(float)
    g2 = frame["gamma2_input_p"].to_numpy(float)
    gmag = np.hypot(g1, g2)
    keep = gmag > 1e-6
    frame = frame.loc[keep].reset_index(drop=True)
    gh1, gh2 = g1[keep] / gmag[keep], g2[keep] / gmag[keep]
    measured = add_measurement_target_features(frame.copy())
    e1 = measured[args.target_cols[0]].to_numpy(float)
    e2 = measured[args.target_cols[1]].to_numpy(float)
    key = frame["case"].to_numpy(np.int64) * 1_000_003 + frame["input_index"].to_numpy(np.int64)
    pos = np.clip(np.searchsorted(snc["key"], key), 0, len(snc["key"]) - 1)
    match = snc["key"][pos] == key
    e1 = e1 - np.where(match, snc["e1"][pos], np.nan)
    e2 = e2 - np.where(match, snc["e2"][pos], np.nan)
    response = (e1 * gh1 + e2 * gh2) / args.nominal_g

    raw_key = key
    _, inverse, multiplicity = np.unique(raw_key, return_inverse=True, return_counts=True)
    weight = 1.0 / multiplicity[inverse]
    finite = (np.isfinite(response) & np.isfinite(frame["r_input_p"].to_numpy(float))
              & np.isfinite(frame["Re_input_p"].to_numpy(float))
              & np.isfinite(frame["r_blend"].to_numpy(float)))
    out = frame.loc[finite, ["case", "r_input_p", "Re_input_p", "r_blend"]].reset_index(drop=True)
    out["response"] = response[finite]
    out["weight"] = weight[finite]
    print(f"audit rows={len(out):,}; SNC matched={int(match.sum()):,}/{len(match):,}")
    return out


def audit_target(name: str, path: str, rows: pd.DataFrame, alpha: float) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    with np.load(path, allow_pickle=False) as target:
        mag_edges = np.asarray(target["edges_flux"], dtype=float)
        size_edges = np.asarray(target["edges_size"], dtype=float)
        crowd_edges = np.asarray(target["edges_crowd"], dtype=float)
        stored_response = np.asarray(target["Rsim"], dtype=float)
        stored_counts = np.asarray(target["counts"], dtype=float)
        conditional = bool(target["crowd_conditional"])
    if not conditional or crowd_edges.ndim != 3 or stored_response.ndim != 3:
        raise RuntimeError(f"{name}: target is not a conditional 3D crowd grid")
    nf, ns, nc = stored_response.shape
    if crowd_edges.shape != (nf, ns, nc + 1):
        raise RuntimeError(f"{name}: inconsistent edge shape")
    mag = rows["r_input_p"].to_numpy(float)
    size = rows["Re_input_p"].to_numpy(float)
    crowd = rows["r_blend"].to_numpy(float)
    response = rows["response"].to_numpy(float)
    weight = rows["weight"].to_numpy(float)
    cases = rows["case"].to_numpy(np.int64)
    cell = assign_conditional_cells(mag, size, crowd, mag_edges, size_edges, crowd_edges)
    ncell = nf * ns * nc
    flat = cases * ncell + cell
    sums = np.bincount(flat, weights=weight * response, minlength=100 * ncell).reshape(100, ncell)
    counts = np.bincount(flat, weights=weight, minlength=100 * ncell).reshape(100, ncell)
    case_mean = np.divide(sums, counts, out=np.full_like(sums, np.nan), where=counts > 0)
    total_response = np.divide(sums.sum(axis=0), counts.sum(axis=0),
                               out=np.full(ncell, np.nan), where=counts.sum(axis=0) > 0)
    np.testing.assert_allclose(counts.sum(axis=0), stored_counts.reshape(-1), rtol=0, atol=1e-6)
    np.testing.assert_allclose(total_response, stored_response.reshape(-1), rtol=0, atol=2e-12)

    left, right = np.arange(50), np.arange(50, 100)
    left_mean = np.nanmean(case_mean[left], axis=0)
    right_mean = np.nanmean(case_mean[right], axis=0)
    left_n = np.isfinite(case_mean[left]).sum(axis=0)
    right_n = np.isfinite(case_mean[right]).sum(axis=0)
    left_sem = np.nanstd(case_mean[left], axis=0, ddof=1) / np.sqrt(left_n)
    right_sem = np.nanstd(case_mean[right], axis=0, ddof=1) / np.sqrt(right_n)
    split_difference = right_mean - left_mean
    split_sem = np.hypot(left_sem, right_sem)
    z = np.divide(split_difference, split_sem, out=np.zeros_like(split_difference), where=split_sem > 0)
    # Normal approximation is adequate for two independent groups of 50 cases and avoids treating
    # millions of object rows as independent. Holm controls the 192/288 simultaneous cell checks.
    p = np.asarray([math.erfc(abs(value) / math.sqrt(2.0)) for value in z])
    rejected = holm_rejections(p, alpha=alpha)
    abs_diff = np.abs(split_difference)
    practical = {
        "min_target_count_at_least_2000": bool(stored_counts.min() >= 2000),
        "all_cells_present_in_each_half": bool((left_n == 50).all() and (right_n == 50).all()),
        "no_holm_rejections": bool(not rejected.any()),
        "p95_absolute_split_below_0p10": bool(np.quantile(abs_diff, 0.95) < 0.10),
        "max_absolute_split_below_0p25": bool(abs_diff.max() < 0.25),
    }
    cell_rows = []
    for index in range(ncell):
        fi = index // (ns * nc)
        rem = index % (ns * nc)
        si, ci = rem // nc, rem % nc
        cell_rows.append({
            "target": name, "flux_bin": fi + 1, "size_bin": si + 1, "crowd_bin": ci + 1,
            "count": float(stored_counts.reshape(-1)[index]),
            "Rsim": float(stored_response.reshape(-1)[index]),
            "case_sem": float(np.nanstd(case_mean[:, index], ddof=1) / np.sqrt(100)),
            "Rsim_c0_49_case_mean": float(left_mean[index]),
            "Rsim_c50_99_case_mean": float(right_mean[index]),
            "split_difference": float(split_difference[index]),
            "split_sem": float(split_sem[index]), "split_z": float(z[index]),
            "holm_reject": bool(rejected[index]),
        })
    size_profile = []
    shaped_count = stored_counts.reshape(nf, ns, nc)
    shaped_response = stored_response.reshape(nf, ns, nc)
    for si in range(ns):
        c = shaped_count[:, si, :]
        r = shaped_response[:, si, :]
        size_profile.append({
            "size_bin": si + 1, "lower": float(size_edges[si]), "upper": float(size_edges[si + 1]),
            "count": float(c.sum()), "Rsim_count_weighted": float((c * r).sum() / c.sum()),
        })
    summary = {
        "target": os.path.abspath(path), "shape": [nf, ns, nc],
        "min_count": float(stored_counts.min()), "median_count": float(np.median(stored_counts)),
        "Rsim_min": float(stored_response.min()), "Rsim_max": float(stored_response.max()),
        "case_sem_median": float(np.median([row["case_sem"] for row in cell_rows])),
        "case_sem_p95": float(np.quantile([row["case_sem"] for row in cell_rows], 0.95)),
        "split_abs_median": float(np.median(abs_diff)),
        "split_abs_p95": float(np.quantile(abs_diff, 0.95)),
        "split_abs_max": float(abs_diff.max()), "split_max_abs_z": float(np.max(np.abs(z))),
        "holm_rejections": int(rejected.sum()), "criteria": practical,
        "pass": bool(all(practical.values())), "size_profile": size_profile,
    }
    return summary, cell_rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--catalogue", required=True)
    ap.add_argument("--snc-lookup", required=True)
    ap.add_argument("--target", action="append", required=True, help="NAME=NPZ; repeatable")
    ap.add_argument("--target-cols", nargs=2, default=["measured_ngmix_g1", "measured_ngmix_g2"])
    ap.add_argument("--snc-cols", nargs=2, default=["ngmix0_g1", "ngmix0_g2"])
    ap.add_argument("--nominal-g", type=float, default=0.05)
    ap.add_argument("--primary-mag-max", type=float, default=25.8)
    ap.add_argument("--primary-re-min", type=float, default=0.5)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--output", required=True)
    ap.add_argument("--csv", required=True)
    args = ap.parse_args()
    specs = []
    for value in args.target:
        if "=" not in value:
            raise ValueError("--target must be NAME=NPZ")
        name, path = value.split("=", 1)
        specs.append((name, path))
    rows = load_response_rows(args)
    summaries, all_cells = {}, []
    for name, path in specs:
        summary, cells = audit_target(name, path, rows, args.alpha)
        summaries[name] = summary
        all_cells.extend(cells)
        print(f"{name} {summary['shape']}: count min={summary['min_count']:.0f}; "
              f"split |dR| median/p95/max={summary['split_abs_median']:.4f}/"
              f"{summary['split_abs_p95']:.4f}/{summary['split_abs_max']:.4f}; "
              f"max|z|={summary['split_max_abs_z']:.2f}; Holm rejects="
              f"{summary['holm_rejections']}; PASS={summary['pass']}")
    result = {
        "status": "pre-training frozen-edge case stability gate; no constgold and no retraining",
        "definitions": {
            "independent_unit": "simulation case",
            "split": "cases 0-49 versus 50-99 with frozen full-sample edges",
            "multiplicity": f"Holm family-wise alpha={args.alpha} separately per target",
        },
        "provenance": {"catalogue": os.path.abspath(args.catalogue),
                       "snc_lookup": os.path.abspath(args.snc_lookup), "rows": len(rows)},
        "targets": summaries,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    with open(args.csv, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_cells[0]))
        writer.writeheader(); writer.writerows(all_cells)
    print(f"saved {args.output} and {args.csv}")
    print("AUDIT_V22_CONDITIONAL_TARGET_STABILITY_DONE")


if __name__ == "__main__":
    main()
