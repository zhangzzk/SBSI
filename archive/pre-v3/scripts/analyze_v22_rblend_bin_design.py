"""Compare proposed V2.2 r_blend grids before any flow retraining.

The decile target is built on the exact half-shear target population.  Because every proposed
global grid is a strict merge of those deciles, its target counts and edges are exact without
replaying the response estimator.  The g=0 flow catalogue is streamed with the exact V2.2 source,
detection, shear-case and finite-target cuts used by the trainer.  Counts under the historical 4M
reservoir are expectations under its uniform priority sampling, not invented exact integers.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc


DESIGNS = {
    # (start-inclusive, stop-exclusive) decile slices.  All boundaries are chosen from half-shear
    # quantiles only; constgold residuals never set an edge.
    "baseline5": [(0, 2), (2, 4), (4, 6), (6, 8), (8, 10)],
    "focused8": [(0, 2), (2, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 10)],
    "focused9": [(0, 1), (1, 2), (2, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 10)],
    "decile10": [(i, i + 1) for i in range(10)],
}


def merge_last_axis(counts: np.ndarray, segments: list[tuple[int, int]]) -> np.ndarray:
    return np.stack([counts[..., lo:hi].sum(axis=-1) for lo, hi in segments], axis=-1)


def edge_indices(segments: list[tuple[int, int]]) -> list[int]:
    if not segments or segments[0][0] != 0 or segments[-1][1] != 10:
        raise ValueError("segments must cover all ten deciles")
    for left, right in zip(segments[:-1], segments[1:]):
        if left[1] != right[0]:
            raise ValueError("segments must be contiguous")
    return [segments[0][0], *[segment[1] for segment in segments]]


def stream_flow_grid_counts(
    path: str, mag_edges: np.ndarray, size_edges: np.ndarray, crowd_edges: np.ndarray,
) -> tuple[np.ndarray, dict[str, int]]:
    columns = [
        "shear_case", "r_input_p", "Re_input_p", "distance", "neighbored", "detected",
        "measured_ngmix_g1", "measured_ngmix_g2", "measured_mag_auto",
        "measured_flux_radius", "r_blend",
    ]
    n_flux, n_size = len(mag_edges) - 1, len(size_edges) - 1
    if crowd_edges.ndim == 1:
        n_crowd = len(crowd_edges) - 1
        conditional = False
    elif crowd_edges.ndim == 3 and crowd_edges.shape[:2] == (n_flux, n_size):
        n_crowd = crowd_edges.shape[2] - 1
        conditional = True
    else:
        raise ValueError(f"unsupported crowd edge shape {crowd_edges.shape}")
    counts = np.zeros((n_flux, n_size, n_crowd), dtype=np.int64)
    accounting = {"raw": 0, "after_source": 0, "detected": 0, "finite_targets": 0,
                  "finite_rblend": 0}
    with ipc.open_file(path) as reader:
        available = set(reader.schema.names)
        missing = sorted(set(columns) - available)
        if missing:
            raise KeyError(f"flow catalogue lacks required columns: {missing}")
        for batch_index in range(reader.num_record_batches):
            frame = pa.Table.from_batches([reader.get_batch(batch_index)]).select(columns).to_pandas()
            accounting["raw"] += len(frame)
            shear = np.isclose(frame["shear_case"].to_numpy(float), 0.0)
            mag = frame["r_input_p"].to_numpy(float)
            size = frame["Re_input_p"].to_numpy(float)
            distance = frame["distance"].to_numpy(float)
            neighbored = frame["neighbored"].astype(bool).to_numpy()
            source = (shear & (mag > 18.0) & (mag < 25.8) & (size > 0.5) & (size < 1.5)
                      & (((distance > 0.0) & (distance < 5.0)) | ~neighbored))
            accounting["after_source"] += int(source.sum())
            selected = source & frame["detected"].astype(bool).to_numpy()
            accounting["detected"] += int(selected.sum())
            finite = (selected & np.isfinite(frame["measured_ngmix_g1"].to_numpy(float))
                      & np.isfinite(frame["measured_ngmix_g2"].to_numpy(float))
                      & np.isfinite(frame["measured_mag_auto"].to_numpy(float)))
            radius = frame["measured_flux_radius"].to_numpy(float)
            finite &= np.isfinite(radius) & (radius > 0.0)
            accounting["finite_targets"] += int(finite.sum())
            crowd = frame["r_blend"].to_numpy(float)
            finite &= np.isfinite(crowd)
            accounting["finite_rblend"] += int(finite.sum())
            if not finite.any():
                continue
            fi = np.clip(np.digitize(mag[finite], mag_edges) - 1, 0, n_flux - 1)
            si = np.clip(np.digitize(size[finite], size_edges) - 1, 0, n_size - 1)
            if conditional:
                row_edges = crowd_edges[fi, si]
                ci = np.sum(crowd[finite, None] >= row_edges[:, 1:-1], axis=1)
            else:
                ci = np.clip(np.digitize(crowd[finite], crowd_edges) - 1, 0, n_crowd - 1)
            flat = (fi * n_size + si) * n_crowd + ci
            counts += np.bincount(
                flat, minlength=n_flux * n_size * n_crowd,
            ).reshape(n_flux, n_size, n_crowd)
    return counts, accounting


def cell_summary(counts: np.ndarray) -> dict[str, float | int]:
    flat = counts.reshape(-1)
    return {
        "min": float(flat.min()),
        "p10": float(np.quantile(flat, 0.10)),
        "median": float(np.median(flat)),
        "max": float(flat.max()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--decile-target", required=True)
    ap.add_argument("--baseline-target", required=True)
    ap.add_argument("--conditional-target", default=None)
    ap.add_argument("--flow-catalogue", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--historical-cap", type=int, default=4_000_000)
    ap.add_argument("--train-fraction", type=float, default=0.85)
    ap.add_argument("--batch-size", type=int, default=8192)
    args = ap.parse_args()

    with np.load(args.decile_target, allow_pickle=False) as target:
        mag_edges = np.asarray(target["edges_flux"], dtype=float)
        size_edges = np.asarray(target["edges_size"], dtype=float)
        crowd_edges = np.asarray(target["edges_crowd"], dtype=float)
        target_decile_counts = np.asarray(target["counts"], dtype=float)
    if target_decile_counts.shape != (6, 6, 10) or len(crowd_edges) != 11:
        raise RuntimeError("decile target is not a 6x6x10 grid")
    with np.load(args.baseline_target, allow_pickle=False) as baseline:
        baseline_edges = np.asarray(baseline["edges_crowd"], dtype=float)
        baseline_counts = np.asarray(baseline["counts"], dtype=float)
    merged_baseline = merge_last_axis(target_decile_counts, DESIGNS["baseline5"])
    if not np.allclose(crowd_edges[::2], baseline_edges, rtol=0, atol=2e-7):
        raise RuntimeError("decile edges do not reproduce the accepted five-bin target")
    if not np.allclose(merged_baseline, baseline_counts, rtol=0, atol=2):
        raise RuntimeError("merged decile counts do not reproduce accepted five-bin counts")

    flow_decile_counts, accounting = stream_flow_grid_counts(
        args.flow_catalogue, mag_edges, size_edges, crowd_edges,
    )
    full_flow_rows = int(flow_decile_counts.sum())
    if full_flow_rows != accounting["finite_rblend"]:
        raise RuntimeError("flow grid does not account for every finite selected row")
    cap_fraction = min(1.0, args.historical_cap / full_flow_rows)

    designs: dict[str, Any] = {}
    csv_rows: list[dict[str, Any]] = []
    for name, segments in DESIGNS.items():
        idx = edge_indices(segments)
        edges = crowd_edges[idx]
        target_counts = merge_last_axis(target_decile_counts, segments)
        flow_counts = merge_last_axis(flow_decile_counts, segments).astype(float)
        target_by_bin = target_counts.sum(axis=(0, 1))
        flow_by_bin = flow_counts.sum(axis=(0, 1))
        bins = []
        for b in range(len(segments)):
            cap_used = flow_by_bin[b] * cap_fraction
            row = {
                "bin": b + 1,
                "lower": float(edges[b]),
                "upper": float(edges[b + 1]),
                "target_rows": int(round(target_by_bin[b])),
                "flow_full_rows": int(round(flow_by_bin[b])),
                "flow_4m_used_expected": float(cap_used),
                "flow_4m_train_expected": float(cap_used * args.train_fraction),
                "flow_all_train_expected": float(flow_by_bin[b] * args.train_fraction),
                "target_cell_min": float(target_counts[..., b].min()),
                "target_cell_p10": float(np.quantile(target_counts[..., b], 0.10)),
                "flow_full_cell_min": float(flow_counts[..., b].min()),
                "flow_4m_train_cell_min_expected": float(
                    flow_counts[..., b].min() * cap_fraction * args.train_fraction
                ),
            }
            bins.append(row)
            csv_rows.append({"design": name, **row})
        batch_occupancy = flow_counts / full_flow_rows * args.batch_size
        designs[name] = {
            "n_bins": len(segments),
            "quantile_segments": [[lo / 10, hi / 10] for lo, hi in segments],
            "edges": edges.tolist(),
            "target_joint_cell_counts": cell_summary(target_counts),
            "flow_full_joint_cell_counts": cell_summary(flow_counts),
            "flow_4m_train_joint_cell_counts_expected": cell_summary(
                flow_counts * cap_fraction * args.train_fraction
            ),
            "flow_all_train_joint_cell_counts_expected": cell_summary(
                flow_counts * args.train_fraction
            ),
            "expected_per_batch_joint_cell_occupancy": cell_summary(batch_occupancy),
            "bins": bins,
        }

    conditional_provenance = None
    if args.conditional_target:
        with np.load(args.conditional_target, allow_pickle=False) as conditional_target:
            conditional_edges = np.asarray(conditional_target["edges_crowd"], dtype=float)
            conditional_counts = np.asarray(conditional_target["counts"], dtype=float)
            conditional_mag_edges = np.asarray(conditional_target["edges_flux"], dtype=float)
            conditional_size_edges = np.asarray(conditional_target["edges_size"], dtype=float)
            stored_conditional = bool(conditional_target["crowd_conditional"])
        if (not stored_conditional or conditional_counts.shape != (6, 6, 8)
                or conditional_edges.shape != (6, 6, 9)):
            raise RuntimeError("conditional target is not the expected 6x6x8 conditional grid")
        if (not np.allclose(conditional_mag_edges, mag_edges)
                or not np.allclose(conditional_size_edges, size_edges)):
            raise RuntimeError("conditional and global targets use different primary grids")
        conditional_flow, conditional_accounting = stream_flow_grid_counts(
            args.flow_catalogue, mag_edges, size_edges, conditional_edges,
        )
        if conditional_accounting != accounting or int(conditional_flow.sum()) != full_flow_rows:
            raise RuntimeError("conditional/global flow accounting differs")
        edge_summary = []
        for boundary in range(9):
            values = conditional_edges[..., boundary]
            edge_summary.append({
                "boundary": boundary,
                "quantile": boundary / 8,
                "min": float(values.min()),
                "median": float(np.median(values)),
                "max": float(values.max()),
            })
        bins = []
        for b in range(8):
            target_by_bin = float(conditional_counts[..., b].sum())
            flow_by_bin = float(conditional_flow[..., b].sum())
            cap_used = flow_by_bin * cap_fraction
            row = {
                "bin": b + 1,
                # A conditional grid has 36 boundaries per ordinal bin.  CSV uses their medians;
                # the JSON carries every edge plus min/median/max summaries.
                "lower": float(np.median(conditional_edges[..., b])),
                "upper": float(np.median(conditional_edges[..., b + 1])),
                "target_rows": int(round(target_by_bin)),
                "flow_full_rows": int(round(flow_by_bin)),
                "flow_4m_used_expected": float(cap_used),
                "flow_4m_train_expected": float(cap_used * args.train_fraction),
                "flow_all_train_expected": float(flow_by_bin * args.train_fraction),
                "target_cell_min": float(conditional_counts[..., b].min()),
                "target_cell_p10": float(np.quantile(conditional_counts[..., b], 0.10)),
                "flow_full_cell_min": float(conditional_flow[..., b].min()),
                "flow_4m_train_cell_min_expected": float(
                    conditional_flow[..., b].min() * cap_fraction * args.train_fraction
                ),
            }
            bins.append(row)
            csv_rows.append({"design": "conditional8", **row})
        batch_occupancy = conditional_flow / full_flow_rows * args.batch_size
        designs["conditional8"] = {
            "n_bins": 8,
            "conditional_on_primary_cell": True,
            "edges": conditional_edges.tolist(),
            "edge_boundary_summary": edge_summary,
            "target_joint_cell_counts": cell_summary(conditional_counts),
            "flow_full_joint_cell_counts": cell_summary(conditional_flow),
            "flow_4m_train_joint_cell_counts_expected": cell_summary(
                conditional_flow * cap_fraction * args.train_fraction
            ),
            "flow_all_train_joint_cell_counts_expected": cell_summary(
                conditional_flow * args.train_fraction
            ),
            "expected_per_batch_joint_cell_occupancy": cell_summary(batch_occupancy),
            "bins": bins,
        }
        conditional_provenance = os.path.abspath(args.conditional_target)

    result = {
        "status": "bin/sample design only; no retraining and no constgold-derived edge",
        "definitions": {
            "focused8": "keep first two current bins; split current bins 3, 4, 5 at half-shear medians",
            "focused9": "focused8 plus split current bin 1; keep weak current bin 2 intact",
            "decile10": "ten equal-count half-shear bins",
            "conditional8": (
                "eight half-shear R_blend quantiles separately inside each 6x6 primary cell"
            ),
            "cap_counts": "expectations under the trainer's uniform priority reservoir",
            "target_counts": "effective unique-target counts from the direct half-shear target",
        },
        "provenance": {
            "decile_target": os.path.abspath(args.decile_target),
            "baseline_target": os.path.abspath(args.baseline_target),
            "conditional_target": conditional_provenance,
            "flow_catalogue": os.path.abspath(args.flow_catalogue),
            "flow_accounting": accounting,
            "flow_full_rows": full_flow_rows,
            "historical_cap": args.historical_cap,
            "historical_cap_fraction": cap_fraction,
            "train_fraction": args.train_fraction,
            "historical_train_rows": args.historical_cap * args.train_fraction,
            "all_data_train_rows": full_flow_rows * args.train_fraction,
            "batch_size": args.batch_size,
        },
        "designs": designs,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    fields = list(csv_rows[0])
    with open(args.csv, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"flow exact selected rows={full_flow_rows:,}; historical cap keeps {cap_fraction:.2%}")
    for name, design in designs.items():
        tc = design["target_joint_cell_counts"]
        fc = design["flow_4m_train_joint_cell_counts_expected"]
        bc = design["expected_per_batch_joint_cell_occupancy"]
        print(f"{name}: {design['n_bins']} bins; target joint min/p10/median="
              f"{tc['min']:.0f}/{tc['p10']:.0f}/{tc['median']:.0f}; "
              f"4M-train expected joint min={fc['min']:.0f}; batch min/p10="
              f"{bc['min']:.2f}/{bc['p10']:.2f}")
        if design.get("conditional_on_primary_cell"):
            print("  conditional boundary medians " + " ".join(
                f"{row['median']:.6g}" for row in design["edge_boundary_summary"]
            ))
        else:
            print("  edges " + " ".join(f"{value:.6g}" for value in design["edges"]))
    print(f"saved {args.output} and {args.csv}")
    print("ANALYZE_V22_RBLEND_BIN_DESIGN_DONE")


if __name__ == "__main__":
    main()
