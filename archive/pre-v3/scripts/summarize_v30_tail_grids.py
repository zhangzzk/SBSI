"""Count target and all-data flow rows for frozen conditional R_blend grids."""
from __future__ import annotations

import argparse
import csv
import json
import os

import numpy as np

from scripts.analyze_v22_rblend_bin_design import cell_summary, stream_flow_grid_counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", action="append", required=True, help="NAME=NPZ; repeatable")
    parser.add_argument("--flow-catalogue", required=True)
    parser.add_argument("--train-fraction", type=float, default=0.85)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--output", required=True)
    parser.add_argument("--csv", required=True)
    args = parser.parse_args()

    specifications = []
    for value in args.target:
        if "=" not in value:
            raise ValueError("--target must be NAME=NPZ")
        specifications.append(value.split("=", 1))

    designs, csv_rows = {}, []
    common_accounting = None
    for name, path in specifications:
        with np.load(path, allow_pickle=False) as target:
            mag_edges = np.asarray(target["edges_flux"], dtype=float)
            size_edges = np.asarray(target["edges_size"], dtype=float)
            crowd_edges = np.asarray(target["edges_crowd"], dtype=float)
            target_counts = np.asarray(target["counts"], dtype=float)
            probabilities = np.asarray(target["crowd_quantiles"], dtype=float)
            conditional = bool(target["crowd_conditional"])
        expected_shape = (len(mag_edges) - 1, len(size_edges) - 1,
                          len(probabilities) - 1)
        if not conditional or target_counts.shape != expected_shape:
            raise RuntimeError(f"{name}: inconsistent conditional target shape")
        if crowd_edges.shape != (*expected_shape[:2], expected_shape[2] + 1):
            raise RuntimeError(f"{name}: inconsistent conditional edge shape")

        flow_counts, accounting = stream_flow_grid_counts(
            args.flow_catalogue, mag_edges, size_edges, crowd_edges,
        )
        if common_accounting is None:
            common_accounting = accounting
        elif accounting != common_accounting:
            raise RuntimeError("flow accounting differs between candidate grids")
        if flow_counts.shape != target_counts.shape:
            raise RuntimeError(f"{name}: target/flow shape mismatch")
        full_flow_rows = int(flow_counts.sum())
        if full_flow_rows != accounting["finite_rblend"]:
            raise RuntimeError(f"{name}: flow counts do not close")

        boundary_summary = []
        for index, probability in enumerate(probabilities):
            values = crowd_edges[..., index]
            boundary_summary.append({
                "boundary": index, "quantile": float(probability),
                "min": float(values.min()), "median": float(np.median(values)),
                "max": float(values.max()),
            })
        bins = []
        for index in range(expected_shape[-1]):
            target_slice = target_counts[..., index]
            flow_slice = flow_counts[..., index]
            row = {
                "design": name, "bin": index + 1,
                "quantile_lower": float(probabilities[index]),
                "quantile_upper": float(probabilities[index + 1]),
                "edge_lower_median": float(np.median(crowd_edges[..., index])),
                "edge_upper_median": float(np.median(crowd_edges[..., index + 1])),
                "target_rows": float(target_slice.sum()),
                "target_cell_min": float(target_slice.min()),
                "target_cell_p10": float(np.quantile(target_slice, 0.10)),
                "flow_full_rows": int(flow_slice.sum()),
                "flow_full_cell_min": int(flow_slice.min()),
                "flow_all_train_cell_min_expected": float(
                    flow_slice.min() * args.train_fraction),
            }
            bins.append({key: value for key, value in row.items() if key != "design"})
            csv_rows.append(row)
        occupancy = flow_counts / full_flow_rows * args.batch_size
        designs[name] = {
            "target": os.path.abspath(path), "shape": list(expected_shape),
            "quantiles": probabilities.tolist(), "edges": crowd_edges.tolist(),
            "edge_boundary_summary": boundary_summary,
            "target_joint_cell_counts": cell_summary(target_counts),
            "flow_full_joint_cell_counts": cell_summary(flow_counts),
            "flow_all_train_joint_cell_counts_expected": cell_summary(
                flow_counts * args.train_fraction),
            "expected_per_batch_joint_cell_occupancy": cell_summary(occupancy),
            "bins": bins,
        }
        target_summary = designs[name]["target_joint_cell_counts"]
        flow_summary = designs[name]["flow_all_train_joint_cell_counts_expected"]
        print(f"{name} {expected_shape}: target min/p10/median="
              f"{target_summary['min']:.0f}/{target_summary['p10']:.0f}/"
              f"{target_summary['median']:.0f}; all-train min={flow_summary['min']:.0f}")
        print("  boundary medians " + " ".join(
            f"{item['median']:.6g}" for item in boundary_summary))

    result = {
        "status": "half-shear target and g=0 flow count audit; no training or constgold",
        "provenance": {
            "flow_catalogue": os.path.abspath(args.flow_catalogue),
            "flow_accounting": common_accounting,
            "train_fraction": args.train_fraction, "batch_size": args.batch_size,
        },
        "designs": designs,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    with open(args.csv, "x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader(); writer.writerows(csv_rows)
    print(f"saved {args.output} and {args.csv}")
    print("SUMMARIZE_V30_TAIL_GRIDS_DONE")


if __name__ == "__main__":
    main()
