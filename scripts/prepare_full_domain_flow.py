#!/usr/bin/env python3
"""Materialize independent per-leg rows for a full-domain measurement flow.

Every valid measured row is retained independently in each leg.  There is no
truth-property cut, measured magnitude/radius cut, or fixed-leg membership
anchor.  Missing and invalid measurements are represented by the separate
usable-event model and therefore do not become flow targets.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc

from sbsi.fixed_g0_domain import (
    FLOW_FEATURES,
    FLOW_TARGETS,
    FlowRows,
    make_flow_rows,
    packed_keys,
    split_group_values,
)


COMMON_COLUMNS = [
    "case",
    "input_index",
    "detected",
    "e1_input_rot0_p",
    "e2_input_rot0_p",
    "gamma1_input_p",
    "gamma2_input_p",
    "sersic_n_input_p",
    "r_input_p",
    "Re_input_p",
    "nbr_flux_near",
    "nbr_flux_far",
    "nbr_flux_max",
    "measured_ngmix_g1",
    "measured_ngmix_g2",
    "measured_mag_auto",
    "measured_flux_radius",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8_388_608), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def selected_columns(reader: ipc.RecordBatchFileReader) -> list[str]:
    available = set(reader.schema.names)
    missing = sorted(set(COMMON_COLUMNS) - available)
    if missing:
        raise KeyError(f"flow catalogue lacks required columns: {missing}")
    columns = list(COMMON_COLUMNS)
    if "axis_ratio_input_p" in available:
        columns.append("axis_ratio_input_p")
    return columns


def append_rows(store: dict[int, list[FlowRows]], rows: FlowRows) -> None:
    for case in np.unique(rows.case):
        take = rows.case == case
        store[int(case)].append(
            FlowRows(
                case=rows.case[take],
                input_index=rows.input_index[take],
                context=rows.context[take],
                target=rows.target[take],
                gamma=rows.gamma[take],
                usable_mask=np.ones(int(take.sum()), dtype=bool),
            )
        )


def save_rows(root: Path, leg: str, store: dict[int, list[FlowRows]]) -> dict[str, dict]:
    destination = root / "flow" / leg
    destination.mkdir(parents=True, exist_ok=False)
    records = {}
    for case in sorted(store):
        parts = store[case]
        case_values = np.concatenate([part.case for part in parts])
        input_index = np.concatenate([part.input_index for part in parts])
        context = np.concatenate([part.context for part in parts])
        target = np.concatenate([part.target for part in parts])
        gamma = np.concatenate([part.gamma for part in parts])
        keys = packed_keys(case_values, input_index)
        if len(np.unique(keys)) != len(keys):
            raise RuntimeError(f"duplicate usable flow key: {leg} case{case:03d}")
        order = np.argsort(keys, kind="stable")
        path = destination / f"case{case:03d}.npz"
        temporary = destination / f"case{case:03d}.tmp.npz"
        np.savez(
            temporary,
            case=case_values[order],
            input_index=input_index[order],
            context=context[order],
            target=target[order],
            gamma=gamma[order],
        )
        os.replace(temporary, path)
        records[str(case)] = {"rows": int(len(order)), "sha256": sha256(path)}
    return records


def truth_parent_mask(frame, magnitude_max: float) -> np.ndarray:
    """Return the strict finite truth-r-magnitude parent boundary."""

    magnitude = frame["r_input_p"].to_numpy(dtype=float, copy=False)
    return np.isfinite(magnitude) & (magnitude < float(magnitude_max))


def read_leg(
    path: Path, leg: str, truth_magnitude_max: float
) -> tuple[dict[int, list[FlowRows]], dict]:
    parts: dict[int, list[FlowRows]] = defaultdict(list)
    counts = {
        "raw_rows": 0,
        "truth_parent_rows": 0,
        "truth_magnitude_dropped_rows": 0,
        "usable_rows": 0,
        "invalid_rows_after_truth_cut": 0,
    }
    with ipc.open_file(path) as reader:
        columns = selected_columns(reader)
        for batch_index in range(reader.num_record_batches):
            frame = (
                pa.Table.from_batches([reader.get_batch(batch_index)])
                .select(columns)
                .to_pandas()
            )
            keep = truth_parent_mask(frame, truth_magnitude_max)
            selected = frame.loc[keep].reset_index(drop=True)
            counts["raw_rows"] += len(frame)
            counts["truth_parent_rows"] += len(selected)
            counts["truth_magnitude_dropped_rows"] += len(frame) - len(selected)
            rows = make_flow_rows(selected)
            counts["usable_rows"] += len(rows.case)
            counts["invalid_rows_after_truth_cut"] += len(selected) - len(rows.case)
            append_rows(parts, rows)
            if (batch_index + 1) % 100 == 0:
                print(
                    f"FULL_DOMAIN_{leg.upper()} batches={batch_index + 1}/"
                    f"{reader.num_record_batches} raw={counts['raw_rows']:,} "
                    f"usable={counts['usable_rows']:,}",
                    flush=True,
                )
    return parts, counts


def prepare(args: argparse.Namespace) -> None:
    if not os.environ.get("SLURM_JOB_ID") and not args.allow_login_smoke:
        raise RuntimeError("full-domain catalogue preparation must run under Slurm")
    if not np.isfinite(args.truth_magnitude_max):
        raise ValueError("truth magnitude boundary must be finite")
    root = args.output_root.resolve()
    if root.exists():
        raise FileExistsError(f"preserving existing full-domain output: {root}")
    root.mkdir(parents=True)
    started = time.monotonic()
    stores = {}
    counts = {}
    for leg, path in (("g0", args.g0_catalogue), ("g05", args.g05_catalogue)):
        stores[leg], counts[leg] = read_leg(path, leg, args.truth_magnitude_max)
    cases0 = set(stores["g0"])
    cases1 = set(stores["g05"])
    if cases0 != cases1:
        raise RuntimeError(
            f"per-leg case sets differ: g0-only={sorted(cases0-cases1)}, "
            f"g05-only={sorted(cases1-cases0)}"
        )
    records = {
        leg: save_rows(root, leg, stores[leg]) for leg in ("g0", "g05")
    }
    cases = sorted(cases0)
    train_cases, validation_cases = split_group_values(
        cases, args.seed, args.validation_size
    )
    manifest = {
        "format_version": 1,
        "population": {
            "name": "full_valid_measurement_domain_true_mag_lt26_v1",
            "anchor_leg": None,
            "predicate": None,
            "truth_analysis_cut": {
                "column": "r_input_p",
                "operator": "<",
                "value": float(args.truth_magnitude_max),
            },
            "measured_selection_cut": None,
            "sheared_leg_recut": None,
            "independent_per_leg_validity": True,
            "inherited_parent_support_only": False,
        },
        "flow": {
            "features": list(FLOW_FEATURES),
            "targets": list(FLOW_TARGETS),
            "per_leg_validity": (
                "detected; finite condition/target/gamma; strictly interior ngmix shape; "
                "positive measured radius and flux"
            ),
            "g0_cases": records["g0"],
            "g05_cases": records["g05"],
        },
        "split": {
            "seed": int(args.seed),
            "validation_size": float(args.validation_size),
            "train_cases": train_cases.tolist(),
            "validation_cases": validation_cases.tolist(),
        },
        "counts": counts,
        "paths": {
            "g0_catalogue": str(args.g0_catalogue.resolve()),
            "g05_catalogue": str(args.g05_catalogue.resolve()),
        },
        "source_files": {
            "g0_catalogue": {
                "size": args.g0_catalogue.stat().st_size,
                "mtime_ns": args.g0_catalogue.stat().st_mtime_ns,
                "sha256": sha256(args.g0_catalogue),
            },
            "g05_catalogue": {
                "size": args.g05_catalogue.stat().st_size,
                "mtime_ns": args.g05_catalogue.stat().st_mtime_ns,
                "sha256": sha256(args.g05_catalogue),
            },
            "script_sha256": sha256(Path(__file__).resolve()),
        },
        "elapsed_seconds": time.monotonic() - started,
    }
    write_json(root / "manifest.json", manifest)
    print(
        f"FULL_DOMAIN_FLOW_PREPARED cases={len(cases)} "
        f"g0={counts['g0']['usable_rows']:,} g05={counts['g05']['usable_rows']:,}",
        flush=True,
    )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--g0-catalogue", type=Path, required=True)
    result.add_argument("--g05-catalogue", type=Path, required=True)
    result.add_argument("--output-root", type=Path, required=True)
    result.add_argument("--seed", type=int, default=501)
    result.add_argument("--validation-size", type=float, default=0.2)
    result.add_argument("--truth-magnitude-max", type=float, default=26.0)
    result.add_argument("--allow-login-smoke", action="store_true")
    return result


if __name__ == "__main__":
    prepare(parser().parse_args())
