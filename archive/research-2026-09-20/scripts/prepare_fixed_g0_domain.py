#!/usr/bin/env python3
"""Materialize a fixed-g0 cohort for flow and response-emulator retraining.

The anchor is evaluated exactly once on the measured g=0 catalogue:

    measured_mag_auto < 25.8
    optionally, measured_flux_radius > 0.6 arcsec = 3.0 pixels

No truth-property selection is performed.  The anchor keys are then carried
to the g=0.05 leg.  Per-leg invalid measurements are removed only from the
conditional-flow rows and are counted explicitly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.feather as feather
import pyarrow.ipc as ipc

from sbsi.fixed_g0_domain import (
    FLUX_RADIUS_MIN_ARCSEC,
    FLUX_RADIUS_MIN_PIXELS,
    FLOW_FEATURES,
    FLOW_TARGETS,
    MAG_AUTO_MAX,
    FlowRows,
    anchor_membership,
    fixed_g0_anchor_mask,
    make_flow_rows,
    packed_keys,
    response_primary_target_mask,
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
    required = set(COMMON_COLUMNS)
    missing = sorted(required - available)
    if missing:
        raise KeyError(f"flow catalogue lacks required columns: {missing}")
    columns = list(COMMON_COLUMNS)
    if "axis_ratio_input_p" in available:
        columns.append("axis_ratio_input_p")
    return columns


def append_rows(store: dict[int, list[FlowRows]], rows: FlowRows) -> None:
    if len(rows.case) == 0:
        return
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


def concatenate_rows(parts: list[FlowRows]) -> FlowRows:
    return FlowRows(
        case=np.concatenate([part.case for part in parts]),
        input_index=np.concatenate([part.input_index for part in parts]),
        context=np.concatenate([part.context for part in parts]),
        target=np.concatenate([part.target for part in parts]),
        gamma=np.concatenate([part.gamma for part in parts]),
        usable_mask=np.ones(sum(len(part.case) for part in parts), dtype=bool),
    )


def save_rows(root: Path, leg: str, store: dict[int, list[FlowRows]]) -> dict[str, dict]:
    destination = root / "flow" / leg
    destination.mkdir(parents=True, exist_ok=False)
    records = {}
    for case in sorted(store):
        rows = concatenate_rows(store[case])
        keys = packed_keys(rows.case, rows.input_index)
        if len(np.unique(keys)) != len(keys):
            raise RuntimeError(f"duplicate usable flow key: {leg} case{case:03d}")
        order = np.argsort(keys, kind="stable")
        path = destination / f"case{case:03d}.npz"
        temporary = destination / f"case{case:03d}.tmp.npz"
        np.savez(
            temporary,
            case=rows.case[order],
            input_index=rows.input_index[order],
            context=rows.context[order],
            target=rows.target[order],
            gamma=rows.gamma[order],
        )
        os.replace(temporary, path)
        records[str(case)] = {
            "rows": int(len(order)),
            "sha256": sha256(path),
        }
    return records


def prepare(args: argparse.Namespace) -> None:
    if not os.environ.get("SLURM_JOB_ID") and not args.allow_login_smoke:
        raise RuntimeError("full catalogue preparation must run under Slurm")
    root = args.output_root.resolve()
    if root.exists():
        raise FileExistsError(f"preserving existing fixed-domain output: {root}")
    root.mkdir(parents=True)
    started = time.monotonic()
    radius_min_pixels = None if args.no_flux_radius_cut else FLUX_RADIUS_MIN_PIXELS

    g0_parts: dict[int, list[FlowRows]] = defaultdict(list)
    anchor_case = []
    anchor_index = []
    counts = {
        "g0_raw_rows": 0,
        "g0_detected_rows": 0,
        "g0_anchor_rows": 0,
        "g0_anchor_invalid_flow_rows": 0,
        "g05_raw_rows": 0,
        "g05_anchor_rows_present": 0,
        "g05_anchor_invalid_flow_rows": 0,
    }

    with ipc.open_file(args.g0_catalogue) as reader:
        columns = selected_columns(reader)
        for batch_index in range(reader.num_record_batches):
            frame = pa.Table.from_batches([reader.get_batch(batch_index)]).select(columns).to_pandas()
            counts["g0_raw_rows"] += len(frame)
            counts["g0_detected_rows"] += int(frame["detected"].fillna(False).sum())
            keep = fixed_g0_anchor_mask(frame, radius_min_pixels=radius_min_pixels)
            selected = frame.loc[keep].reset_index(drop=True)
            counts["g0_anchor_rows"] += len(selected)
            if len(selected):
                anchor_case.append(selected["case"].to_numpy(np.int32))
                anchor_index.append(selected["input_index"].to_numpy(np.int64))
                rows = make_flow_rows(selected)
                counts["g0_anchor_invalid_flow_rows"] += len(selected) - len(rows.case)
                append_rows(g0_parts, rows)
            if (batch_index + 1) % 50 == 0:
                print(
                    f"G0_PROGRESS batches={batch_index + 1}/{reader.num_record_batches} "
                    f"raw={counts['g0_raw_rows']:,} anchor={counts['g0_anchor_rows']:,}",
                    flush=True,
                )

    if not anchor_case:
        raise RuntimeError("fixed g=0 measured selection retained no rows")
    anchor_case_array = np.concatenate(anchor_case)
    anchor_index_array = np.concatenate(anchor_index)
    anchor_keys = packed_keys(anchor_case_array, anchor_index_array)
    if len(np.unique(anchor_keys)) != len(anchor_keys):
        raise RuntimeError("fixed g=0 anchor contains duplicate (case,input_index) keys")
    anchor_order = np.argsort(anchor_keys, kind="stable")
    anchor_keys = anchor_keys[anchor_order]
    anchor_frame = pd.DataFrame(
        {
            "case": anchor_case_array[anchor_order],
            "input_index": anchor_index_array[anchor_order],
        }
    )
    anchor_path = root / "flow_anchor.feather"
    temporary_anchor = root / "flow_anchor.tmp.feather"
    feather.write_feather(anchor_frame, temporary_anchor, compression="zstd")
    os.replace(temporary_anchor, anchor_path)

    # BlendEMU's response catalogue targets the complementary primary half of
    # each rendered scene, whereas the self-response flow targets the secondary
    # half.  Build a separate key anchor from the primary g=0 measurements;
    # both anchors implement the same measured predicate and neither filters
    # truth properties.
    response_anchor_parts = []
    response_counts = {
        "raw_cross_rows": 0,
        "cross_without_shape": 0,
        "shape_without_cross": 0,
        "primary_target_cross_rows": 0,
        "non_primary_target_cross_rows": 0,
        "selected_rows": 0,
    }
    response_role_splits = {}
    for case in sorted(anchor_frame["case"].unique().astype(int)):
        catalogue_root = args.raw_simulation_root / f"case{case}_0.0" / "real0" / "catalogues"
        cross = pd.read_feather(
            catalogue_root / "CrossMatch" / f"{args.tile_name}_rot0_matched.feather",
            columns=["id_detec", "id_input"],
        )
        shape = pd.read_feather(
            catalogue_root / "Shapes" / f"shape_catalogue_detect_position_{args.tile_name}.feather",
            columns=["NUMBER", "MAG_AUTO", "FLUX_RADIUS"],
        )
        if cross["id_detec"].duplicated().any() or cross["id_input"].duplicated().any():
            raise RuntimeError(f"non-unique g=0 primary crossmatch in case{case:03d}")
        if shape["NUMBER"].duplicated().any():
            raise RuntimeError(f"non-unique g=0 primary shape row in case{case:03d}")
        joined = cross.merge(
            shape,
            left_on="id_detec",
            right_on="NUMBER",
            how="inner",
            validate="one_to_one",
        )
        response_counts["raw_cross_rows"] += len(cross)
        response_counts["cross_without_shape"] += len(cross) - len(joined)
        response_counts["shape_without_cross"] += len(shape) - len(joined)
        generated_path = args.raw_simulation_root / f"gals{case}_{args.response_shear_tag}.feather"
        if not generated_path.exists():
            raise FileNotFoundError(
                f"missing response generated catalogue for target-role split: {generated_path}"
            )
        total_input_count = feather.read_table(generated_path, columns=["index"]).num_rows
        primary_target = response_primary_target_mask(joined["id_input"].to_numpy(), total_input_count)
        n_primary = int(primary_target.sum())
        response_counts["primary_target_cross_rows"] += n_primary
        response_counts["non_primary_target_cross_rows"] += len(joined) - n_primary
        response_role_splits[str(case)] = {
            "generated_catalogue": str(generated_path.resolve()),
            "total_input_count": int(total_input_count),
            "primary_id_upper_exclusive": int(total_input_count // 2),
            "joined_primary_target_rows": n_primary,
            "joined_non_primary_target_rows": int(len(joined) - n_primary),
        }
        joined = joined.loc[primary_target].reset_index(drop=True)
        measured = pd.DataFrame(
            {
                "detected": np.ones(len(joined), dtype=bool),
                "measured_mag_auto": joined["MAG_AUTO"].to_numpy(float),
                "measured_flux_radius": joined["FLUX_RADIUS"].to_numpy(float),
            }
        )
        keep = fixed_g0_anchor_mask(measured, radius_min_pixels=radius_min_pixels)
        selected_response = pd.DataFrame(
            {
                "case": np.full(int(keep.sum()), case, dtype=np.int32),
                "input_index": joined.loc[keep, "id_input"].to_numpy(np.int64),
            }
        )
        response_counts["selected_rows"] += len(selected_response)
        response_anchor_parts.append(selected_response)
        if case % 20 == 19:
            print(
                f"RESPONSE_ANCHOR_PROGRESS case={case} selected={response_counts['selected_rows']:,}",
                flush=True,
            )
    response_anchor = pd.concat(response_anchor_parts, ignore_index=True)
    response_keys = packed_keys(response_anchor["case"], response_anchor["input_index"])
    if len(np.unique(response_keys)) != len(response_keys):
        raise RuntimeError("response-emulator g=0 anchor contains duplicate keys")
    response_order = np.argsort(response_keys, kind="stable")
    response_anchor = response_anchor.iloc[response_order].reset_index(drop=True)
    response_anchor_path = root / "response_anchor.feather"
    temporary_response_anchor = root / "response_anchor.tmp.feather"
    feather.write_feather(response_anchor, temporary_response_anchor, compression="zstd")
    os.replace(temporary_response_anchor, response_anchor_path)

    g05_parts: dict[int, list[FlowRows]] = defaultdict(list)
    seen_keys = []
    with ipc.open_file(args.g05_catalogue) as reader:
        columns = selected_columns(reader)
        for batch_index in range(reader.num_record_batches):
            frame = pa.Table.from_batches([reader.get_batch(batch_index)]).select(columns).to_pandas()
            counts["g05_raw_rows"] += len(frame)
            keep = anchor_membership(frame, anchor_keys)
            selected = frame.loc[keep].reset_index(drop=True)
            counts["g05_anchor_rows_present"] += len(selected)
            if len(selected):
                seen_keys.append(packed_keys(selected["case"].to_numpy(), selected["input_index"].to_numpy()))
                rows = make_flow_rows(selected)
                counts["g05_anchor_invalid_flow_rows"] += len(selected) - len(rows.case)
                append_rows(g05_parts, rows)
            if (batch_index + 1) % 100 == 0:
                print(
                    f"G05_PROGRESS batches={batch_index + 1}/{reader.num_record_batches} "
                    f"raw={counts['g05_raw_rows']:,} anchor_present={counts['g05_anchor_rows_present']:,}",
                    flush=True,
                )

    present = np.concatenate(seen_keys) if seen_keys else np.empty(0, dtype=np.uint64)
    if len(np.unique(present)) != len(present):
        raise RuntimeError("g=0.05 catalogue contains duplicate anchored keys")
    counts["g05_anchor_missing_rows"] = int(len(anchor_keys) - len(present))
    if not np.isin(present, anchor_keys, assume_unique=True).all():
        raise RuntimeError("g=0.05 preparation escaped the fixed g=0 anchor")

    g0_records = save_rows(root, "g0", g0_parts)
    g05_records = save_rows(root, "g05", g05_parts)
    cases = sorted(set(map(int, anchor_frame["case"].unique())))
    train_cases, validation_cases = split_group_values(cases, args.seed, args.validation_size)
    if radius_min_pixels is None:
        population_name = "fixed_g0_measured_mag258_no_radius_cut_primaryrole_v1"
        predicate = (
            "detected and finite(measured_mag_auto,measured_flux_radius) and "
            "measured_mag_auto < 25.8; no measured flux-radius science cut"
        )
    else:
        population_name = "fixed_g0_measured_mag258_radius060_primaryrole_v2"
        predicate = (
            "detected and finite(measured_mag_auto,measured_flux_radius) and "
            "measured_mag_auto < 25.8 and measured_flux_radius > 3.0 pixels"
        )
    manifest = {
        "format_version": 2,
        "population": {
            "name": population_name,
            "anchor_leg": "g=0",
            "predicate": predicate,
            "mag_auto_max": MAG_AUTO_MAX,
            "flux_radius_min_arcsec": (None if radius_min_pixels is None else FLUX_RADIUS_MIN_ARCSEC),
            "flux_radius_min_pixels": radius_min_pixels,
            "strict_radius_inequality": radius_min_pixels is not None,
            "truth_analysis_cut": None,
            "sheared_leg_recut": False,
            "inherited_parent_support_only": True,
            "response_target_role": (
                "stable input_index < floor(generated response catalogue row count / 2)"
            ),
        },
        "flow": {
            "features": list(FLOW_FEATURES),
            "targets": list(FLOW_TARGETS),
            "per_leg_validity": (
                "detected; finite condition/target/gamma; strictly interior ngmix shape; "
                "positive measured radius and flux"
            ),
            "g0_cases": g0_records,
            "g05_cases": g05_records,
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
            "raw_simulation_root": str(args.raw_simulation_root.resolve()),
            "response_shear_tag": args.response_shear_tag,
            "flow_anchor": str(anchor_path),
            "response_anchor": str(response_anchor_path),
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
        "anchors": {
            "flow": {
                "rows": len(anchor_frame),
                "sha256": sha256(anchor_path),
                "target_role": "secondary/self-response flow",
            },
            "response": {
                "rows": len(response_anchor),
                "sha256": sha256(response_anchor_path),
                "target_role": "primary/blending-response emulator",
                "target_role_definition": (
                    f"input_index < floor(total rows in gals{{case}}_{args.response_shear_tag}.feather / 2)"
                ),
                "counts": response_counts,
                "role_splits": response_role_splits,
            },
        },
        "elapsed_seconds": time.monotonic() - started,
    }
    write_json(root / "manifest.json", manifest)
    print(
        f"FIXED_G0_DOMAIN_PREPARED anchor={len(anchor_keys):,} "
        f"response_anchor={len(response_anchor):,} "
        f"g0_usable={sum(v['rows'] for v in g0_records.values()):,} "
        f"g05_usable={sum(v['rows'] for v in g05_records.values()):,} "
        f"g05_anchor_missing={counts['g05_anchor_missing_rows']:,}",
        flush=True,
    )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--g0-catalogue", type=Path, required=True)
    result.add_argument("--g05-catalogue", type=Path, required=True)
    result.add_argument("--output-root", type=Path, required=True)
    result.add_argument("--raw-simulation-root", type=Path, required=True)
    result.add_argument("--tile-name", default="tile180.0_-0.5")
    result.add_argument(
        "--response-shear-tag",
        default="0.2",
        help="generated-catalogue shear tag used by BlendEMU to split primary/secondary roles",
    )
    result.add_argument("--seed", type=int, default=501)
    result.add_argument("--validation-size", type=float, default=0.2)
    result.add_argument(
        "--no-flux-radius-cut",
        action="store_true",
        help="retain every finite g=0 FLUX_RADIUS in the anchor; flow validity still requires positive radius",
    )
    result.add_argument("--allow-login-smoke", action="store_true")
    return result


if __name__ == "__main__":
    prepare(parser().parse_args())
