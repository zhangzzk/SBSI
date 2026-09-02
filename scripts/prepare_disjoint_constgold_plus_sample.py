#!/usr/bin/env python
"""Draw a reproducible ConstGold plus-leg sample disjoint from frozen identities."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.ipc as ipc


SOURCE_COLUMNS = (
    "case",
    "input_index",
    "neighbored",
    "distance",
    "r_input_p",
    "Re_input_p",
    "measured_e1_plus",
    "measured_e2_plus",
    "measured_mag_auto_plus",
    "measured_flux_radius_plus",
)


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _keys(case, input_index):
    case = np.asarray(case, dtype=np.int64)
    index = np.asarray(input_index, dtype=np.int64)
    if (case < 0).any() or (index < 0).any() or (index >= 2**32).any():
        raise ValueError("case/input_index cannot be packed into nonnegative 32-bit keys")
    return (case << np.int64(32)) | index


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--constgold", type=Path, required=True)
    parser.add_argument("--exclude-sample", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--min-case", type=int, default=40)
    parser.add_argument("--max-case", type=int, default=140)
    parser.add_argument("--pixel-size", type=float, default=0.2)
    parser.add_argument("--shape-max", type=float, default=0.6)
    parser.add_argument("--mag-max", type=float, default=25.8)
    parser.add_argument("--radius-min", type=float, default=0.75)
    return parser.parse_args(argv)


def _and(*masks):
    result = masks[0]
    for mask in masks[1:]:
        result = pc.and_(result, mask)
    return pc.fill_null(result, False)


def main(argv=None):
    args = parse_args(argv)
    if args.sample_size <= 0:
        raise ValueError("sample-size must be positive")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    excluded = pd.read_parquet(args.exclude_sample, columns=["case", "input_index"])
    excluded_keys = np.unique(_keys(excluded["case"], excluded["input_index"]))
    if len(excluded_keys) != len(excluded):
        raise ValueError("exclude sample contains duplicate identities")

    eligible_parts = []
    counts = {
        "catalogue_rows": 0,
        "case_range": 0,
        "source_and_domain": 0,
        "finite_plus": 0,
        "measured_selection": 0,
        "excluded_existing_identity": 0,
        "eligible_after_exclusion": 0,
    }
    with pa.memory_map(str(args.constgold), "r") as stream:
        reader = ipc.open_file(stream)
        missing = sorted(set(SOURCE_COLUMNS) - set(reader.schema.names))
        if missing:
            raise KeyError(f"ConstGold source lacks columns: {missing}")
        for batch_index in range(reader.num_record_batches):
            table = pa.Table.from_batches([reader.get_batch(batch_index)]).select(
                SOURCE_COLUMNS
            )
            counts["catalogue_rows"] += table.num_rows
            case_mask = _and(
                pc.greater_equal(table["case"], args.min_case),
                pc.less(table["case"], args.max_case),
            )
            counts["case_range"] += int(pc.sum(pc.cast(case_mask, "int64")).as_py())
            truth_mask = _and(
                case_mask,
                pc.greater(table["r_input_p"], 18.0),
                pc.less(table["r_input_p"], 25.8),
                pc.greater(table["Re_input_p"], 0.5),
                pc.less(table["Re_input_p"], 1.5),
                pc.or_(
                    _and(
                        pc.greater(table["distance"], 0.0),
                        pc.less(table["distance"], 5.0),
                    ),
                    pc.invert(table["neighbored"]),
                ),
            )
            selected_truth = table.filter(truth_mask)
            counts["source_and_domain"] += selected_truth.num_rows
            if not selected_truth.num_rows:
                continue
            frame = selected_truth.to_pandas()
            measured = frame[
                [
                    "measured_e1_plus",
                    "measured_e2_plus",
                    "measured_mag_auto_plus",
                    "measured_flux_radius_plus",
                ]
            ].to_numpy(float)
            finite = np.isfinite(measured).all(axis=1)
            counts["finite_plus"] += int(finite.sum())
            selected = (
                finite
                & (np.hypot(measured[:, 0], measured[:, 1]) < args.shape_max)
                & (measured[:, 2] < args.mag_max)
                & (args.pixel_size * measured[:, 3] >= args.radius_min)
            )
            counts["measured_selection"] += int(selected.sum())
            frame = frame.loc[selected].reset_index(drop=True)
            if frame.empty:
                continue
            key = _keys(frame["case"], frame["input_index"])
            is_excluded = np.isin(key, excluded_keys, assume_unique=False)
            counts["excluded_existing_identity"] += int(is_excluded.sum())
            frame = frame.loc[~is_excluded]
            counts["eligible_after_exclusion"] += len(frame)
            if len(frame):
                eligible_parts.append(frame)
            if batch_index % 100 == 0:
                print(
                    f"batch {batch_index}/{reader.num_record_batches}: "
                    f"eligible={counts['eligible_after_exclusion']:,}",
                    flush=True,
                )

    if not eligible_parts:
        raise RuntimeError("no eligible ConstGold rows remain after exclusion")
    eligible = pd.concat(eligible_parts, ignore_index=True)
    if len(eligible) < args.sample_size:
        raise RuntimeError(
            f"only {len(eligible):,} disjoint rows are eligible for {args.sample_size:,} draws"
        )
    rng = np.random.default_rng(args.seed)
    selected = rng.choice(len(eligible), size=args.sample_size, replace=False)
    chosen = eligible.iloc[selected].reset_index(drop=True)
    output = pd.DataFrame(
        {
            "case": chosen["case"].to_numpy(np.int64),
            "input_index": chosen["input_index"].to_numpy(np.int64),
            "measured_ngmix_g1": chosen["measured_e1_plus"].to_numpy(float),
            "measured_ngmix_g2": chosen["measured_e2_plus"].to_numpy(float),
            "measured_mag_auto": chosen["measured_mag_auto_plus"].to_numpy(float),
            "measured_log_flux_radius": np.log(
                chosen["measured_flux_radius_plus"].to_numpy(float)
            ),
        }
    )
    output_keys = _keys(output["case"], output["input_index"])
    overlap = int(np.isin(output_keys, excluded_keys).sum())
    if overlap:
        raise RuntimeError(f"new sample overlaps the excluded sample on {overlap} identities")
    if len(np.unique(output_keys)) != len(output):
        raise RuntimeError("new sample contains duplicate identities")

    args.output.mkdir(parents=True)
    sample_path = args.output / "constgold_plus_selected_sample_100k.parquet"
    output.to_parquet(sample_path, index=False)
    manifest = {
        "catalogue": "constgold",
        "leg": "plus",
        "sample_size": int(len(output)),
        "seed": int(args.seed),
        "case_range": [int(args.min_case), int(args.max_case)],
        "source_selection": (
            "18<r_input_p<25.8; 0.5<Re_input_p<1.5; "
            "(0<distance<5 or not neighbored)"
        ),
        "measured_selection": (
            f"|e|<{args.shape_max}; measured_mag<{args.mag_max}; "
            f"measured_radius>={args.radius_min} arcsec"
        ),
        "excluded_sample": str(args.exclude_sample),
        "excluded_sample_sha256": _sha256(args.exclude_sample),
        "overlap_with_excluded_sample": overlap,
        "counts": counts,
        "output": str(sample_path),
        "output_sha256": _sha256(sample_path),
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
