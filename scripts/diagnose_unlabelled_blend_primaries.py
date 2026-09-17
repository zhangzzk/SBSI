"""Split the ConstGold R_blend by whether a primary can be labelled at all.

The ConstGold `R_blend` lookup sums emulator predictions for every fixed-g0
anchor primary.  The response catalogue, which supplies the only measured
blend-response labels, contains no valid pair for a few percent of those
primaries.  Every emulator validation to date is therefore silent about them,
while their predictions still enter `m` at full weight.

This diagnostic partitions the existing per-object ConstGold `R_blend` lookup
into primaries that do and do not have at least one valid labelled pair, and
reports what each partition contributes to the cohort mean.  It evaluates no
emulator and fits nothing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc

from scripts.build_fixed_g0_response_profile_predictions import (
    BLEND_MEASUREMENT_COLUMNS,
    BLEND_RESPONSE_COLUMNS,
)

# BLEND_RESPONSE_COLUMNS already carries "case" and "input_index"; Arrow's
# select() would otherwise hand back duplicate columns.
SCAN_COLUMNS = tuple(
    dict.fromkeys(
        ("case", "input_index", *BLEND_RESPONSE_COLUMNS, *BLEND_MEASUREMENT_COLUMNS)
    )
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def valid_pairs(frame: pd.DataFrame) -> np.ndarray:
    """The finite, physical two-leg response-label validity rule."""
    finite = np.isfinite(
        frame[list(BLEND_RESPONSE_COLUMNS)].to_numpy(np.float64, copy=False)
    ).all(axis=1)
    measured = frame[list(BLEND_MEASUREMENT_COLUMNS)].to_numpy(np.float64, copy=False)
    finite &= np.square(measured[:, :2]).sum(axis=1) < 1.0
    finite &= np.square(measured[:, 2:]).sum(axis=1) < 1.0
    return finite


def labelled_primary_keys(path: Path, cases: set[int]) -> tuple[np.ndarray, int]:
    """Keys of primaries holding at least one valid labelled pair."""
    chunks: list[np.ndarray] = []
    scanned = 0
    with ipc.open_file(path) as reader:
        for number in range(reader.num_record_batches):
            batch = pa.Table.from_batches([reader.get_batch(number)])
            frame = batch.select(list(SCAN_COLUMNS)).to_pandas()
            frame = frame.loc[frame["case"].isin(cases)]
            if frame.empty:
                continue
            frame = frame.loc[valid_pairs(frame)]
            scanned += len(frame)
            chunks.append(
                pack_keys(
                    frame["case"].to_numpy(np.int64),
                    frame["input_index"].to_numpy(np.int64),
                )
            )
    if not chunks:
        return np.empty(0, dtype=np.int64), scanned
    return np.unique(np.concatenate(chunks)), scanned


def pack_keys(case: np.ndarray, index: np.ndarray) -> np.ndarray:
    return (np.asarray(case, dtype=np.int64) << np.int64(32)) | np.asarray(
        index, dtype=np.int64
    )


def partition_lookup(lookup: pd.DataFrame, labelled: np.ndarray) -> pd.DataFrame:
    keys = pack_keys(
        lookup["case"].to_numpy(np.int64), lookup["input_index"].to_numpy(np.int64)
    )
    frame = lookup.copy()
    frame["has_label"] = np.isin(keys, labelled, assume_unique=False)
    return frame


def case_partition_means(frame: pd.DataFrame) -> dict:
    out = {}
    for case, part in frame.groupby("case", sort=True):
        entry = {"n_all": int(len(part)), "sum_all": float(part["R_blend"].sum())}
        for name, mask in (
            ("labelled", part["has_label"]),
            ("unlabelled", ~part["has_label"]),
        ):
            sub = part.loc[mask]
            entry[f"n_{name}"] = int(len(sub))
            entry[f"sum_{name}"] = float(sub["R_blend"].sum())
        out[int(case)] = entry
    return out


def pool(entries: list[dict]) -> dict:
    totals = {}
    for entry in entries:
        for key, value in entry.items():
            totals[key] = totals.get(key, 0.0) + value
    summary = {}
    for name in ("all", "labelled", "unlabelled"):
        n = totals.get(f"n_{name}", 0.0)
        summary[name] = {
            "n": int(n),
            "mean_R_blend": (totals[f"sum_{name}"] / n) if n > 0 else None,
            "row_fraction": (n / totals["n_all"]) if totals["n_all"] > 0 else None,
            "contribution_to_cohort_mean": (
                totals[f"sum_{name}"] / totals["n_all"]
            )
            if totals["n_all"] > 0
            else None,
        }
    return summary


def bootstrap(per_case: dict, n_boot: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    entries = list(per_case.values())
    draws = {name: [] for name in ("all", "labelled", "unlabelled")}
    for _ in range(n_boot):
        pick = rng.integers(0, len(entries), size=len(entries))
        summary = pool([entries[i] for i in pick])
        for name in draws:
            value = summary[name]["mean_R_blend"]
            if value is not None:
                draws[name].append(value)
    return {
        name: (float(np.std(values, ddof=1)) if len(values) > 1 else None)
        for name, values in draws.items()
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rblend-lookup", type=Path, required=True)
    parser.add_argument("--response-catalogue", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n-boot", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260915)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("run under the scheduler; SLURM_JOB_ID is unset")
    if args.output.exists():
        raise RuntimeError(f"refusing to overwrite {args.output}")

    lookup = pd.read_feather(args.rblend_lookup)
    if lookup.duplicated(["case", "input_index"]).any():
        raise RuntimeError("R_blend lookup contains duplicate primary keys")
    cases = set(int(c) for c in lookup["case"].unique())
    print(f"UNLABELLED_BLEND_SCAN cases={len(cases)} rows={len(lookup)}", flush=True)

    labelled, scanned_pairs = labelled_primary_keys(args.response_catalogue, cases)
    print(
        f"UNLABELLED_BLEND_LABELS primaries={labelled.size} valid_pairs={scanned_pairs}",
        flush=True,
    )

    frame = partition_lookup(lookup, labelled)
    per_case = case_partition_means(frame)
    summary = pool(list(per_case.values()))
    errors = bootstrap(per_case, args.n_boot, args.bootstrap_seed)
    for name, entry in summary.items():
        entry["standard_error_mean_R_blend"] = errors.get(name)

    result = {
        "format_version": 1,
        "purpose": (
            "Partition the ConstGold R_blend lookup by whether the primary has any "
            "valid labelled pair, to expose the part of R_blend that no emulator "
            "validation can reach."
        ),
        "cases": sorted(cases),
        "inputs": {
            "rblend_lookup": str(args.rblend_lookup),
            "rblend_lookup_sha256": file_sha256(args.rblend_lookup),
            "response_catalogue": str(args.response_catalogue),
        },
        "bootstrap": {"n_boot": args.n_boot, "seed": args.bootstrap_seed},
        "summary": summary,
        "case_reports": per_case,
        "limitations": [
            "Partitions one additive term; evaluates no emulator and reports no m.",
            "'Unlabelled' means the response catalogue holds no valid pair for that "
            "primary; it does not mean the primary has no neighbours.",
            "Case bootstrap over the cases present in the lookup.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    print(f"UNLABELLED_BLEND_COMPLETE output={args.output}", flush=True)


if __name__ == "__main__":
    main()
