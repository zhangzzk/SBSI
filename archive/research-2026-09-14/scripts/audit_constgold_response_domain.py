#!/usr/bin/env python
"""Verify that ConstGold response support exactly matches two model domains."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.feather as pf
import torch


KEYS = ("case", "input_index")


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _bounds(value, *, label: str) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{label} is not a two-element bound: {value!r}")
    return float(value[0]), float(value[1])


def _require_equal_bounds(
    actual: tuple[float, float], expected: tuple[float, float], *, label: str
) -> None:
    if actual != expected:
        raise ValueError(f"{label} mismatch: expected {expected}, found {actual}")


def _checkpoint_metadata(path: Path) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not isinstance(payload.get("metadata"), dict):
        raise ValueError(f"measurement checkpoint lacks metadata: {path}")
    return payload["metadata"]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--constgold-root", type=Path, required=True)
    parser.add_argument("--rblend-lookup", type=Path, action="append", required=True)
    parser.add_argument("--measurement-model", type=Path, required=True)
    parser.add_argument("--emulator-model", type=Path, required=True)
    parser.add_argument("--emulator-metadata", type=Path, required=True)
    parser.add_argument("--expected-flow-sha256", required=True)
    parser.add_argument("--expected-emulator-sha256", required=True)
    parser.add_argument("--case", type=int, action="append", required=True)
    parser.add_argument("--primary-mag-min", type=float, required=True)
    parser.add_argument("--primary-mag-max", type=float, required=True)
    parser.add_argument("--primary-re-min", type=float, required=True)
    parser.add_argument("--primary-re-max", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    cases = tuple(dict.fromkeys(args.case))
    if not cases:
        raise ValueError("no ConstGold cases requested")
    expected_mag = (args.primary_mag_min, args.primary_mag_max)
    expected_re = (args.primary_re_min, args.primary_re_max)

    flow_sha = _sha256(args.measurement_model)
    emulator_sha = _sha256(args.emulator_model)
    if flow_sha != args.expected_flow_sha256:
        raise ValueError(
            f"flow hash mismatch: expected {args.expected_flow_sha256}, found {flow_sha}"
        )
    if emulator_sha != args.expected_emulator_sha256:
        raise ValueError(
            "emulator hash mismatch: expected "
            f"{args.expected_emulator_sha256}, found {emulator_sha}"
        )

    flow_metadata = _checkpoint_metadata(args.measurement_model)
    flow_cuts = flow_metadata.get("selection_cuts")
    if not isinstance(flow_cuts, list) or len(flow_cuts) < 4:
        raise ValueError("flow metadata lacks the primary selection-cut domain")
    _require_equal_bounds(
        _bounds(flow_cuts[1], label="flow primary magnitude"),
        expected_mag,
        label="flow primary magnitude",
    )
    _require_equal_bounds(
        _bounds(flow_cuts[3], label="flow primary size"),
        expected_re,
        label="flow primary size",
    )

    emulator_metadata = json.loads(args.emulator_metadata.read_text())
    regression = emulator_metadata.get("tasks", {}).get("regression", {})
    emulator_cuts = regression.get("cuts")
    if not isinstance(emulator_cuts, list) or len(emulator_cuts) < 4:
        raise ValueError("emulator metadata lacks the regression-cut domain")
    _require_equal_bounds(
        _bounds(emulator_cuts[1], label="emulator primary magnitude"),
        expected_mag,
        label="emulator primary magnitude",
    )
    _require_equal_bounds(
        _bounds(emulator_cuts[3], label="emulator primary size"),
        expected_re,
        label="emulator primary size",
    )

    lookup_by_case: dict[int, pd.DataFrame] = {}
    lookup_artifacts = []
    for path in args.rblend_lookup:
        frame = pf.read_table(path, columns=[*KEYS, "R_blend"]).to_pandas()
        if frame.duplicated(list(KEYS)).any():
            raise ValueError(f"duplicate response keys in {path}")
        if not np.isfinite(frame["R_blend"]).all():
            raise ValueError(f"non-finite response values in {path}")
        lookup_artifacts.append(
            {"path": str(path), "sha256": _sha256(path), "rows": int(len(frame))}
        )
        for case, part in frame.groupby("case", sort=False):
            case = int(case)
            if case in lookup_by_case:
                raise ValueError(f"case {case} occurs in more than one response shard")
            lookup_by_case[case] = part.loc[:, ["input_index", "R_blend"]].copy()
    if set(lookup_by_case) != set(cases):
        raise ValueError(
            "response-shard case mismatch: "
            f"missing={sorted(set(cases) - set(lookup_by_case))}, "
            f"unexpected={sorted(set(lookup_by_case) - set(cases))}"
        )

    summaries = []
    total_rows = 0
    total_new_band = 0
    for case in cases:
        truth_path = (
            args.constgold_root
            / f"case{case}_0.02"
            / "real0/catalogues/input/gals_info_tile180.0_-0.5.feather"
        )
        truth = pf.read_table(
            truth_path, columns=["index_input", "r_input", "Re_input"]
        ).to_pandas()
        if truth["index_input"].duplicated().any():
            raise ValueError(f"duplicate truth identities in case {case}")
        finite = np.isfinite(truth[["r_input", "Re_input"]]).all(axis=1)
        in_domain = (
            finite
            & (truth["r_input"] > expected_mag[0])
            & (truth["r_input"] < expected_mag[1])
            & (truth["Re_input"] > expected_re[0])
            & (truth["Re_input"] < expected_re[1])
        )
        expected_ids = np.sort(truth.loc[in_domain, "index_input"].to_numpy(np.int64))
        lookup = lookup_by_case[case]
        actual_ids = np.sort(lookup["input_index"].to_numpy(np.int64))
        if not np.array_equal(expected_ids, actual_ids):
            missing = np.setdiff1d(expected_ids, actual_ids, assume_unique=True)
            extra = np.setdiff1d(actual_ids, expected_ids, assume_unique=True)
            raise ValueError(
                f"case {case} support mismatch: expected={len(expected_ids)}, "
                f"lookup={len(actual_ids)}, missing={len(missing)}, extra={len(extra)}"
            )
        new_band = int(
            (
                in_domain
                & (truth["Re_input"] < 0.5)
            ).sum()
        )
        total_rows += len(actual_ids)
        total_new_band += new_band
        summaries.append(
            {
                "case": case,
                "truth_rows": int(len(truth)),
                "support_rows": int(len(actual_ids)),
                "support_rows_re_lt_0p5": new_band,
                "mean_rblend": float(lookup["R_blend"].mean()),
            }
        )

    report = {
        "status": "pass",
        "domain_definition": "strict lower and upper bounds",
        "primary_domain": {
            "r_input": list(expected_mag),
            "Re_input": list(expected_re),
        },
        "cases": list(cases),
        "case_count": len(cases),
        "support_rows": total_rows,
        "support_rows_re_lt_0p5": total_new_band,
        "support_rows_re_lt_0p5_fraction": total_new_band / total_rows,
        "flow": {
            "path": str(args.measurement_model),
            "sha256": flow_sha,
            "selection_cuts": flow_cuts,
            "condition_features": flow_metadata.get("condition_features"),
        },
        "emulator": {
            "path": str(args.emulator_model),
            "sha256": emulator_sha,
            "metadata": str(args.emulator_metadata),
            "metadata_sha256": _sha256(args.emulator_metadata),
            "regression_cuts": emulator_cuts,
            "k": regression.get("k"),
            "r_max": regression.get("r_max"),
        },
        "rblend_lookups": lookup_artifacts,
        "per_case": summaries,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in (
        "status", "primary_domain", "case_count", "support_rows",
        "support_rows_re_lt_0p5", "support_rows_re_lt_0p5_fraction",
    )}, indent=2))


if __name__ == "__main__":
    main()
