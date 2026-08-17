"""Combine full-scene lookups into compact top-two and third-plus feature lookups."""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pyarrow as pa
import pyarrow.feather as pf


SOURCE_ALL = "logflux_all_0_10"
SOURCE_THIRD = "logflux_thirdplus_0_10"
TARGET_TOP2 = "nbr_flux_top2"
TARGET_THIRD = "nbr_flux_thirdplus"


def top2_log_ratio(log_all: np.ndarray, log_third: np.ndarray) -> np.ndarray:
    """Return log10(1 + F_top2/F_primary) from all- and third-plus log ratios."""
    all_ratio = np.expm1(np.asarray(log_all, float) * np.log(10.0))
    third_ratio = np.expm1(np.asarray(log_third, float) * np.log(10.0))
    return np.log10(1.0 + np.maximum(all_ratio - third_ratio, 0.0))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    if os.path.exists(args.output):
        raise SystemExit(f"REFUSING to overwrite {args.output}")

    parts = []
    seen_cases: set[int] = set()
    provenance = []
    for path in args.inputs:
        table = pf.read_table(
            path, columns=["case", "input_index", SOURCE_ALL, SOURCE_THIRD],
        )
        cases = set(np.unique(table["case"].to_numpy()).astype(int).tolist())
        overlap = cases & seen_cases
        if overlap:
            raise RuntimeError(f"input case ranges overlap at {sorted(overlap)}")
        seen_cases |= cases
        top2 = top2_log_ratio(
            table[SOURCE_ALL].to_numpy(), table[SOURCE_THIRD].to_numpy(),
        )
        out = pa.table({
            "case": table["case"],
            "input_index": table["input_index"],
            TARGET_TOP2: top2,
            TARGET_THIRD: table[SOURCE_THIRD],
        })
        parts.append(out)
        provenance.append({"path": os.path.abspath(path), "rows": len(table),
                           "case_lo": min(cases), "case_hi": max(cases),
                           "n_cases": len(cases)})
        print(f"{path}: {len(table):,} rows, cases {min(cases)}--{max(cases)}", flush=True)

    out = pa.concat_tables(parts)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    pf.write_feather(out, args.output)
    meta = {
        "output": os.path.abspath(args.output), "rows": len(out),
        "case_lo": min(seen_cases), "case_hi": max(seen_cases),
        "n_cases": len(seen_cases), "features": [TARGET_TOP2, TARGET_THIRD],
        "definitions": {
            TARGET_TOP2: "log10(1 + flux of two brightest neighbours within 10 arcsec / primary flux)",
            TARGET_THIRD: "log10(1 + flux of third-brightest and later neighbours within 10 arcsec / primary flux)",
        },
        "inputs": provenance,
    }
    with open(os.path.splitext(args.output)[0] + ".json", "w") as handle:
        json.dump(meta, handle, indent=1, sort_keys=True)
    print(f"wrote {args.output}: {len(out):,} rows over {len(seen_cases)} cases", flush=True)


if __name__ == "__main__":
    main()
