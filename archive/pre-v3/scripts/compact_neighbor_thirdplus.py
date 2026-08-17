"""Combine full-scene lookups into a compact third-plus feature lookup."""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pyarrow as pa
import pyarrow.feather as pf


SOURCE = "logflux_thirdplus_0_10"
TARGET = "nbr_flux_thirdplus"


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
        table = pf.read_table(path, columns=["case", "input_index", SOURCE])
        cases = set(np.unique(table["case"].to_numpy()).astype(int).tolist())
        overlap = cases & seen_cases
        if overlap:
            raise RuntimeError(f"input case ranges overlap at {sorted(overlap)}")
        seen_cases |= cases
        table = table.rename_columns(["case", "input_index", TARGET])
        parts.append(table)
        provenance.append({"path": os.path.abspath(path), "rows": len(table),
                           "case_lo": min(cases), "case_hi": max(cases),
                           "n_cases": len(cases)})
        print(f"{path}: {len(table):,} rows, cases {min(cases)}--{max(cases)}", flush=True)

    out = pa.concat_tables(parts)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    pf.write_feather(out, args.output)
    meta = {"output": os.path.abspath(args.output), "rows": len(out),
            "case_lo": min(seen_cases), "case_hi": max(seen_cases),
            "n_cases": len(seen_cases), "feature": TARGET,
            "definition": "log10(1 + flux of third-brightest and later neighbours within 10 arcsec / primary flux)",
            "inputs": provenance}
    with open(os.path.splitext(args.output)[0] + ".json", "w") as handle:
        json.dump(meta, handle, indent=1, sort_keys=True)
    print(f"wrote {args.output}: {len(out):,} rows over {len(seen_cases)} cases", flush=True)


if __name__ == "__main__":
    main()
