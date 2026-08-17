"""Build the unique (case,input_index) union needed by scene-feature jobs."""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pyarrow as pa
import pyarrow.feather as pf
import pyarrow.ipc as ipc


KEYMUL = 1_000_003


def keys_from_catalogue(path: str) -> np.ndarray:
    parts = []
    with ipc.open_file(pa.memory_map(path)) as reader:
        for batch_index in range(reader.num_record_batches):
            table = pa.Table.from_batches([reader.get_batch(batch_index)]).select(
                ["case", "input_index"])
            case = table["case"].to_numpy().astype(np.int64)
            index = table["input_index"].to_numpy().astype(np.int64)
            parts.append(np.unique(case * KEYMUL + index))
    return np.unique(np.concatenate(parts))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--catalogues", nargs="+", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    if os.path.exists(args.output):
        raise FileExistsError(f"refusing to overwrite {args.output}")
    keys = []
    provenance = []
    for path in args.catalogues:
        value = keys_from_catalogue(path)
        keys.append(value)
        provenance.append({"path": os.path.abspath(path), "unique_keys": len(value)})
        print(f"{path}: {len(value):,} unique keys", flush=True)
    key = np.unique(np.concatenate(keys))
    case = key // KEYMUL
    index = key % KEYMUL
    if np.any(index < 0) or np.any(index >= KEYMUL):
        raise RuntimeError("input_index exceeds manifest key encoding")
    output = pa.table({"case": case.astype(np.int32), "input_index": index.astype(np.int64)})
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    pf.write_feather(output, args.output)
    meta = {
        "output": os.path.abspath(args.output), "rows": len(output),
        "case_lo": int(case.min()), "case_hi": int(case.max()),
        "n_cases": int(len(np.unique(case))), "inputs": provenance,
    }
    with open(os.path.splitext(args.output)[0] + ".json", "x", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"wrote {args.output}: {len(output):,} keys over {meta['n_cases']} cases", flush=True)


if __name__ == "__main__":
    main()
