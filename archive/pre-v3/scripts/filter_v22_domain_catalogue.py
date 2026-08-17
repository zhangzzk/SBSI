"""Stream an IPC catalogue into the exact V2.2 true-primary box.

This is a storage/compute preparation step only.  The trainer still reapplies
its normal selection and finiteness cuts; filtering here merely prevents us
from computing expensive scene features for rows that V2.2 can never use.
"""
from __future__ import annotations

import argparse
import json
import os

import pyarrow as pa
import pyarrow.ipc as ipc


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--primary-mag-max", type=float, default=25.8)
    ap.add_argument("--primary-re-min", type=float, default=0.5)
    ap.add_argument("--min-case", type=int)
    ap.add_argument("--max-case", type=int,
                    help="exclusive upper case bound")
    args = ap.parse_args()
    if os.path.exists(args.output):
        raise FileExistsError(f"refusing to overwrite {args.output}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    writer = None
    rows_in = rows_out = 0
    case_values: set[int] = set()
    with ipc.open_file(pa.memory_map(args.input)) as reader:
        required = {"r_input_p", "Re_input_p", "case"}
        missing = required - set(reader.schema.names)
        if missing:
            raise KeyError(f"input is missing {sorted(missing)}")
        for batch_index in range(reader.num_record_batches):
            table = pa.Table.from_batches([reader.get_batch(batch_index)])
            frame = table.to_pandas()
            rows_in += len(frame)
            keep = ((frame["r_input_p"].to_numpy(float) < args.primary_mag_max)
                    & (frame["Re_input_p"].to_numpy(float) > args.primary_re_min))
            if args.min_case is not None:
                keep &= frame["case"].to_numpy(int) >= args.min_case
            if args.max_case is not None:
                keep &= frame["case"].to_numpy(int) < args.max_case
            out = pa.Table.from_pandas(frame.loc[keep], preserve_index=False)
            if len(out):
                if writer is None:
                    writer = ipc.new_file(args.output, out.schema)
                writer.write_table(out)
                rows_out += len(out)
                case_values.update(out["case"].to_numpy().astype(int).tolist())
            if batch_index % 25 == 0:
                print(f"batch {batch_index}: input={rows_in:,}, kept={rows_out:,}", flush=True)
    if writer is None:
        raise RuntimeError("no rows survived the requested domain/case filter")
    writer.close()
    source_meta_path = os.path.splitext(args.input)[0] + ".json"
    source_meta = {}
    if os.path.exists(source_meta_path):
        with open(source_meta_path, encoding="utf-8") as handle:
            source_meta = json.load(handle)
    filtering = {
        "input": os.path.abspath(args.input), "output": os.path.abspath(args.output),
        "rows_input": rows_in, "rows_output": rows_out,
        "primary_mag_max": args.primary_mag_max, "primary_re_min": args.primary_re_min,
        "min_case": args.min_case, "max_case_exclusive": args.max_case,
        "n_cases": len(case_values),
    }
    meta = {**source_meta, **filtering, "source_meta": source_meta_path,
            "domain_filter": filtering}
    with open(os.path.splitext(args.output)[0] + ".json", "x", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"wrote {args.output}: {rows_out:,}/{rows_in:,} rows", flush=True)


if __name__ == "__main__":
    main()
