"""Validate and concatenate case-sharded per-object response dumps."""
from __future__ import annotations

import argparse
import os

import numpy as np
import pyarrow as pa
import pyarrow.feather as pf
import pyarrow.ipc as ipc


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shard-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--min-case", type=int, default=40)
    ap.add_argument("--max-case", type=int, default=140)
    ap.add_argument("--cases-per-shard", type=int, default=10)
    args = ap.parse_args()

    if args.max_case <= args.min_case or args.cases_per_shard <= 0:
        raise SystemExit("invalid case range or shard width")
    os.makedirs(args.output_dir, exist_ok=True)

    for seed in args.seeds:
        bounds = [
            (lo, min(lo + args.cases_per_shard, args.max_case))
            for lo in range(args.min_case, args.max_case, args.cases_per_shard)
        ]
        paths = [
            os.path.join(args.shard_dir, f"{args.tag}_perobj_s{seed}_c{lo:03d}-{hi:03d}.feather")
            for lo, hi in bounds
        ]
        missing = [p for p in paths if not os.path.exists(p)]
        if missing:
            raise SystemExit(f"seed {seed}: missing {len(missing)} shards; first={missing[0]}")

        out = os.path.join(args.output_dir, f"{args.tag}_perobj_s{seed}.feather")
        if os.path.exists(out):
            raise SystemExit(f"refusing to overwrite {out}")
        tmp = f"{out}.tmp.{os.getpid()}"
        writer = None
        schema = None
        total = 0
        seen_cases = []
        try:
            for (lo, hi), path in zip(bounds, paths):
                with ipc.open_file(pa.memory_map(path)) as reader:
                    if schema is None:
                        schema = reader.schema
                        writer = ipc.new_file(
                            tmp, schema, options=ipc.IpcWriteOptions(compression="lz4")
                        )
                    elif not reader.schema.equals(schema):
                        raise RuntimeError(f"schema mismatch in {path}")
                    shard_cases = set()
                    shard_rows = 0
                    for bi in range(reader.num_record_batches):
                        batch = reader.get_batch(bi)
                        cases = batch.column(schema.get_field_index("case")).to_numpy()
                        if len(cases) and (int(cases.min()) < lo or int(cases.max()) >= hi):
                            raise RuntimeError(f"out-of-range case in {path}")
                        shard_cases.update(np.unique(cases).astype(int).tolist())
                        writer.write_batch(batch)
                        shard_rows += batch.num_rows
                    expected = set(range(lo, hi))
                    if shard_cases != expected:
                        raise RuntimeError(
                            f"{path}: case coverage {sorted(shard_cases)} != {sorted(expected)}"
                        )
                    seen_cases.extend(sorted(shard_cases))
                    total += shard_rows
                    print(f"seed {seed}: {lo:03d}-{hi:03d} rows={shard_rows:,}", flush=True)
            if seen_cases != list(range(args.min_case, args.max_case)):
                raise RuntimeError(f"seed {seed}: duplicated or missing cases")
            writer.close()
            writer = None
            os.replace(tmp, out)
        finally:
            if writer is not None:
                writer.close()
            if os.path.exists(tmp):
                os.unlink(tmp)
        check = pf.read_table(out, columns=["case"])
        if check.num_rows != total:
            raise RuntimeError(f"row-count check failed for {out}: {check.num_rows} != {total}")
        print(f"seed {seed}: wrote {out} rows={total:,}", flush=True)


if __name__ == "__main__":
    main()
