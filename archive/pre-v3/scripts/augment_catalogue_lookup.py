"""Stream one or more strict per-(case,input_index) lookup columns into an IPC catalogue."""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pyarrow as pa
import pyarrow.feather as pf
import pyarrow.ipc as ipc


KEYMUL = 1_000_003


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--catalogue", required=True)
    ap.add_argument("--lookup", required=True)
    ap.add_argument("--columns", nargs="+", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--drop-unmatched", action="store_true",
                    help="inner-join semantics for model/emulator features: explicitly drop source "
                         "rows absent from the lookup instead of zero-filling them")
    ap.add_argument("--repair-meta-only", action="store_true",
                    help="repair an existing output sidecar by merging the source sidecar; "
                         "the output feather is verified and not rewritten")
    args = ap.parse_args()
    if os.path.exists(args.output) and not args.repair_meta_only:
        raise SystemExit(f"REFUSING to overwrite {args.output}")

    source_meta_path = os.path.splitext(args.catalogue)[0] + ".json"
    source_meta = {}
    if os.path.exists(source_meta_path):
        with open(source_meta_path) as handle:
            source_meta = json.load(handle)
    output_meta_path = os.path.splitext(args.output)[0] + ".json"
    if args.repair_meta_only:
        if not os.path.exists(args.output) or not os.path.exists(output_meta_path):
            raise SystemExit("--repair-meta-only requires the existing output feather and sidecar")
        with open(output_meta_path) as handle:
            augmentation = json.load(handle)
        source_rows = pf.read_table(args.catalogue, columns=[]).num_rows
        output_rows = pf.read_table(args.output, columns=[]).num_rows
        if source_rows != output_rows or output_rows != int(augmentation["rows"]):
            raise RuntimeError(
                f"cannot repair inconsistent artifacts: source={source_rows:,}, "
                f"output={output_rows:,}, sidecar={augmentation['rows']:,}"
            )
        meta = {**source_meta, **augmentation, "source_meta": source_meta_path,
                "augmentation": augmentation}
        with open(output_meta_path, "w") as handle:
            json.dump(meta, handle, indent=1, sort_keys=True)
        print(f"repaired {output_meta_path}: preserved source keys {sorted(source_meta)}")
        return

    lookup = pf.read_table(args.lookup, columns=["case", "input_index", *args.columns])
    lk = (lookup["case"].to_numpy().astype(np.int64) * KEYMUL
          + lookup["input_index"].to_numpy().astype(np.int64))
    order = np.argsort(lk)
    lk = lk[order]
    if np.any(lk[1:] == lk[:-1]):
        raise RuntimeError("lookup keys are not unique")
    values = {name: lookup[name].to_numpy()[order] for name in args.columns}
    print(f"lookup: {len(lk):,} unique keys; columns={args.columns}", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    writer = None
    input_rows = rows = matched = dropped_unmatched = 0
    with ipc.open_file(pa.memory_map(args.catalogue)) as reader:
        for batch_index in range(reader.num_record_batches):
            frame = pa.Table.from_batches([reader.get_batch(batch_index)]).to_pandas()
            input_rows += len(frame)
            key = (frame["case"].to_numpy(np.int64) * KEYMUL
                   + frame["input_index"].to_numpy(np.int64))
            pos = np.clip(np.searchsorted(lk, key), 0, len(lk) - 1)
            match = lk[pos] == key
            if not np.all(match):
                missing_rows = frame.loc[~match, ["case", "input_index"]].head(10).to_dict("records")
                if not args.drop_unmatched:
                    raise RuntimeError(
                        f"batch {batch_index}: lookup misses {int((~match).sum()):,}/{len(match):,} "
                        f"rows; first missing keys={missing_rows}; zero-filling scene context is "
                        "forbidden"
                    )
                dropped_unmatched += int((~match).sum())
                print(f"batch {batch_index}: dropping {int((~match).sum()):,} unmatched rows; "
                      f"first keys={missing_rows}", flush=True)
                frame = frame.loc[match].reset_index(drop=True)
                pos = pos[match]
            for name, value in values.items():
                frame[name] = value[pos]
            out = pa.Table.from_pandas(frame, preserve_index=False)
            if writer is None:
                writer = ipc.new_file(args.output, out.schema)
            writer.write_table(out)
            rows += len(frame)
            matched += int(match.sum())
            if batch_index % 25 == 0:
                print(f"batch {batch_index}: rows={rows:,}", flush=True)
    if writer is None:
        raise RuntimeError("input catalogue had no record batches")
    writer.close()
    augmentation = {"output": os.path.abspath(args.output),
                    "catalogue": os.path.abspath(args.catalogue),
                    "lookup": os.path.abspath(args.lookup), "columns": args.columns,
                    "input_rows": input_rows, "rows": rows, "matched": matched,
                    "dropped_unmatched": dropped_unmatched,
                    "match_fraction": matched / input_rows,
                    "drop_unmatched": args.drop_unmatched}
    meta = {**source_meta, **augmentation, "source_meta": source_meta_path,
            "augmentation": augmentation}
    with open(output_meta_path, "w") as handle:
        json.dump(meta, handle, indent=1, sort_keys=True)
    print(f"wrote {args.output}: {rows:,}/{input_rows:,} rows, "
          f"lookup match {matched/input_rows:.6%}; dropped={dropped_unmatched:,}", flush=True)


if __name__ == "__main__":
    main()
