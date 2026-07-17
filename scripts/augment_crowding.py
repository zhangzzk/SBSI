"""Bake crowding columns into a det_meas catalogue, keyed by (case, input_index).

Adds: nbr_flux_near, nbr_flux_far and (when the flux lookup carries it) nbr_flux_max
from a shell-flux lookup, plus r_blend from a blend lookup. Missing (case,input_index)
-> 0 (isolated / not scored). Streaming: reads and writes one record batch at a time,
so it never holds the whole catalogue in memory.
"""
import argparse
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.feather as pf

KEYMUL = 1_000_003


def _sorted_lookup(path, cols):
    t = pf.read_table(path).to_pandas()
    k = t["case"].to_numpy(np.int64) * KEYMUL + t["input_index"].to_numpy(np.int64)
    o = np.argsort(k)
    return k[o], {c: t[c].to_numpy(float)[o] for c in cols}


def _map(keys, vals_by_col, batch_key):
    pos = np.clip(np.searchsorted(keys, batch_key), 0, len(keys) - 1)
    match = keys[pos] == batch_key
    return {c: np.where(match, v[pos], 0.0) for c, v in vals_by_col.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalogue", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--flux-lookup", default=None)
    ap.add_argument("--blend-lookup", default=None)
    ap.add_argument("--max-case", type=int, default=None)
    args = ap.parse_args()

    fk = fv = bk = bv = None
    flux_cols = ["nbr_flux_near", "nbr_flux_far"]
    if args.flux_lookup:
        # include nbr_flux_max (flux concentration) when the lookup carries it
        have = set(pf.read_table(args.flux_lookup, columns=None).schema.names)
        if "nbr_flux_max" in have:
            flux_cols = flux_cols + ["nbr_flux_max"]
        fk, fv = _sorted_lookup(args.flux_lookup, flux_cols)
        print(f"flux lookup: {len(fk):,} entries, cols={flux_cols}")
    if args.blend_lookup:
        bk, bv = _sorted_lookup(args.blend_lookup, ["R_blend"])
        print(f"blend lookup: {len(bk):,} entries")

    n_in = n_out = 0
    writer = None
    with ipc.open_file(args.catalogue) as r:
        for bi in range(r.num_record_batches):
            df = pa.Table.from_batches([r.get_batch(bi)]).to_pandas()
            n_in += len(df)
            if args.max_case is not None and "case" in df.columns:
                df = df[df["case"] <= args.max_case].reset_index(drop=True)
                if len(df) == 0:
                    continue
            key = df["case"].to_numpy(np.int64) * KEYMUL + df["input_index"].to_numpy(np.int64)
            if fk is not None:
                m = _map(fk, fv, key)
                df["nbr_flux_near"] = m["nbr_flux_near"]; df["nbr_flux_far"] = m["nbr_flux_far"]
                if "nbr_flux_max" in m:
                    df["nbr_flux_max"] = m["nbr_flux_max"]
            if bk is not None:
                df["r_blend"] = _map(bk, bv, key)["R_blend"]
            tbl = pa.Table.from_pandas(df, preserve_index=False)
            if writer is None:
                writer = ipc.new_file(args.output, tbl.schema)
            writer.write_table(tbl)
            n_out += len(df)
            if bi % 50 == 0:
                print(f"  batch {bi}: in={n_in:,} out={n_out:,}", flush=True)
    if writer is not None:
        writer.close()
    print(f"wrote {args.output}: {n_out:,} rows (from {n_in:,})", flush=True)


if __name__ == "__main__":
    main()
