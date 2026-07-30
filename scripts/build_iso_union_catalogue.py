"""Add the isolated primaries back into the response-target catalogue, with zero neighbour flux.

THE PROBLEM (WORKLOG 2026-07-30b RESULT 10/11). The pin's target is built on
`det_meas_crowd_g0.05_val_full` = the np7 NEAREST-PAIR build plus crowd columns. np7 emits one row per
(primary, nearest neighbour within 7"), so **a primary with no neighbour inside 7" has no row at all**.
The result is a target catalogue that is 99.97% neighboured, supervising a flow whose deliverable
population is ~24% isolated. The pin has never been shown an isolated galaxy.

THE FIX, AND WHY IT IS THIS SMALL. The dropped primaries are not missing data -- they are present in
the ruler catalogue `det_meas_ngmix_g0.05_val`, which was built with the nearest-neighbour attach and
therefore carries one row per primary whether or not it has a close companion. So this script does not
simulate or measure anything new; it takes the target catalogue and UNIONS IN the primaries the np7
build dropped:

    output = det_meas_crowd_g0.05_val_full            (unchanged, byte-for-byte)
           + the ruler's rows for primaries absent from it, with
                nbr_flux_near = nbr_flux_far = r_blend = 0

Zero is not an approximation chosen for convenience -- it is what `augment_crowding.py` already
assigns to any (case, input_index) missing from the crowd lookups, so these rows get exactly the
crowding values the existing pipeline would have given them. They land in the lowest-crowding bin,
which is where an isolated galaxy belongs.

Deliberately conservative: the kept rows are passed through UNTOUCHED, so the only difference between
the old target and the new one is the added population. That keeps the change interpretable -- if the
rebuilt target moves the pin, it moved because isolated galaxies entered it, not because the existing
rows were re-annotated under a different neighbour convention.

Selection on the added rows matches what the np7 build applied to the kept ones (`--flow-only`):
detected, with a finite ngmix shape. Rows failing that would not have survived np7 either.

FIREWALL: half-shear catalogues only. No constgold is read.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.ipc as ipc

KEY = ["case", "input_index"]
KEYMUL = 1_000_003
ZERO_COLS = ["nbr_flux_near", "nbr_flux_far", "nbr_flux_max", "r_blend"]


def keyof(df):
    return df["case"].to_numpy(np.int64) * KEYMUL + df["input_index"].to_numpy(np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crowd-cat", required=True, help="the current target catalogue (np7 + crowd cols)")
    ap.add_argument("--ruler-cat", required=True, help="the all-primary catalogue to take missing rows from")
    ap.add_argument("--output", required=True)
    ap.add_argument("--max-case", type=int, default=None)
    args = ap.parse_args()

    if os.path.exists(args.output):
        raise SystemExit(f"REFUSING to overwrite existing {args.output}")

    # ---- pass 1: which primaries does the target already have, and what is its schema?
    src = ipc.open_file(args.crowd_cat)
    schema = src.schema
    names = list(schema.names)
    print(f"target catalogue: {os.path.basename(args.crowd_cat)}  "
          f"{src.num_record_batches} batches, {len(names)} cols")
    have_zero = [c for c in ZERO_COLS if c in names]
    print(f"  crowd columns present, will be set to 0 on the added rows: {have_zero}")

    keys = []
    for bi in range(src.num_record_batches):
        b = pa.Table.from_batches([src.get_batch(bi)]).select(KEY).to_pandas()
        keys.append(keyof(b))
    kept_keys = np.unique(np.concatenate(keys))
    del keys
    print(f"  primaries already in the target: {len(kept_keys):,}")

    # ---- pass 2: stream the ruler, keep only primaries the target lacks
    need = [c for c in names if c not in ZERO_COLS]
    rd = ds.dataset(args.ruler_cat, format="feather")
    missing_from_ruler = [c for c in need if c not in rd.schema.names]
    if missing_from_ruler:
        raise SystemExit(f"ruler lacks columns the target has: {missing_from_ruler}")

    filt = None if args.max_case is None else (ds.field("case") <= args.max_case)
    chunks = []
    n_seen = n_kept = 0
    scanner = rd.scanner(columns=need, filter=filt, batch_size=2_000_000)
    for bi, batch in enumerate(scanner.to_batches()):
        d = pa.Table.from_batches([batch]).to_pandas()
        n_seen += len(d)
        k = keyof(d)
        # not already in the target, detected, and with a usable shape -- the np7 --flow-only rule
        m = ~np.isin(k, kept_keys, assume_unique=False)
        m &= d["detected"].astype(bool).to_numpy()
        m &= np.isfinite(d["measured_ngmix_g1"].to_numpy(float))
        m &= np.isfinite(d["measured_ngmix_g2"].to_numpy(float))
        if m.any():
            chunks.append(d[m].reset_index(drop=True))
            n_kept += int(m.sum())
        if bi % 20 == 0:
            print(f"  ruler batch {bi}: seen={n_seen:,} candidate rows={n_kept:,}", flush=True)
        del d
    add = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=need)
    del chunks
    print(f"  ruler rows scanned: {n_seen:,}; candidate added rows: {len(add):,}")

    # an all-pairs ruler repeats a primary once per neighbour -- keep the NEAREST row per primary
    if len(add) and "distance" in add.columns:
        add = add.sort_values("distance", kind="stable", na_position="last")
    add = add.drop_duplicates(KEY, keep="first").reset_index(drop=True)
    print(f"  after one-row-per-primary: {len(add):,} added primaries")

    for c in have_zero:
        add[c] = 0.0
    add = add[names]
    for c in names:                                  # match the target's dtypes exactly
        want = schema.field(c).type.to_pandas_dtype()
        try:
            add[c] = add[c].astype(want, copy=False)
        except (TypeError, ValueError):
            pass

    # ---- write: the target's batches verbatim, then the added rows
    writer = ipc.new_file(args.output, schema)
    n_out = 0
    for bi in range(src.num_record_batches):
        t = pa.Table.from_batches([src.get_batch(bi)])
        writer.write_table(t)
        n_out += t.num_rows
    if len(add):
        t = pa.Table.from_pandas(add, schema=schema, preserve_index=False)
        writer.write_table(t)
        n_out += t.num_rows
    writer.close()
    print(f"\nwrote {args.output}")
    print(f"  {n_out:,} rows = {n_out - len(add):,} original + {len(add):,} added isolated primaries "
          f"({100 * len(add) / max(n_out, 1):.2f}% of the new catalogue)")


if __name__ == "__main__":
    main()
