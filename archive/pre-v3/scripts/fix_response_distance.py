"""Rewrite the emulator's training catalogue with an INPUT-frame `distance`, matching inference.

THE MISMATCH
------------
`blendemu/response.py::retrieve_response` measures the pair separation from the primary's DETECTED
centroid in the unsheared leg:

    pri_pos = scat1[['X_WORLD', 'Y_WORLD']].loc[idx_det1]      # detection
    dst, ind = kdt_in.query(pri_pos, ...)                      # to INPUT secondary positions

so the emulator's `distance` feature is a detected-to-input separation. But the half-shear catalogue
that the ruler and the m pipeline both use is built by `_nearest_neighbor_features`, which measures
input-to-input:

    all_pos = icat[['RA_input', 'DEC_input']]

A blended primary's centroid is pulled toward its neighbour, so the training distance runs
systematically SHORT of the input separation for close pairs -- and the model is then queried at
input separations it never saw for those pairs. Under-prediction at small d, vanishing at large d.

THE FIX IS FREE
---------------
The response catalogue already stores `RA_input_p/DEC_input_p` and `RA_input_s/DEC_input_s` (both
copied straight from the input catalogue), so the input-frame separation can be recomputed from the
existing rows. No re-simulation, no re-measurement, not even a catalogue rebuild -- only the feature
column changes; every label is untouched.

The output goes to a sibling directory holding a single `response_catalogue_train.feather`, so
blendemu's own `scripts/retrain_extnbr.py` can be pointed at it with nothing but a CONFIG_PATH whose
`simulation.output_path` is that directory. blendemu itself is never edited.

This also PRINTS the comparison between the two definitions over the whole training set, which is
the direct measurement of the mismatch independent of any join.
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc

SRC = "/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876/response_catalogue_train.feather"
DIST_EDGES = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0]


def input_frame_distance(df):
    """Great-circle separation in arcsec between the two stored INPUT positions."""
    ra_p = np.deg2rad(df["RA_input_p"].to_numpy(float))
    de_p = np.deg2rad(df["DEC_input_p"].to_numpy(float))
    ra_s = np.deg2rad(df["RA_input_s"].to_numpy(float))
    de_s = np.deg2rad(df["DEC_input_s"].to_numpy(float))
    # haversine: stable at the sub-arcsecond separations that matter here
    dra, dde = ra_s - ra_p, de_s - de_p
    h = np.sin(dde / 2) ** 2 + np.cos(de_p) * np.cos(de_s) * np.sin(dra / 2) ** 2
    return np.rad2deg(2 * np.arcsin(np.sqrt(np.clip(h, 0, 1)))) * 3600.0


class Stats:
    """Per-bin accumulators over the streamed read (keyed on the DETECTED distance, as stored)."""

    def __init__(self, edges):
        self.edges = np.asarray(edges, float)
        n = len(self.edges) - 1
        self.count = np.zeros(n)
        self.s_det = np.zeros(n)
        self.s_in = np.zeros(n)
        self.s_diff = np.zeros(n)
        self.s_closer = np.zeros(n)

    def add(self, d_det, d_in):
        idx = np.digitize(d_det, self.edges) - 1
        ok = (idx >= 0) & (idx < len(self.edges) - 1)
        idx, dd, di = idx[ok], d_det[ok], d_in[ok]
        if not idx.size:
            return
        n = len(self.edges) - 1
        self.count += np.bincount(idx, minlength=n)
        self.s_det += np.bincount(idx, weights=dd, minlength=n)
        self.s_in += np.bincount(idx, weights=di, minlength=n)
        self.s_diff += np.bincount(idx, weights=dd - di, minlength=n)
        self.s_closer += np.bincount(idx, weights=(dd < di).astype(float), minlength=n)

    def report(self):
        print(f"\n[DISTANCE DEFINITION over the whole training catalogue]")
        print(f"  {'stored (det)':>16} {'<d_det>':>9} {'<d_input>':>10} {'<d_det-d_in>':>13} "
              f"{'frac closer':>12} {'N':>13}")
        for i in range(len(self.edges) - 1):
            if self.count[i] < 200:
                continue
            c = self.count[i]
            print(f"  [{self.edges[i]:>6.2f},{self.edges[i+1]:>6.2f}) {self.s_det[i]/c:>9.4f} "
                  f"{self.s_in[i]/c:>10.4f} {self.s_diff[i]/c:>+13.4f} "
                  f"{self.s_closer[i]/c:>12.3f} {int(c):>13,}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--out-dir",
                    default="/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_inputdist")
    ap.add_argument("--stats-only", action="store_true",
                    help="measure the two definitions without writing the corrected catalogue")
    args = ap.parse_args()

    out_file = os.path.join(args.out_dir, "response_catalogue_train.feather")
    if not args.stats_only:
        os.makedirs(args.out_dir, exist_ok=True)
        if os.path.exists(out_file):
            raise SystemExit(f"refusing to overwrite existing {out_file}")

    stats = Stats(DIST_EDGES)
    writer = None
    schema = None
    n_rows = 0
    t0 = time.time()
    with ipc.open_file(args.src) as r:
        nb = r.num_record_batches
        for bi in range(nb):
            tbl = pa.Table.from_batches([r.get_batch(bi)])
            df = tbl.to_pandas()
            d_det = df["distance"].to_numpy(float)
            d_in = input_frame_distance(df)
            stats.add(d_det, d_in)
            if not args.stats_only:
                df["distance"] = d_in
                df["distance_detected_frame"] = d_det
                out = pa.Table.from_pandas(df, preserve_index=False)
                if writer is None:
                    schema = out.schema
                    writer = ipc.new_file(out_file, schema)
                else:
                    out = out.cast(schema)
                for b in out.to_batches(max_chunksize=65536):
                    writer.write_batch(b)
                n_rows += len(df)
            if bi % 400 == 0:
                print(f"  batch {bi}/{nb}  ({time.time()-t0:.0f}s)", flush=True)
    if writer is not None:
        writer.close()

    stats.report()
    if not args.stats_only:
        print(f"\nwrote {n_rows:,} rows -> {out_file}  ({time.time()-t0:.0f}s)")
        print("`distance` is now INPUT-frame; the original value is kept as "
              "`distance_detected_frame`.")
        print("Retrain with blendemu's own script by pointing CONFIG_PATH at a config whose")
        print(f"simulation.output_path is {args.out_dir}")
    print("FIX_DISTANCE_DONE", flush=True)


if __name__ == "__main__":
    main()
