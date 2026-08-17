"""Is the blend response nonlinear in shear? Same pairs, Dg=0.2 vs Dg=0.05, measured directly.

WHY THIS EXISTS
---------------
After correcting the distance definition and the close-pair training imbalance, ~22% of the <1"
deficit survives, and it is NOT a population effect: at <1" the model predicts 0.0391 on the ruler
and 0.0393 on its own training catalogue -- the same features -- yet the ruler measures 0.0518 and
the label 0.0423 (WORKLOG 2026-07-28n, RESULT 9). The two differ in how the response was
DIFFERENCED:

    training label : (e(g_s=0.2)  - e(0)) / 0.2
    ruler truth    : (e(g_s=0.05) - e(0)) / 0.05

Equal only if the response is linear in the neighbour's shear out to |g|=0.2. Job 15329219 bounds
the nonlinearity below ~2% over 1-3", but that bound says nothing about <0.5", where the profiles
overlap and saturation is physically expected -- exactly where the residual lives.

An earlier attempt to test this by joining the response catalogue to the half-shear ruler failed:
blendemu shears only the second half of the input catalogue, so their targets are disjoint. But the
g=0.05 legs exist on disk, so the RESPONSE CATALOGUE ITSELF can be rebuilt at Dg=0.05 from existing
shape and cross-match files (`rebuild_response_norej.py` with configs/fs2_lsst_r_extnbr_g005.yaml).
That gives the same galaxies, same estimator, same pairing, same bright-neighbour rejection, and the
same unsheared reference leg -- only the sheared leg changes. This joins the two per pair.

Because both catalogues take their geometry from the SAME g=0 leg, the `distance` column is
identical for a matched pair, so the binning is consistent by construction.

WHAT TO CONCLUDE
----------------
    ratio = label(0.2) / label(0.05)

    ratio ~ 1 everywhere      -> linear; amplitude is NOT the residual, and something else is left
    ratio < 1 at small d only -> saturation, as predicted. The Dg=0.2 labels UNDERSTATE the response
                                 our pipeline needs, and rebuilding at Dg=0.05 is not just the test
                                 but the fix.

NOTE cases 100-199 have no g=0.05 shape catalogue, so this covers cases 0-99.
FIREWALL: reads response catalogues only; constgold is never touched.
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc
from sbs_shear.paths import SIM_BASE as BASE

CAT_G02 = os.path.join(BASE, "response_catalogue_train.feather")
CAT_G005 = os.path.join(BASE, "response_catalogue_g005_train.feather")
DIST_EDGES = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
POS_ROUND = 7          # deg; 0.36 mas -- far below any galaxy separation, above the ~1e-10 noise


def load(path, max_dist, re_min, mag_max, max_case, label_shear, tag):
    cols = ["case", "input_index", "RA_input_s", "DEC_input_s", "distance", "delta_et1",
            "Re_input_p", "r_input_p"]
    parts = []
    t0 = time.time()
    with ipc.open_file(path) as r:
        nb = r.num_record_batches
        for bi in range(nb):
            b = pa.Table.from_batches([r.get_batch(bi)]).select(cols).to_pandas()
            b = b[(b["distance"].to_numpy(float) < max_dist)
                  & np.isfinite(b["delta_et1"].to_numpy(float))
                  & (b["Re_input_p"].to_numpy(float) > re_min)
                  & (b["r_input_p"].to_numpy(float) < mag_max)]
            if max_case is not None:
                b = b[b["case"] < max_case]
            if len(b):
                parts.append(b[["case", "input_index", "RA_input_s", "DEC_input_s",
                                "distance", "delta_et1"]])
    d = pd.concat(parts, ignore_index=True)
    d[f"resp_{tag}"] = d["delta_et1"].to_numpy(float) / label_shear
    d["_ra"] = np.round(d["RA_input_s"].to_numpy(float), POS_ROUND)
    d["_dec"] = np.round(d["DEC_input_s"].to_numpy(float), POS_ROUND)
    print(f"{tag}: {len(d):,} in-domain pairs < {max_dist}\"  cases "
          f"{d['case'].min()}-{d['case'].max()}  ({time.time()-t0:.0f}s)", flush=True)
    return d


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-dist", type=float, default=3.0)
    ap.add_argument("--max-case", type=int, default=100)
    ap.add_argument("--true-re-min", type=float, default=0.3)
    ap.add_argument("--true-mag-max", type=float, default=26.0)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    a = load(CAT_G02, args.max_dist, args.true_re_min, args.true_mag_max, args.max_case, 0.2, "g02")
    b = load(CAT_G005, args.max_dist, args.true_re_min, args.true_mag_max, args.max_case, 0.05,
             "g005")

    keys = ["case", "input_index", "_ra", "_dec"]
    m = a[keys + ["distance", "resp_g02"]].merge(
        b[keys + ["distance", "resp_g005"]], on=keys, how="inner", suffixes=("", "_b"))
    m = m.drop_duplicates(keys)
    print(f"\nmatched pairs: {len(m):,}  (g02 {len(a):,}, g005 {len(b):,})")
    if len(m) < 1000:
        print("*** join failed -- cannot compare per-pair ***")
        print("AMPLITUDE_DIRECT_DONE", flush=True)
        return
    dd = (m["distance"] - m["distance_b"]).abs().to_numpy(float)
    print(f"  sanity: |distance - distance_b| max={dd.max():.2e}\" "
          f"(both take geometry from the same g=0 leg, so this must be ~0)")

    r02 = m["resp_g02"].to_numpy(float)
    r005 = m["resp_g005"].to_numpy(float)
    dist = m["distance"].to_numpy(float)
    ok = np.isfinite(r02) & np.isfinite(r005)

    print(f"\n[SAME PAIRS, two shear amplitudes]")
    print(f"  {'distance':>16} {'resp(Dg=0.05)':>14} {'resp(Dg=0.2)':>13} {'ratio 0.2/0.05':>16} "
          f"{'sem(.05)':>9} {'matched':>10}")
    rows = []
    for i in range(len(DIST_EDGES) - 1):
        sel = ok & (dist >= DIST_EDGES[i]) & (dist < DIST_EDGES[i + 1])
        n = int(sel.sum())
        if n < 200:
            continue
        lo, hi = r005[sel].mean(), r02[sel].mean()
        slo = r005[sel].std(ddof=1) / np.sqrt(n)
        shi = r02[sel].std(ddof=1) / np.sqrt(n)
        ratio = hi / lo if lo else np.nan
        rsem = abs(ratio) * np.sqrt((shi / hi) ** 2 + (slo / lo) ** 2) if (hi and lo) else np.nan
        rows.append((DIST_EDGES[i], DIST_EDGES[i + 1], lo, hi, ratio, rsem, n))
        print(f"  [{DIST_EDGES[i]:>6.2f},{DIST_EDGES[i+1]:>6.2f}) {lo:>14.4f} {hi:>13.4f} "
              f"{ratio:>9.3f}+-{rsem:<6.3f} {slo:>9.4f} {n:>10,}")
    lo, hi = r005[ok].mean(), r02[ok].mean()
    print(f"  {'ALL MATCHED':>16} {lo:>14.4f} {hi:>13.4f} {hi/lo:>16.3f} "
          f"{r005[ok].std(ddof=1)/np.sqrt(ok.sum()):>9.4f} {int(ok.sum()):>10,}")

    print("\nREAD THIS AS: ratio < 1 at small separation = the Dg=0.2 secant understates the slope,")
    print("i.e. the certified labels are the wrong quantity for the m pipeline exactly where the")
    print("residual deficit lives. ratio ~ 1 = linear, and the residual is something else.")

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        np.savez(args.output, r02=r02, r005=r005, distance=dist,
                 rows=np.array(rows, dtype=float))
        print(f"\nsaved {args.output}")
    print("AMPLITUDE_DIRECT_DONE", flush=True)


if __name__ == "__main__":
    main()
