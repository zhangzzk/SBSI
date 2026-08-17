"""Are blendemu's "primary" and "secondary" halves the same galaxy population?

WHY THIS MATTERS
----------------
`_primary_secondary_rows` splits the input catalogue at the midpoint of the stable input id:

    split = int(total_input_count / 2)
    return np.where(ids < split)[0], np.where(ids >= split)[0]   # unsheared primaries, sheared secondaries

Job 15328681 showed the consequence: in the half-shear sim only the SECOND half ever carries shear,
so the ruler's both-sheared sample (`eval_rblend_gap`) consists entirely of second-half galaxies,
while the emulator's training pairs are (first-half target, second-half neighbour). The two share
ZERO targets -- which is why the per-pair join returned nothing, and why the Dg=0.2 vs Dg=0.05
amplitude comparison cannot be done on these catalogues at all (ngmix shapes exist only where the
target is sheared, i.e. only on second-half targets, which the response catalogue never contains).

That leaves a question that DOES matter for every cross-catalogue number in this investigation, and
for the m pipeline itself (which applies the emulator to all pairs, both halves): if the split is by
index and the catalogue order is random with respect to galaxy properties, the two halves are the
same population and the emulator transfers. If the order is sorted -- by magnitude, size, position,
anything -- they are not, and the emulator is being asked about a population it never saw.

WHAT THIS MEASURES
------------------
Per half, the distribution of every feature the emulator conditions on, plus sky position, both over
the whole catalogue and restricted to the deliverable domain. A KS-style comparison is not needed:
if the halves differ, the quantiles will say so plainly.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc

RULER = "/project/ls-gruen/users/zekang.zhang/sbsi_catalogues/det_meas_ngmix_g0.05_val.feather"
FEATURES = ["r_input_p", "Re_input_p", "sersic_n_input_p", "RA_input_p", "DEC_input_p"]
QUANTILES = [0.01, 0.10, 0.25, 0.50, 0.75, 0.90, 0.99]


def read_case(path, case, cols):
    parts = []
    with ipc.open_file(path) as r:
        for bi in range(r.num_record_batches):
            b = pa.Table.from_batches([r.get_batch(bi)]).select(cols).to_pandas()
            b = b[b["case"] == case]
            if len(b):
                parts.append(b)
    return pd.concat(parts, ignore_index=True)


def compare(a, b, label_a, label_b, title):
    print(f"\n[{title}]  {label_a}: N={len(a):,}   {label_b}: N={len(b):,}")
    print(f"  {'feature':>16} {'quantile':>9} {label_a:>12} {label_b:>12} {'diff':>10}")
    for f in FEATURES:
        va, vb = a[f].to_numpy(float), b[f].to_numpy(float)
        va, vb = va[np.isfinite(va)], vb[np.isfinite(vb)]
        if not len(va) or not len(vb):
            continue
        for q in QUANTILES:
            qa, qb = np.quantile(va, q), np.quantile(vb, q)
            print(f"  {f if q == QUANTILES[0] else '':>16} {q:>9.2f} {qa:>12.4f} {qb:>12.4f} "
                  f"{qb-qa:>+10.4f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", type=int, default=0)
    ap.add_argument("--true-re-min", type=float, default=0.3)
    ap.add_argument("--true-mag-max", type=float, default=26.0)
    args = ap.parse_args()

    cols = ["case", "input_index", "detected", "distance",
            "gamma1_input_p", "gamma2_input_p"] + FEATURES
    d = read_case(RULER, args.case, cols)
    ids = d["input_index"].to_numpy(int)
    split = int((ids.max() + 1) / 2)
    first, second = d[ids < split], d[ids >= split]
    print(f"case {args.case}: N={len(d):,}  input_index max={ids.max():,}  split at {split:,}")

    gp = np.hypot(d["gamma1_input_p"].to_numpy(float), d["gamma2_input_p"].to_numpy(float))
    print(f"  sheared targets: {int((gp > 1e-6).sum()):,}  "
          f"of which first half {int(((gp > 1e-6) & (ids < split)).sum()):,}, "
          f"second half {int(((gp > 1e-6) & (ids >= split)).sum()):,}")
    print("  -> confirms which half carries shear")

    compare(first, second, "first(pri)", "second(sec)", "WHOLE CATALOGUE")

    dom = d[(d["Re_input_p"].to_numpy(float) > args.true_re_min)
            & (d["r_input_p"].to_numpy(float) < args.true_mag_max)]
    idsd = dom["input_index"].to_numpy(int)
    compare(dom[idsd < split], dom[idsd >= split], "first(pri)", "second(sec)",
            "DELIVERABLE DOMAIN")

    # interleaving on the sky: if the halves were spatially segregated, neighbour statistics would
    # differ between them regardless of marginal property distributions
    for name, part in (("first(pri) ", first), ("second(sec)", second)):
        v = part["distance"].to_numpy(float)
        v = v[np.isfinite(v)]
        print(f"\n  {name} nearest-neighbour distance: median={np.median(v):.4f}\"  "
              f"frac<1\"={np.mean(v < 1):.4f}  frac<2\"={np.mean(v < 2):.4f}  N={len(v):,}")
    print("DIAG_PRISEC_DONE", flush=True)


if __name__ == "__main__":
    main()
