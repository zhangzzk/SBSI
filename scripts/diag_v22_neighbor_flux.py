"""Localise V2.2 constgold closure by intrinsic full-scene neighbour flux.

Unlike the predicted-R_blend diagnostic, the binning variables here contain no
model prediction.  They are computed from the complete input field before
detection: summed secondary-to-primary flux in 0--1, 1--3, and 3--10 arcsec
shells, plus the flux in the third-brightest and later neighbours within 10
arcsec.  This tests whether a pair-only emulator misses scene context from
additional galaxies even when response derivatives add linearly.

Constgold is evaluation-only.  Nothing is fitted or corrected.
"""
from __future__ import annotations

import argparse
import glob
import time

import numpy as np
import pyarrow.feather as pf

from scripts.diag_rflow_v21 import Table
from scripts.diag_v22_blendness import report_total_m
from scripts.eval_v2_indomain_m import catalogue_true_props


FLUX_COLUMNS = [
    ("near 0--1 arcsec", "logflux_near_0_1"),
    ("mid 1--3 arcsec", "logflux_mid_1_3"),
    ("far 3--10 arcsec", "logflux_far_3_10"),
    ("all 0--10 arcsec", "logflux_all_0_10"),
    ("third+ bright rank", "logflux_thirdplus_0_10"),
]


def flux_quantile_table(title, key, domain, blocks):
    """Zero-flux class plus four positive-flux quantiles, fixed without responses."""
    positive = domain & (key > 0)
    if positive.sum() < 8_000:
        raise RuntimeError(f"too few positive rows for {title}: {positive.sum():,}")
    edges = np.quantile(key[positive], [0.0, 0.25, 0.5, 0.75, 1.0])
    edges[-1] = np.nextafter(edges[-1], np.inf)
    masks, labels = [], []
    if np.any(domain & (key == 0)):
        masks.append(domain & (key == 0))
        labels.append("zero flux")
    for q in range(4):
        masks.append(domain & (key >= edges[q]) & (key < edges[q + 1]))
        labels.append(f"q{q+1} [{edges[q]:.3f},{edges[q+1]:.3f})")
    ids = np.full(len(key), -1, np.int64)
    for i, mask in enumerate(masks):
        if np.any((ids >= 0) & mask):
            raise RuntimeError(f"overlapping masks in {title}")
        ids[mask] = i
    if np.any(domain & (ids < 0)):
        raise RuntimeError(f"unassigned in-domain rows in {title}")
    return Table(title, "log10(1+Fn/Fp)", labels, ids, len(labels), *blocks), edges


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dump-glob", required=True)
    ap.add_argument("--catalogue", required=True)
    ap.add_argument("--flux-lookup", required=True)
    ap.add_argument("--min-case", type=int, default=40)
    ap.add_argument("--mag-max", type=float, default=25.8)
    ap.add_argument("--re-min", type=float, default=0.5)
    args = ap.parse_args()
    t0 = time.time()

    dumps = sorted(glob.glob(args.dump_glob))
    if len(dumps) != 16:
        raise SystemExit(f"expected 16 V2.2 dumps, found {len(dumps)}")
    truth = catalogue_true_props(args.catalogue, args.min_case, t0)
    mag = truth["r_input_p"].to_numpy(float)
    re_ = truth["Re_input_p"].to_numpy(float)
    domain = (mag < args.mag_max) & (re_ > args.re_min)

    lookup = pf.read_table(
        args.flux_lookup,
        columns=["case", "input_index", *[column for _, column in FLUX_COLUMNS]],
    ).to_pandas()
    if lookup.duplicated(["case", "input_index"]).any():
        raise RuntimeError("neighbour-flux lookup keys are not unique")
    keyed = truth[["case", "input_index"]].merge(
        lookup, on=["case", "input_index"], how="left", sort=False, validate="many_to_one",
    )
    if len(keyed) != len(truth):
        raise RuntimeError("neighbour-flux join changed row count")
    missing = keyed[FLUX_COLUMNS[0][1]].isna().to_numpy()
    if missing.any():
        raise RuntimeError(f"neighbour-flux lookup misses {missing.sum():,} catalogue rows")
    if not np.array_equal(keyed["case"].to_numpy(), truth["case"].to_numpy()) \
            or not np.array_equal(keyed["input_index"].to_numpy(), truth["input_index"].to_numpy()):
        raise RuntimeError("neighbour-flux join changed catalogue order")

    cases = truth["case"].to_numpy(np.int64)
    unique_cases = np.unique(cases)
    blocks = (np.searchsorted(unique_cases, cases), len(unique_cases))
    tables, edge_report = [], {}
    for title, column in FLUX_COLUMNS:
        table, edges = flux_quantile_table(
            f"by intrinsic neighbour flux: {title}",
            keyed[column].to_numpy(float), domain, blocks,
        )
        tables.append(table)
        edge_report[column] = edges.tolist()

    first_rb = None
    for path in dumps:
        dump = pf.read_table(path, columns=["case", "input_index", "r_sim", "R_flow", "R_blend"])
        if len(dump) != len(truth):
            raise RuntimeError(f"dump/catalogue length mismatch: {path}")
        if not np.array_equal(dump["case"].to_numpy(), truth["case"].to_numpy(np.int64)) \
                or not np.array_equal(dump["input_index"].to_numpy(),
                                      truth["input_index"].to_numpy(np.int64)):
            raise RuntimeError(f"dump key alignment failure: {path}")
        rb = dump["R_blend"].to_numpy(zero_copy_only=False).astype(float)
        if first_rb is None:
            first_rb = rb.copy()
        elif not np.array_equal(rb, first_rb):
            raise RuntimeError(f"R_blend differs across seeds: {path}")
        values = (
            dump["r_sim"].to_numpy(zero_copy_only=False).astype(float),
            dump["R_flow"].to_numpy(zero_copy_only=False).astype(float),
            rb,
        )
        for table in tables:
            table.add(*values)

    print(f"\nV2.2 domain N={domain.sum():,}; cases={len(unique_cases)}; dumps={len(dumps)}")
    print("Flux variables use the complete intrinsic input field and no measured/model quantity.")
    print("Shells were fixed before response inspection: 0--1, 1--3, 3--10 arcsec.")
    print(f"Positive-flux quartile edges: {edge_report}")
    for table in tables:
        table.report()
        report_total_m(table)
    print("\nDIAG_V22_NEIGHBOUR_FLUX_DONE", flush=True)


if __name__ == "__main__":
    main()
