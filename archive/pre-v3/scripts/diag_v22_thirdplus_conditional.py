"""Does third-plus neighbour flux predict V2.2 residual after matching confounders?

The marginal neighbour-flux diagnostic found structured, non-monotonic closure.
This follow-up asks the narrower question suggested by the toy/emulator
distinction: at comparable primaries and comparable leading pairs, does flux in
the third-brightest and later neighbours still separate the residual?

Two controls are reported, without fitting any response:

  A. primary true magnitude, primary true size, and the summed flux of the two
     brightest neighbours inside 10 arcsec;
  B. the same intrinsic controls plus deployed total predicted R_blend.

Each control axis is split into global quantile bins.  Inside every occupied joint
cell, rows are ranked by intrinsic third-plus flux and assigned to four equal
count groups.  Pooling the within-cell ranks approximately balances the
controls while retaining the whole population.  All quartiles and all qN-q1
contrasts are reported.  Constgold is evaluation-only; no correction is fit.
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


LOOKUP_COLUMNS = ["logflux_all_0_10", "logflux_thirdplus_0_10"]


def quantile_index(
    x: np.ndarray, mask: np.ndarray, n_bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    edges = np.quantile(x[mask], np.linspace(0.0, 1.0, n_bins + 1))
    if np.any(np.diff(edges) <= 0):
        raise RuntimeError(f"non-distinct control quartiles: {edges.tolist()}")
    edges[-1] = np.nextafter(edges[-1], np.inf)
    index = np.searchsorted(edges, x, side="right") - 1
    index[~mask] = -1
    if np.any(mask & ((index < 0) | (index >= n_bins))):
        raise RuntimeError("control quantiles do not cover the analysis domain")
    return index.astype(np.int16), edges


def joint_cells(
    controls: list[np.ndarray], bases: list[int], mask: np.ndarray,
) -> np.ndarray:
    cell = np.zeros(len(mask), dtype=np.int32)
    for control, base in zip(controls, bases):
        cell = cell * base + control
    cell[~mask] = -1
    return cell


def within_cell_quartiles(
    exposure: np.ndarray,
    cells: np.ndarray,
    mask: np.ndarray,
    min_cell: int = 400,
) -> tuple[np.ndarray, dict[str, int]]:
    """Assign exposure ranks 0..3 within cells; exclude only undersized cells."""
    selected = np.flatnonzero(mask)
    order = np.lexsort((exposure[selected], cells[selected]))
    row = selected[order]
    sorted_cells = cells[row]
    starts = np.r_[0, np.flatnonzero(sorted_cells[1:] != sorted_cells[:-1]) + 1]
    stops = np.r_[starts[1:], len(row)]
    ids = np.full(len(mask), -1, dtype=np.int16)
    kept_cells = 0
    for start, stop in zip(starts, stops):
        n = int(stop - start)
        if n < min_cell:
            continue
        rr = row[start:stop]
        # Stable rank splitting reports every row even if exposure values tie.
        q = np.minimum((np.arange(n, dtype=np.int64) * 4) // n, 3)
        ids[rr] = q.astype(np.int16)
        kept_cells += 1
    info = {
        "total_cells": int(len(starts)),
        "kept_cells": int(kept_cells),
        "kept_rows": int((ids >= 0).sum()),
        "excluded_rows": int((mask & (ids < 0)).sum()),
    }
    return ids, info


def balance_report(name: str, ids: np.ndarray, controls: dict[str, np.ndarray]) -> None:
    print(f"\n  balance after {name} within-cell ranking:")
    print(f"  {'control':>18}" + "".join(f"{'q'+str(q+1):>12}" for q in range(4))
          + f"{'max SMD':>12}")
    use = ids >= 0
    for label, x in controls.items():
        means = np.asarray([x[ids == q].mean() for q in range(4)])
        scale = x[use].std(ddof=1)
        imbalance = (means.max() - means.min()) / scale if scale else np.nan
        print(f"  {label:>18}" + "".join(f"{v:>12.5f}" for v in means)
              + f"{imbalance:>12.4f}")


def contrast_report(table: Table, first_rsim: np.ndarray) -> None:
    """Paired-seed and case-blocked-simulation errors for every qN-q1 contrast."""
    a = np.stack(table.per_seed)  # seed, q, (Rsim, Rflow, Rblend)
    m = 100.0 * (a[:, :, 0] / (a[:, :, 1] + a[:, :, 2]) - 1.0)
    totals = (a[:, :, 1] + a[:, :, 2]).mean(axis=0)

    ids, case_idx, ncase = table.ids, table.case_idx, table.ncase
    valid = ids >= 0
    flat = ids[valid].astype(np.int64) * ncase + case_idx[valid]
    size = table.nrow * ncase
    sums = np.bincount(flat, weights=first_rsim[valid], minlength=size).reshape(table.nrow, ncase)
    counts = np.bincount(flat, minlength=size).reshape(table.nrow, ncase)
    case_means = np.divide(sums, counts, out=np.full_like(sums, np.nan), where=counts > 0)

    print("  conditional contrasts (qN-q1), paired across seeds and cases:")
    print(f"  {'contrast':>12}{'delta m %':>12}{'+-seed':>10}{'+-sim':>10}{'+-total':>10}")
    for q in range(1, 4):
        seed_delta = m[:, q] - m[:, 0]
        e_seed = seed_delta.std(ddof=1) / np.sqrt(len(seed_delta))
        have = np.isfinite(case_means[q]) & np.isfinite(case_means[0])
        case_delta = 100.0 * (
            case_means[q, have] / totals[q] - case_means[0, have] / totals[0]
        )
        e_sim = case_delta.std(ddof=1) / np.sqrt(have.sum())
        print(f"  {'q'+str(q+1)+'-q1':>12}{seed_delta.mean():>+12.3f}{e_seed:>10.3f}"
              f"{e_sim:>10.3f}{np.hypot(e_seed, e_sim):>10.3f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dump-glob", required=True)
    ap.add_argument("--catalogue", required=True)
    ap.add_argument("--flux-lookup", required=True)
    ap.add_argument("--min-case", type=int, default=40)
    ap.add_argument("--mag-max", type=float, default=25.8)
    ap.add_argument("--re-min", type=float, default=0.5)
    ap.add_argument("--min-cell", type=int, default=100)
    ap.add_argument("--mag-bins", type=int, default=32,
                    help="primary-mag matching bins; 32 chosen after pre-outcome balance checks "
                         "gave max SMD 0.42 with 4 bins and 0.176 with 8")
    ap.add_argument("--other-bins", type=int, default=8,
                    help="matching bins for primary size, top-two flux and predicted R_blend")
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
        args.flux_lookup, columns=["case", "input_index", *LOOKUP_COLUMNS],
    ).to_pandas()
    if lookup.duplicated(["case", "input_index"]).any():
        raise RuntimeError("neighbour-flux lookup keys are not unique")
    keyed = truth[["case", "input_index"]].merge(
        lookup, on=["case", "input_index"], how="left", sort=False, validate="many_to_one",
    )
    if len(keyed) != len(truth) or keyed[LOOKUP_COLUMNS[0]].isna().any():
        raise RuntimeError("neighbour-flux lookup does not cover the catalogue exactly")
    if not np.array_equal(keyed["case"].to_numpy(), truth["case"].to_numpy()) \
            or not np.array_equal(keyed["input_index"].to_numpy(), truth["input_index"].to_numpy()):
        raise RuntimeError("neighbour-flux join changed catalogue order")

    first = pf.read_table(
        dumps[0], columns=["case", "input_index", "r_sim", "R_flow", "R_blend"],
    )
    if len(first) != len(truth) \
            or not np.array_equal(first["case"].to_numpy(), truth["case"].to_numpy(np.int64)) \
            or not np.array_equal(first["input_index"].to_numpy(),
                                  truth["input_index"].to_numpy(np.int64)):
        raise RuntimeError("first dump does not align with catalogue")
    rblend = first["R_blend"].to_numpy(zero_copy_only=False).astype(float)

    log_all = keyed["logflux_all_0_10"].to_numpy(float)
    log_third = keyed["logflux_thirdplus_0_10"].to_numpy(float)
    all_ratio = np.expm1(log_all * np.log(10.0))
    third_ratio = np.expm1(log_third * np.log(10.0))
    log_top2 = np.log10(1.0 + np.maximum(all_ratio - third_ratio, 0.0))

    values = {"primary mag": mag, "primary Re": re_, "top2 flux": log_top2,
              "pred R_blend": rblend}
    control_bins = {"primary mag": args.mag_bins, "primary Re": args.other_bins,
                    "top2 flux": args.other_bins, "pred R_blend": args.other_bins}
    indices, edges = {}, {}
    for name, value in values.items():
        indices[name], edges[name] = quantile_index(value, domain, control_bins[name])

    cases = truth["case"].to_numpy(np.int64)
    unique_cases = np.unique(cases)
    blocks = (np.searchsorted(unique_cases, cases), len(unique_cases))
    specifications = [
        ("A: primary + top2 intrinsic flux", ["primary mag", "primary Re", "top2 flux"]),
        ("B: A + predicted R_blend", ["primary mag", "primary Re", "top2 flux", "pred R_blend"]),
    ]
    tables = []
    print(f"Control quantile bins: {control_bins}")
    print(f"Control quantile edges: { {k: v.tolist() for k, v in edges.items()} }")
    for title, names in specifications:
        cells = joint_cells(
            [indices[name] for name in names], [control_bins[name] for name in names], domain,
        )
        ids, info = within_cell_quartiles(log_third, cells, domain, min_cell=args.min_cell)
        print(f"{title}: {info}")
        table = Table(
            title, "conditional third+", ["q1 low", "q2", "q3", "q4 high"],
            ids.astype(np.int64), 4, *blocks,
        )
        tables.append((table, names))
        balance_report(title, ids, {name: values[name] for name in names})

    first_rsim = None
    for path in dumps:
        dump = pf.read_table(path, columns=["case", "input_index", "r_sim", "R_flow", "R_blend"])
        if len(dump) != len(truth) \
                or not np.array_equal(dump["case"].to_numpy(), truth["case"].to_numpy(np.int64)) \
                or not np.array_equal(dump["input_index"].to_numpy(),
                                      truth["input_index"].to_numpy(np.int64)):
            raise RuntimeError(f"dump key alignment failure: {path}")
        rb = dump["R_blend"].to_numpy(zero_copy_only=False).astype(float)
        if not np.array_equal(rb, rblend):
            raise RuntimeError(f"R_blend differs across seeds: {path}")
        rs = dump["r_sim"].to_numpy(zero_copy_only=False).astype(float)
        if first_rsim is None:
            first_rsim = rs.copy()
        elif not np.array_equal(rs, first_rsim):
            raise RuntimeError(f"R_sim differs across seeds: {path}")
        components = (
            rs,
            dump["R_flow"].to_numpy(zero_copy_only=False).astype(float),
            rb,
        )
        for table, _ in tables:
            table.add(*components)

    print(f"\nV2.2 domain N={domain.sum():,}; cases={len(unique_cases)}; dumps={len(dumps)}")
    for table, _ in tables:
        table.report()
        report_total_m(table)
        contrast_report(table, first_rsim)
    print("\nDIAG_V22_THIRDPLUS_CONDITIONAL_DONE", flush=True)


if __name__ == "__main__":
    main()
