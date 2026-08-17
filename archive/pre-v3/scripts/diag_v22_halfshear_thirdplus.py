"""Test V2.2 flow against its own half-shear self-response by scene context.

This is the attribution test for the third-plus-neighbour structure seen on
constgold. The half-shear simulation isolates the primary SELF response by
projecting the forward leg difference on the primary's independent shear
direction. Therefore this script compares only

    m_self = mean(r_sim_self) / mean(R_flow) - 1,

with no BlendEMU term.

Rows are ranked by intrinsic third-and-later neighbour flux within joint cells
of primary true magnitude, primary true size, and the summed intrinsic flux of
the two brightest neighbours. This is the same balance construction used for
the constgold diagnostic. It fits no response and uses no constgold result.
All four conditional quartiles and all qN-q1 contrasts are reported. The run
aborts if any control has a maximum standardized mean difference above 0.1.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pyarrow.feather as pf

from scripts.diag_v22_thirdplus_conditional import (
    joint_cells,
    quantile_index,
    within_cell_quartiles,
)


KEYS = ["case", "input_index"]
LOOKUP_COLUMNS = ["logflux_all_0_10", "logflux_thirdplus_0_10"]
EXPECTED_SEEDS = ["501", "502", "503"] + [str(s) for s in range(505, 518)]


def balance_statistics(
    ids: np.ndarray, controls: dict[str, np.ndarray], max_smd: float,
) -> dict[str, float]:
    """Print and return max standardized mean differences across quartiles."""
    print("\n  balance after within-cell ranking:")
    print(f"  {'control':>18}" + "".join(f"{'q'+str(q+1):>12}" for q in range(4))
          + f"{'max SMD':>12}")
    use = ids >= 0
    out: dict[str, float] = {}
    for label, x in controls.items():
        means = np.asarray([x[ids == q].mean() for q in range(4)])
        scale = x[use].std(ddof=1)
        smd = float((means.max() - means.min()) / scale) if scale else np.nan
        out[label] = smd
        print(f"  {label:>18}" + "".join(f"{v:>12.5f}" for v in means)
              + f"{smd:>12.4f}")
    bad = {name: value for name, value in out.items()
           if not np.isfinite(value) or value > max_smd}
    if bad:
        raise RuntimeError(f"balance gate failed (max SMD {max_smd}): {bad}")
    print(f"  balance gate PASS: every max SMD <= {max_smd:.3f}")
    return out


def report(
    ids: np.ndarray,
    cases: np.ndarray,
    r_sim: np.ndarray,
    flow: np.ndarray,
    absolute: bool = True,
) -> None:
    """Report absolute closure and paired seed/case-blocked contrasts."""
    labels = ["q1 low", "q2", "q3", "q4 high"]
    nseed = flow.shape[1]
    rs = np.asarray([r_sim[ids == q].mean() for q in range(4)])
    rf = np.asarray([[flow[ids == q, s].mean() for q in range(4)]
                     for s in range(nseed)])
    m = 100.0 * (rs[None, :] / rf - 1.0)

    unique_cases = np.unique(cases)
    case_index = np.searchsorted(unique_cases, cases)
    ncase = len(unique_cases)
    use = ids >= 0
    flat = ids[use].astype(np.int64) * ncase + case_index[use]
    sums = np.bincount(flat, weights=r_sim[use], minlength=4 * ncase).reshape(4, ncase)
    counts = np.bincount(flat, minlength=4 * ncase).reshape(4, ncase)
    case_means = np.divide(sums, counts, out=np.full_like(sums, np.nan), where=counts > 0)

    if absolute:
        print("\n  half-shear self-response closure by conditional third-plus flux:")
        print(f"  {'group':>10}{'N':>12}{'R_sim,self':>14}{'R_flow':>12}"
              f"{'m_self %':>12}{'+-seed':>10}{'+-sim':>10}{'+-total':>10}")
        for q, label in enumerate(labels):
            e_seed = m[:, q].std(ddof=1) / np.sqrt(nseed)
            have = np.isfinite(case_means[q])
            # Hold the model denominator fixed so this term carries only
            # case-blocked simulation scatter; checkpoint scatter is separate.
            case_m = 100.0 * (case_means[q, have] / rf[:, q].mean() - 1.0)
            e_sim = case_m.std(ddof=1) / np.sqrt(have.sum())
            print(f"  {label:>10}{int((ids == q).sum()):>12,}{rs[q]:>14.5f}"
                  f"{rf[:, q].mean():>12.5f}{m[:, q].mean():>+12.3f}{e_seed:>10.3f}"
                  f"{e_sim:>10.3f}{np.hypot(e_seed, e_sim):>10.3f}")

    print("\n  conditional contrasts (qN-q1), paired across seeds and cases:")
    print(f"  {'contrast':>12}{'delta m %':>12}{'+-seed':>10}{'+-sim':>10}{'+-total':>10}")
    for q in range(1, 4):
        seed_delta = m[:, q] - m[:, 0]
        e_seed = seed_delta.std(ddof=1) / np.sqrt(nseed)
        have = np.isfinite(case_means[q]) & np.isfinite(case_means[0])
        case_delta = 100.0 * (
            case_means[q, have] / rf[:, q].mean()
            - case_means[0, have] / rf[:, 0].mean()
        )
        e_sim = case_delta.std(ddof=1) / np.sqrt(have.sum())
        print(f"  {'q'+str(q+1)+'-q1':>12}{seed_delta.mean():>+12.3f}{e_seed:>10.3f}"
              f"{e_sim:>10.3f}{np.hypot(e_seed, e_sim):>10.3f}")


def compare_models(
    ids: np.ndarray,
    r_sim: np.ndarray,
    flow: np.ndarray,
    reference_flow: np.ndarray,
) -> None:
    """Paired change in conditional contrasts for two models on identical rows/seeds.

    The simulation response and conditional groups are shared exactly, so this is a
    checkpoint-paired model comparison.  It deliberately reports no absolute bias and
    no simulation-error term: the latter is common to both arms and cancels in the
    model difference to first order.
    """
    rs = np.asarray([r_sim[ids == q].mean() for q in range(4)])
    rf = np.asarray([[flow[ids == q, s].mean() for q in range(4)]
                     for s in range(flow.shape[1])])
    rr = np.asarray([[reference_flow[ids == q, s].mean() for q in range(4)]
                     for s in range(reference_flow.shape[1])])
    m = 100.0 * (rs[None, :] / rf - 1.0)
    mr = 100.0 * (rs[None, :] / rr - 1.0)
    print("\n  paired model change in conditional contrast (candidate-reference):")
    print(f"  {'contrast':>12}{'delta-delta m %':>18}{'+-paired seed':>16}")
    for q in range(1, 4):
        delta = (m[:, q] - m[:, 0]) - (mr[:, q] - mr[:, 0])
        print(f"  {'q'+str(q+1)+'-q1':>12}{delta.mean():>+18.3f}"
              f"{delta.std(ddof=1) / np.sqrt(len(delta)):>16.3f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selfresp", required=True)
    ap.add_argument("--flux-lookup", required=True)
    ap.add_argument("--reference-selfresp", default=None,
                    help="optional dump with the same keys/R_sim and seed columns; report the "
                         "checkpoint-paired change in each conditional contrast")
    ap.add_argument("--mag-max", type=float, default=25.8)
    ap.add_argument("--re-min", type=float, default=0.5)
    ap.add_argument("--min-case", type=int, default=40)
    ap.add_argument("--max-case", type=int, default=199)
    ap.add_argument("--mag-bins", type=int, default=32)
    ap.add_argument("--other-bins", type=int, default=8)
    ap.add_argument("--min-cell", type=int, default=100)
    ap.add_argument("--max-smd", type=float, default=0.1)
    ap.add_argument("--expect-seeds", nargs="+", default=EXPECTED_SEEDS)
    ap.add_argument("--contrasts-only", action="store_true",
                    help="suppress absolute m_self, permitting a four-seed paired-contrast screen")
    args = ap.parse_args()
    t0 = time.time()

    table = pf.read_table(args.selfresp).to_pandas()
    seed_cols = [f"R_flow_s{s}" for s in args.expect_seeds]
    if not args.contrasts_only and len(seed_cols) != 16:
        raise RuntimeError("absolute m_self requires the 16-seed e-response ensemble")
    if args.contrasts_only and len(seed_cols) < 4:
        raise RuntimeError("paired-contrast screening requires at least four seeds")
    missing = [c for c in [*KEYS, "r_sim_self", "r_input_p", "Re_input_p", *seed_cols]
               if c not in table]
    if missing:
        raise RuntimeError(f"self-response dump is missing columns: {missing}")
    if table.duplicated(KEYS).any():
        raise RuntimeError("self-response keys are not unique")

    lookup = pf.read_table(
        args.flux_lookup, columns=[*KEYS, *LOOKUP_COLUMNS],
    ).to_pandas()
    if lookup.duplicated(KEYS).any():
        raise RuntimeError("neighbour-flux lookup keys are not unique")
    joined = table[KEYS].merge(
        lookup, on=KEYS, how="left", sort=False, validate="one_to_one",
    )
    if len(joined) != len(table) or joined[LOOKUP_COLUMNS].isna().any().any():
        raise RuntimeError("neighbour-flux lookup does not cover the self-response dump exactly")
    if not all(np.array_equal(joined[k].to_numpy(), table[k].to_numpy()) for k in KEYS):
        raise RuntimeError("neighbour-flux join changed row order")

    mag = table["r_input_p"].to_numpy(float)
    re_ = table["Re_input_p"].to_numpy(float)
    r_sim = table["r_sim_self"].to_numpy(float)
    flow = table[seed_cols].to_numpy(float)
    log_all = joined["logflux_all_0_10"].to_numpy(float)
    log_third = joined["logflux_thirdplus_0_10"].to_numpy(float)
    all_ratio = np.expm1(log_all * np.log(10.0))
    third_ratio = np.expm1(log_third * np.log(10.0))
    log_top2 = np.log10(1.0 + np.maximum(all_ratio - third_ratio, 0.0))

    finite = np.isfinite(r_sim) & np.isfinite(flow).all(axis=1)
    case = table["case"].to_numpy(np.int64)
    domain = ((mag < args.mag_max) & (re_ > args.re_min) & finite
              & (case >= args.min_case) & (case <= args.max_case))
    controls = {"primary mag": mag, "primary Re": re_, "top2 flux": log_top2}
    bins = {"primary mag": args.mag_bins, "primary Re": args.other_bins,
            "top2 flux": args.other_bins}
    indices = {name: quantile_index(value, domain, bins[name])[0]
               for name, value in controls.items()}
    cells = joint_cells(
        [indices[name] for name in controls], [bins[name] for name in controls], domain,
    )
    ids, info = within_cell_quartiles(log_third, cells, domain, min_cell=args.min_cell)

    print(f"V2.2 half-shear dump rows={len(table):,}; domain finite rows={domain.sum():,}; "
          f"case window=[{args.min_case},{args.max_case}]; "
          f"cases={np.unique(case[domain]).size}; seeds={len(seed_cols)}")
    print(f"Control quantile bins: {bins}")
    print(f"Conditional-cell retention: {info}; elapsed={time.time()-t0:.1f}s")
    balance_statistics(ids, controls, args.max_smd)
    report(ids, case, r_sim, flow, absolute=not args.contrasts_only)
    if args.reference_selfresp:
        ref = pf.read_table(
            args.reference_selfresp, columns=[*KEYS, "r_sim_self", *seed_cols],
        ).to_pandas()
        if ref.duplicated(KEYS).any():
            raise RuntimeError("reference self-response keys are not unique")
        ref = table[KEYS].merge(
            ref, on=KEYS, how="left", sort=False, validate="one_to_one",
        )
        if len(ref) != len(table) or ref[seed_cols].isna().any().any():
            raise RuntimeError("reference self-response does not cover candidate keys exactly")
        if not all(np.array_equal(ref[k].to_numpy(), table[k].to_numpy()) for k in KEYS):
            raise RuntimeError("reference join changed row order")
        ref_rsim = ref["r_sim_self"].to_numpy(float)
        if not np.allclose(ref_rsim, r_sim, rtol=0, atol=1e-7, equal_nan=True):
            finite_both = np.isfinite(ref_rsim) & np.isfinite(r_sim)
            maxdiff = float(np.max(np.abs(ref_rsim[finite_both] - r_sim[finite_both])))
            raise RuntimeError(
                f"candidate and reference simulation responses differ (max |delta|={maxdiff:.3g})"
            )
        compare_models(ids, r_sim, flow, ref[seed_cols].to_numpy(float))
    print("\nDIAG_V22_HALFSHEAR_THIRDPLUS_DONE", flush=True)


if __name__ == "__main__":
    main()
