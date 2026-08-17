"""Does third-plus self-response failure survive control for every V2.2 flow input?

The parent diagnostic balances primary magnitude/size and top-two neighbour
flux. This stricter follow-up additionally balances the flow's deployed scene
summaries (0--3 arcsec flux, 3--7 arcsec flux, and brightest-neighbour flux).
Primary Sersic index and intrinsic e1/e2 are not cell axes because they should
be independent of scene density, but their balance is measured and subjected
to the same max-SMD gate. No response is fit and no constgold quantity enters.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pyarrow.feather as pf

from scripts.diag_v22_halfshear_thirdplus import (
    EXPECTED_SEEDS,
    KEYS,
    LOOKUP_COLUMNS,
    balance_statistics,
    report,
)
from scripts.diag_v22_thirdplus_conditional import (
    joint_cells,
    quantile_index,
    within_cell_quartiles,
)


BASE_COLUMNS = [
    *KEYS,
    "sersic_n_input_p",
    "e1_input_rot0_p",
    "e2_input_rot0_p",
    "nbr_flux_near",
    "nbr_flux_far",
    "nbr_flux_max",
]


def analyse_window(
    table,
    scene,
    log_top2: np.ndarray,
    log_third: np.ndarray,
    flow: np.ndarray,
    min_case: int,
    max_case: int,
    args,
) -> None:
    mag = table["r_input_p"].to_numpy(float)
    re_ = table["Re_input_p"].to_numpy(float)
    r_sim = table["r_sim_self"].to_numpy(float)
    case = table["case"].to_numpy(np.int64)
    finite = np.isfinite(r_sim) & np.isfinite(flow).all(axis=1)
    domain = ((mag < args.mag_max) & (re_ > args.re_min) & finite
              & (case >= min_case) & (case <= max_case))

    joint = {
        "primary mag": mag,
        "primary Re": re_,
        "top2 flux": log_top2,
        "near flux input": scene["nbr_flux_near"].to_numpy(float),
        "far flux input": scene["nbr_flux_far"].to_numpy(float),
        "max flux input": scene["nbr_flux_max"].to_numpy(float),
    }
    bins = {name: args.other_bins for name in joint}
    bins["primary mag"] = args.mag_bins
    bins["top2 flux"] = args.top2_bins
    bins["far flux input"] = args.far_bins
    indices = {name: quantile_index(value, domain, bins[name])[0]
               for name, value in joint.items()}
    cells = joint_cells(
        [indices[name] for name in joint], [bins[name] for name in joint], domain,
    )
    ids, info = within_cell_quartiles(log_third, cells, domain, min_cell=args.min_cell)
    auxiliary = {
        "primary sersic": scene["sersic_n_input_p"].to_numpy(float),
        "primary e1": scene["e1_input_rot0_p"].to_numpy(float),
        "primary e2": scene["e2_input_rot0_p"].to_numpy(float),
    }

    print(f"\n=== CASE WINDOW [{min_case},{max_case}] ===")
    print(f"domain finite rows={domain.sum():,}; cases={np.unique(case[domain]).size}; "
          f"control bins={bins}; retention={info}")
    balance_statistics(ids, {**joint, **auxiliary}, args.max_smd)
    report(ids, case, r_sim, flow)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selfresp", required=True)
    ap.add_argument("--base-cache", required=True)
    ap.add_argument("--flux-lookup", required=True)
    ap.add_argument("--mag-max", type=float, default=25.8)
    ap.add_argument("--re-min", type=float, default=0.5)
    ap.add_argument("--mag-bins", type=int, default=8)
    ap.add_argument("--other-bins", type=int, default=4)
    ap.add_argument("--top2-bins", type=int, default=8)
    ap.add_argument("--far-bins", type=int, default=8)
    ap.add_argument("--min-cell", type=int, default=50)
    ap.add_argument("--max-smd", type=float, default=0.1)
    args = ap.parse_args()
    t0 = time.time()

    table = pf.read_table(args.selfresp).to_pandas()
    seed_cols = [f"R_flow_s{s}" for s in EXPECTED_SEEDS]
    need = [*KEYS, "r_sim_self", "r_input_p", "Re_input_p", *seed_cols]
    missing = [name for name in need if name not in table]
    if missing:
        raise RuntimeError(f"self-response dump is missing columns: {missing}")
    scene = pf.read_table(args.base_cache, columns=BASE_COLUMNS).to_pandas()
    if len(scene) != len(table) or not all(
        np.array_equal(scene[k].to_numpy(), table[k].to_numpy()) for k in KEYS
    ):
        raise RuntimeError("base cache and self-response dump do not align elementwise")

    lookup = pf.read_table(
        args.flux_lookup, columns=[*KEYS, *LOOKUP_COLUMNS],
    ).to_pandas()
    if lookup.duplicated(KEYS).any():
        raise RuntimeError("neighbour-flux lookup keys are not unique")
    joined = table[KEYS].merge(
        lookup, on=KEYS, how="left", sort=False, validate="one_to_one",
    )
    if len(joined) != len(table) or joined[LOOKUP_COLUMNS].isna().any().any():
        raise RuntimeError("neighbour-flux lookup does not cover the dump exactly")
    if not all(np.array_equal(joined[k].to_numpy(), table[k].to_numpy()) for k in KEYS):
        raise RuntimeError("neighbour-flux join changed row order")

    log_all = joined["logflux_all_0_10"].to_numpy(float)
    log_third = joined["logflux_thirdplus_0_10"].to_numpy(float)
    all_ratio = np.expm1(log_all * np.log(10.0))
    third_ratio = np.expm1(log_third * np.log(10.0))
    log_top2 = np.log10(1.0 + np.maximum(all_ratio - third_ratio, 0.0))
    flow = table[seed_cols].to_numpy(float)
    print(f"loaded {len(table):,} half-shear rows x {len(seed_cols)} seeds in "
          f"{time.time()-t0:.1f}s")

    # The parent, lower-dimensional match already established the target-overlap
    # and held-out replication. This sparse exact-input match uses all 160 cases.
    for min_case, max_case in ((40, 199),):
        analyse_window(table, scene, log_top2, log_third, flow,
                       min_case, max_case, args)
    print("\nDIAG_V22_HALFSHEAR_FLOWINPUTS_DONE", flush=True)


if __name__ == "__main__":
    main()
