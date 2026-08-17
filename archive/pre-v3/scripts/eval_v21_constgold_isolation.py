"""Test V2.1 flow closure on geometrically isolated constgold primaries.

Isolation is defined from the intrinsic input catalogue, not from the historical
``neighbored`` flag and not from a measured or emulator quantity.  For each
constgold primary, the nearest *other rendered source* is found in the matching
input field.  A radius ladder then reports

    m_flow = <R_sim> / <R_flow> - 1

using all 16 V2.1 flow seeds.  This is a valid flow-only test only where the
nearest-source radius is large enough that neighbour response is negligible.
The number of surviving objects and the native emulator ``R_blend`` are printed
so an underpowered or contaminated "isolated" sample cannot silently pass.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import time

import numpy as np
import pandas as pd
import pyarrow.feather as pf
from scipy.spatial import cKDTree

from sbs_shear import domain
from scripts.eval_v2_indomain_m import catalogue_true_props


EXPECTED_SEEDS = [501, 502, 503, 505, 506, 507, 508, 509,
                  510, 511, 512, 513, 514, 515, 516, 517]


def seed_from_path(path: str) -> int:
    match = re.search(r"_s(\d+)\.feather$", os.path.basename(path))
    if match is None:
        raise RuntimeError(f"cannot parse seed from {path}")
    return int(match.group(1))


def nearest_other_source(frame: pd.DataFrame, target_ids: np.ndarray, workers: int) -> np.ndarray:
    """Nearest intrinsic catalogue source other than the target itself, in arcsec."""
    source_ids = frame["index"].to_numpy(np.int64)
    if pd.Index(source_ids).has_duplicates:
        raise RuntimeError("raw input catalogue has duplicate source ids")
    rows = pd.Index(source_ids).get_indexer(np.asarray(target_ids, dtype=np.int64))
    if np.any(rows < 0):
        raise RuntimeError(f"{int((rows < 0).sum())} constgold target ids missing from raw input")

    ra = frame["RA"].to_numpy(float)
    dec = frame["DEC"].to_numpy(float)
    cosdec = np.cos(np.deg2rad(float(np.median(dec))))
    xy = np.column_stack((ra * cosdec * 3600.0, dec * 3600.0))
    k = min(4, len(frame))
    distances, neighbours = cKDTree(xy).query(xy[rows], k=k, workers=workers)
    if k == 1:
        raise RuntimeError("raw input contains only one source")
    if distances.ndim == 1:
        distances = distances[:, None]
        neighbours = neighbours[:, None]
    candidate_ids = source_ids[neighbours]
    other = candidate_ids != target_ids[:, None]
    if np.any(~other.any(axis=1)):
        raise RuntimeError("could not find a non-self neighbour in the four nearest sources")
    first = np.argmax(other, axis=1)
    return distances[np.arange(len(rows)), first]


def case_bootstrap_percent(
    case: np.ndarray,
    keep: np.ndarray,
    r_sim: np.ndarray,
    r_flow_ensemble: np.ndarray,
    n_boot: int,
    seed: int,
) -> float:
    """Case-bootstrap uncertainty of the object-weighted flow-only m."""
    selected_case = case[keep].astype(np.int64)
    if not len(selected_case):
        return float("nan")
    offset = int(selected_case.min())
    compact = selected_case - offset
    counts = np.bincount(compact)
    sim_sum = np.bincount(compact, weights=r_sim[keep])
    flow_sum = np.bincount(compact, weights=r_flow_ensemble[keep])
    active = counts > 0
    sim_sum, flow_sum = sim_sum[active], flow_sum[active]
    if len(sim_sum) < 2:
        return float("nan")
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot, dtype=float)
    for j in range(n_boot):
        take = rng.integers(0, len(sim_sum), len(sim_sum))
        draws[j] = 100.0 * (sim_sum[take].sum() / flow_sum[take].sum() - 1.0)
    return float(draws.std(ddof=1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--dump-glob",
        default=("/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/"
                 "v21_domain_dumps/ablate_s2c_lt500_v21_perobj_s*.feather"),
    )
    ap.add_argument(
        "--catalogue",
        default=("/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant/"
                 "constant_response_catalogue_train.feather"),
    )
    ap.add_argument(
        "--raw-base",
        default="/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant",
    )
    ap.add_argument("--min-case", type=int, default=40)
    ap.add_argument("--radii", type=float, nargs="+", default=[0, 3, 5, 7, 10, 15, 20])
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--bootstrap", type=int, default=20_000)
    ap.add_argument("--seed", type=int, default=25876)
    ap.add_argument("--output-json")
    args = ap.parse_args()
    started = time.time()

    paths = sorted(glob.glob(args.dump_glob), key=seed_from_path)
    seeds = [seed_from_path(path) for path in paths]
    if seeds != EXPECTED_SEEDS:
        raise RuntimeError(f"expected seeds {EXPECTED_SEEDS}, found {seeds}")

    truth = catalogue_true_props(args.catalogue, args.min_case, started)
    ref = pf.read_table(
        paths[0], columns=["case", "input_index", "r_sim", "R_flow", "R_blend"],
        memory_map=True,
    )
    case = ref["case"].to_numpy(zero_copy_only=False).astype(np.int32)
    input_index = ref["input_index"].to_numpy(zero_copy_only=False).astype(np.int64)
    truth_case = truth["case"].to_numpy(np.int32)
    truth_index = truth["input_index"].to_numpy(np.int64)
    if len(truth) != len(ref) or not np.array_equal(case, truth_case) \
            or not np.array_equal(input_index, truth_index):
        raise RuntimeError("constgold catalogue replay does not align exactly with V2.1 dumps")
    mag = truth["r_input_p"].to_numpy(float)
    re_arcsec = truth["Re_input_p"].to_numpy(float)
    v21 = domain.in_domain(mag, re_arcsec)
    if int(v21.sum()) != 5_226_377:
        raise RuntimeError(f"canonical V2.1 population drifted: {int(v21.sum()):,}")

    r_sim = ref["r_sim"].to_numpy(zero_copy_only=False).astype(np.float64)
    r_blend = ref["R_blend"].to_numpy(zero_copy_only=False).astype(np.float64)
    flows = []
    for path in paths:
        table = pf.read_table(
            path, columns=["case", "input_index", "r_sim", "R_flow", "R_blend"],
            memory_map=True,
        )
        c = table["case"].to_numpy(zero_copy_only=False)
        i = table["input_index"].to_numpy(zero_copy_only=False)
        if not np.array_equal(c, case) or not np.array_equal(i, input_index):
            raise RuntimeError(f"row identity/order differs in {path}")
        rs = table["r_sim"].to_numpy(zero_copy_only=False).astype(np.float64)
        rb = table["R_blend"].to_numpy(zero_copy_only=False).astype(np.float64)
        if not np.array_equal(rs, r_sim, equal_nan=True) \
                or not np.array_equal(rb, r_blend, equal_nan=True):
            raise RuntimeError(f"seed-invariant r_sim/R_blend changed in {path}")
        flows.append(table["R_flow"].to_numpy(zero_copy_only=False).astype(np.float32))
        print(f"loaded seed {seed_from_path(path)}", flush=True)

    nearest = np.full(len(ref), np.nan, dtype=np.float32)
    if np.any(np.diff(case) < 0):
        raise RuntimeError("V2.1 dump is not case-ordered; cannot stream intrinsic fields safely")
    for value in np.unique(case[v21]):
        lo = int(np.searchsorted(case, value, side="left"))
        hi = int(np.searchsorted(case, value, side="right"))
        selected = np.arange(lo, hi, dtype=np.int64)
        selected = selected[v21[selected]]
        raw_path = os.path.join(args.raw_base, f"gals{int(value)}_0.02.feather")
        raw = pf.read_table(raw_path, columns=["index", "RA", "DEC"], memory_map=True).to_pandas()
        nearest[selected] = nearest_other_source(raw, input_index[selected], args.workers)
        print(
            f"case {int(value)}: V2.1={len(selected):,}, "
            f"nearest median={float(np.median(nearest[selected])):.3f} arcsec",
            flush=True,
        )
    if not np.isfinite(nearest[v21]).all():
        raise RuntimeError("nearest-source calculation has non-finite V2.1 rows")

    flow_ensemble = np.zeros(len(ref), dtype=np.float64)
    for values in flows:
        flow_ensemble += values
    flow_ensemble /= len(flows)
    finite = np.isfinite(r_sim) & np.isfinite(r_blend) & np.isfinite(flow_ensemble)
    for values in flows:
        finite &= np.isfinite(values)

    results = []
    print("\nV2.1 CONSTGOLD -- INTRINSIC NEAREST-SOURCE ISOLATION LADDER")
    print(
        f"  {'nearest other':>15}{'N':>12}{'fraction':>11}{'<R_sim>':>11}"
        f"{'<R_flow>':>11}{'<R_blend>':>12}{'m_flow':>12}{'seed SEM':>11}"
        f"{'case boot':>12}{'combined':>11}"
    )
    for radius in args.radii:
        keep = v21 & finite & (nearest > float(radius))
        n = int(keep.sum())
        if not n:
            print(f"  >{radius:5.1f} arcsec{n:>12,}   EMPTY")
            results.append({"radius_arcsec": float(radius), "n": 0})
            continue
        sim_mean = float(r_sim[keep].mean())
        blend_mean = float(r_blend[keep].mean())
        flow_means = np.array([float(values[keep].mean()) for values in flows])
        m_seed = 100.0 * (sim_mean / flow_means - 1.0)
        m_mean = float(m_seed.mean())
        seed_sem = float(m_seed.std(ddof=1) / np.sqrt(len(m_seed)))
        case_boot = case_bootstrap_percent(
            case, keep, r_sim, flow_ensemble, args.bootstrap, args.seed + int(10 * radius),
        )
        combined = float(np.hypot(seed_sem, case_boot))
        print(
            f"  >{radius:5.1f} arcsec{n:>12,}{n/int(v21.sum()):>10.4%}"
            f"{sim_mean:>11.5f}{float(flow_means.mean()):>11.5f}{blend_mean:>12.5f}"
            f"{m_mean:>+11.3f}%{seed_sem:>10.3f}{case_boot:>12.3f}{combined:>11.3f}",
            flush=True,
        )
        results.append({
            "radius_arcsec": float(radius),
            "n": n,
            "fraction_v21": n / int(v21.sum()),
            "r_sim_mean": sim_mean,
            "r_flow_mean": float(flow_means.mean()),
            "r_blend_mean": blend_mean,
            "m_flow_percent": m_mean,
            "m_flow_seed_sem_percent": seed_sem,
            "m_flow_case_bootstrap_percent": case_boot if np.isfinite(case_boot) else None,
            "m_flow_combined_percent": combined if np.isfinite(combined) else None,
        })

    output = {
        "definition": "nearest other intrinsic rendered source farther than radius",
        "population": "canonical V2.1 constgold",
        "n_population": int(v21.sum()),
        "seeds": seeds,
        "results": results,
    }
    print("\n" + json.dumps(output, indent=2, sort_keys=True))
    if args.output_json:
        with open(args.output_json, "x", encoding="utf-8") as handle:
            json.dump(output, handle, indent=2, sort_keys=True)
            handle.write("\n")
    print(f"V21_CONSTGOLD_ISOLATION_DONE elapsed={time.time()-started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
