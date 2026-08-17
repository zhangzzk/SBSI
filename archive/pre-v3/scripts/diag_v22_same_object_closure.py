"""Decompose V2.2 constgold closure on exact half-shear/constgold object keys.

This is an evaluation-only diagnostic.  It does not fit a correction or train a
model.  On the key intersection it evaluates the exact identity

    Rc + Rb - Rs = (Rh - Sh) + (Rc - Rh) + (Rb - (Rs - Sh)),

where ``c`` is the constgold antithetic flow extraction, ``h`` is the
half-shear forward extraction, ``Sh`` is direct half-shear self truth, ``Rb``
is the deployed blend lookup, and ``Rs`` is direct constgold total truth.

The three right-hand terms distinguish half-shear self closure, flow extraction
convention, and the remaining blend/estimand gap.  The last term must not be
called an emulator error without additional evidence: ``Rs-Sh`` also contains
differences between the two simulation/selection estimands.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.ipc as ipc


KEYS = ["case", "input_index"]
SCENE_FEATURES = [
    "r_input_p", "Re_input_p", "nbr_flux_near", "nbr_flux_far",
    "nbr_flux_max", "R_blend",
]


def read_many(pattern: str, columns: list[str], expected: int | None = None) -> pd.DataFrame:
    paths = sorted(glob.glob(pattern))
    if expected is not None and len(paths) != expected:
        raise RuntimeError(f"expected {expected} files for {pattern}, found {len(paths)}")
    if not paths:
        raise RuntimeError(f"no files for {pattern}")
    out = pd.concat([pd.read_feather(path, columns=columns) for path in paths], ignore_index=True)
    if out.duplicated(KEYS).any():
        raise RuntimeError(f"duplicate keys in {pattern}")
    return out


def case_filtered(path: str, columns: list[str], lo: int, hi: int) -> pd.DataFrame:
    parts = []
    with ipc.open_file(pa.memory_map(path)) as reader:
        missing = set(columns) - set(reader.schema.names)
        if missing:
            raise KeyError(f"{path} missing {sorted(missing)}")
        for i in range(reader.num_record_batches):
            table = pa.Table.from_batches([reader.get_batch(i)]).select(columns)
            table = table.filter(pc.and_(
                pc.greater_equal(table["case"], pa.scalar(lo)),
                pc.less(table["case"], pa.scalar(hi)),
            ))
            if table.num_rows:
                parts.append(table)
    if not parts:
        raise RuntimeError(f"no rows in {path} for cases [{lo},{hi})")
    out = pa.concat_tables(parts).to_pandas()
    if out.duplicated(KEYS).any():
        raise RuntimeError(f"duplicate keys in {path}")
    return out


def stable_sort(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sort_values(KEYS, kind="mergesort").reset_index(drop=True)


def mean_sem(values: list[float]) -> dict:
    x = np.asarray(values, dtype=float)
    return {
        "mean": float(x.mean()),
        "seed_sem": float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else None,
        "seed_values": x.tolist(),
    }


def case_summary(case: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    unique, inverse = np.unique(case, return_inverse=True)
    count = np.bincount(inverse)
    means = np.bincount(inverse, weights=values) / count
    return unique, means


def equality_summary(left: np.ndarray, right: np.ndarray) -> dict:
    left = np.asarray(left)
    right = np.asarray(right)
    finite = np.isfinite(left) & np.isfinite(right)
    diff = np.abs(left[finite].astype(float) - right[finite].astype(float))
    return {
        "n": int(len(left)),
        "finite_both_fraction": float(finite.mean()),
        "exact_equal_fraction": float(np.mean(left[finite] == right[finite])) if finite.any() else None,
        "mean_abs_difference": float(diff.mean()) if len(diff) else None,
        "max_abs_difference": float(diff.max()) if len(diff) else None,
    }


def decompose(rc: np.ndarray, rh: np.ndarray, rb: np.ndarray,
              rs: np.ndarray, sh: np.ndarray) -> dict[str, np.ndarray]:
    terms = {
        "total_model_minus_const_truth": rc + rb - rs,
        "halfshear_flow_minus_self_truth": rh - sh,
        "flow_extraction_const_minus_half": rc - rh,
        "blend_estimand_gap": rb - (rs - sh),
        "direct_const_minus_half_self": rs - sh,
    }
    reconstructed = (
        terms["halfshear_flow_minus_self_truth"]
        + terms["flow_extraction_const_minus_half"]
        + terms["blend_estimand_gap"]
    )
    terms["identity_error"] = terms["total_model_minus_const_truth"] - reconstructed
    return terms


def parse_seed(path: str) -> int:
    match = re.search(r"_s(\d+)(?:_|\.)", os.path.basename(path))
    if not match:
        raise ValueError(f"cannot parse seed from {path}")
    return int(match.group(1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--const-domain-glob", required=True,
                    help="ten already-scored domain shards; defines the exact constgold population")
    ap.add_argument("--const-primary-features", required=True)
    ap.add_argument("--const-crowd-lookup", required=True)
    ap.add_argument("--const-dump-template", required=True,
                    help="template containing {seed} for full V2.2 constgold dumps")
    ap.add_argument("--half-features", required=True)
    ap.add_argument("--half-selfresp", required=True,
                    help="direct half-shear dump containing r_sim_self and R_flow_s<seed>")
    ap.add_argument("--seeds", nargs="+", type=int, required=True)
    ap.add_argument("--min-case", type=int, default=40)
    ap.add_argument("--max-case", type=int, default=140)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    if os.path.exists(args.output):
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("seeds must be unique")

    print("Loading the exact current constgold population", flush=True)
    const = read_many(
        args.const_domain_glob,
        KEYS + ["r_input_p", "r_sim", "R_blend"], expected=10,
    ).rename(columns={
        "r_input_p": "r_input_p_const",
        "R_blend": "R_blend_const",
    })
    const = const.loc[(const["case"] >= args.min_case) & (const["case"] < args.max_case)].copy()
    primary = pd.read_feather(
        args.const_primary_features, columns=KEYS + ["Re_input_p"]
    ).rename(columns={"Re_input_p": "Re_input_p_const"})
    const = const.merge(primary, on=KEYS, how="left", validate="one_to_one", sort=False)
    del primary
    crowd = case_filtered(
        args.const_crowd_lookup,
        KEYS + ["nbr_flux_near", "nbr_flux_far", "nbr_flux_max"],
        args.min_case, args.max_case,
    ).rename(columns={name: f"{name}_const" for name in SCENE_FEATURES[2:5]})
    const = const.merge(crowd, on=KEYS, how="left", validate="one_to_one", sort=False)
    del crowd
    if const.isna().any().any():
        raise RuntimeError(f"incomplete constgold features: {const.isna().mean().to_dict()}")
    const = stable_sort(const)

    print("Loading half-shear truth and scene summaries", flush=True)
    half = pd.read_feather(
        args.half_features,
        columns=KEYS + ["r_input_p", "Re_input_p", "nbr_flux_near", "nbr_flux_far",
                        "nbr_flux_max", "r_blend"],
    ).rename(columns={
        "r_input_p": "r_input_p_half", "Re_input_p": "Re_input_p_half",
        "nbr_flux_near": "nbr_flux_near_half", "nbr_flux_far": "nbr_flux_far_half",
        "nbr_flux_max": "nbr_flux_max_half", "r_blend": "R_blend_half",
    })
    half = half.loc[(half["case"] >= args.min_case) & (half["case"] < args.max_case)].copy()
    if half.duplicated(KEYS).any():
        raise RuntimeError("duplicate half-shear feature keys")
    half_flow_columns = [f"R_flow_s{seed}" for seed in args.seeds]
    half_scores = pd.read_feather(
        args.half_selfresp, columns=KEYS + ["r_sim_self"] + half_flow_columns
    )
    half_scores = half_scores.loc[
        (half_scores["case"] >= args.min_case) & (half_scores["case"] < args.max_case)
    ].copy()
    if half_scores.duplicated(KEYS).any():
        raise RuntimeError("duplicate half-shear self-response keys")
    half = half.merge(half_scores, on=KEYS, how="inner", validate="one_to_one", sort=False)
    del half_scores
    half_before_finite = len(half)
    response_finite = np.isfinite(
        half[["r_sim_self"] + half_flow_columns].to_numpy(float)
    ).all(axis=1)
    half = half.loc[response_finite].copy()
    half_nonfinite_removed = int(half_before_finite - len(half))
    half = stable_sort(half)

    paired = const.merge(half, on=KEYS, how="inner", validate="one_to_one", sort=False)
    paired = stable_sort(paired)
    if paired.empty:
        raise RuntimeError("no shared half-shear/constgold keys")
    print(
        f"  const={len(const):,}; half={len(half):,}; shared={len(paired):,} "
        f"({len(paired)/len(const):.2%} const, {len(paired)/len(half):.2%} half)",
        flush=True,
    )

    comparisons = {}
    for name in SCENE_FEATURES:
        comparisons[name] = equality_summary(
            paired[f"{name}_const"].to_numpy(), paired[f"{name}_half"].to_numpy()
        )

    pair_keys = paired[KEYS]
    case = paired["case"].to_numpy(np.int64)
    rs = paired["r_sim"].to_numpy(float)
    rb = paired["R_blend_const"].to_numpy(float)
    sh = paired["r_sim_self"].to_numpy(float)
    all_rs = const["r_sim"].to_numpy(float)
    all_rb = const["R_blend_const"].to_numpy(float)

    per_seed = []
    case_terms: dict[str, list[np.ndarray]] = {}
    reference_cases = None
    max_identity_error = 0.0
    for seed in args.seeds:
        print(f"Loading seed {seed}", flush=True)
        hcol = f"R_flow_s{seed}"
        rh = paired[hcol].to_numpy(float)
        rh_all = half[hcol].to_numpy(float)
        sh_all = half["r_sim_self"].to_numpy(float)

        cpath = args.const_dump_template.format(seed=seed)
        cd = pd.read_feather(cpath, columns=KEYS + ["r_sim", "R_flow", "R_blend"])
        cd_domain = const[KEYS].merge(cd, on=KEYS, how="left", validate="one_to_one", sort=False)
        if cd_domain[["r_sim", "R_flow", "R_blend"]].isna().any().any():
            raise RuntimeError(f"incomplete constgold domain join for seed {seed}")
        if not np.allclose(cd_domain["r_sim"].to_numpy(float), all_rs, equal_nan=True):
            raise RuntimeError(f"constgold R_sim mismatch for seed {seed}")
        if not np.allclose(cd_domain["R_blend"].to_numpy(float), all_rb, equal_nan=True):
            raise RuntimeError(f"constgold R_blend mismatch for seed {seed}")
        rc_all = cd_domain["R_flow"].to_numpy(float)
        rc_pair_frame = pair_keys.merge(
            cd_domain[KEYS + ["R_flow"]], on=KEYS, how="left",
            validate="one_to_one", sort=False,
        )
        rc = rc_pair_frame["R_flow"].to_numpy(float)
        del cd, cd_domain, rc_pair_frame

        terms = decompose(rc, rh, rb, rs, sh)
        max_identity_error = max(max_identity_error, float(np.max(np.abs(terms["identity_error"]))))
        case_row = {}
        for name, values in terms.items():
            uc, cm = case_summary(case, values)
            if reference_cases is None:
                reference_cases = uc
            elif not np.array_equal(uc, reference_cases):
                raise RuntimeError("case set changed between terms/seeds")
            case_terms.setdefault(name, []).append(cm)
            case_row[name] = float(values.mean())

        m_all = float(all_rs.mean() / (rc_all.mean() + all_rb.mean()) - 1.0)
        m_pair = float(rs.mean() / (rc.mean() + rb.mean()) - 1.0)
        m_direct_self_hybrid = float(rs.mean() / (sh.mean() + rb.mean()) - 1.0)
        direct_neighbor = rs - sh
        m_direct_neighbor_hybrid = float(rs.mean() / (rc.mean() + direct_neighbor.mean()) - 1.0)
        per_seed.append({
            "seed": seed,
            "n_all_const": int(len(const)),
            "n_paired": int(len(paired)),
            "R_sim_all_const": float(all_rs.mean()),
            "R_blend_all_const": float(all_rb.mean()),
            "R_flow_all_const": float(rc_all.mean()),
            "m_all_const": m_all,
            "R_sim_paired": float(rs.mean()),
            "R_self_half_paired": float(sh.mean()),
            "R_blend_paired": float(rb.mean()),
            "R_flow_const_paired": float(rc.mean()),
            "R_flow_half_paired": float(rh.mean()),
            "R_self_half_all": float(sh_all.mean()),
            "R_flow_half_all": float(rh_all.mean()),
            "halfshear_flow_over_self_minus_one_all": float(
                rh_all.mean() / sh_all.mean() - 1.0
            ),
            "m_paired": m_pair,
            "m_direct_halfself_plus_modelblend": m_direct_self_hybrid,
            "m_modelconstflow_plus_directdifference": m_direct_neighbor_hybrid,
            "terms": case_row,
        })
        print(
            f"  m all={100*m_all:+.3f}% paired={100*m_pair:+.3f}%; "
            f"terms total/self/extract/blend-gap="
            f"{case_row['total_model_minus_const_truth']:+.6f}/"
            f"{case_row['halfshear_flow_minus_self_truth']:+.6f}/"
            f"{case_row['flow_extraction_const_minus_half']:+.6f}/"
            f"{case_row['blend_estimand_gap']:+.6f}",
            flush=True,
        )

    term_summary = {}
    for name, arrays in case_terms.items():
        by_seed = np.stack(arrays)
        seed_means = by_seed.mean(axis=1)
        ensemble_case = by_seed.mean(axis=0)
        term_summary[name] = {
            **mean_sem(seed_means.tolist()),
            "case_sem_of_seed_ensemble": (
                float(ensemble_case.std(ddof=1) / np.sqrt(len(ensemble_case)))
                if len(ensemble_case) > 1 else None
            ),
            "n_cases": int(len(ensemble_case)),
        }

    result = {
        "status": "diagnostic only; no correction fitted or applied",
        "interpretation_guard": (
            "blend_estimand_gap compares deployed R_blend with constgold total minus "
            "half-shear self truth. It includes emulator error, simulation/selection-estimand "
            "differences, and any non-additivity; it is not uniquely an emulator residual."
        ),
        "seeds": args.seeds,
        "population": {
            "n_constgold": int(len(const)),
            "n_halfshear_in_case_range": int(len(half)),
            "n_halfshear_nonfinite_response_removed": half_nonfinite_removed,
            "n_shared": int(len(paired)),
            "constgold_shared_fraction": float(len(paired) / len(const)),
            "halfshear_shared_fraction": float(len(paired) / len(half)),
            "cases": [int(x) for x in np.unique(case)],
        },
        "same_key_feature_comparisons": comparisons,
        "per_seed": per_seed,
        "ensemble": {
            "m_all_const": mean_sem([row["m_all_const"] for row in per_seed]),
            "m_paired": mean_sem([row["m_paired"] for row in per_seed]),
            "m_direct_halfself_plus_modelblend": mean_sem(
                [row["m_direct_halfself_plus_modelblend"] for row in per_seed]
            ),
            "m_modelconstflow_plus_directdifference": mean_sem(
                [row["m_modelconstflow_plus_directdifference"] for row in per_seed]
            ),
            "terms": term_summary,
            "max_abs_identity_error": max_identity_error,
        },
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(f"Wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
