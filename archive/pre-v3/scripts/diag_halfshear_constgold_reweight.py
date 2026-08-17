"""Reweight half-shear truth and flow residuals to the constgold feature population.

This is a population-transfer diagnostic, not a calibration.  It asks whether the
half-shear residual ``R_flow - R_self,true`` becomes the constgold closure residual
after changing only the occupancy of measured response-grid cells.  The constgold
closure residual is ``R_flow + R_blend - R_sim`` and therefore is not flow-only.

All response values come from existing score dumps.  No model is trained or scored.
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd


KEYS = ["case", "input_index"]


def read_unique(paths: list[str], columns: list[str]) -> pd.DataFrame:
    frame = pd.concat(
        [pd.read_feather(path, columns=columns) for path in sorted(paths)],
        ignore_index=True,
    )
    if frame.duplicated(KEYS).any():
        raise RuntimeError(f"duplicate keys in {len(paths)} input files")
    return frame


def attach_score(features: pd.DataFrame, path: str, flow_column: str) -> pd.DataFrame:
    score = pd.read_feather(path, columns=KEYS + ["r_sim_self", flow_column])
    out = features.merge(score, on=KEYS, how="inner", validate="one_to_one")
    if len(out) != len(features) or len(out) != len(score):
        raise RuntimeError(f"incomplete half-shear join for {path}: {len(out):,} rows")
    return out


def cell_index(
    mag: np.ndarray,
    size: np.ndarray,
    crowd: np.ndarray,
    mag_edges: np.ndarray,
    size_edges: np.ndarray,
    crowd_edges: np.ndarray | None,
) -> tuple[np.ndarray, int]:
    """Apply the same edge clipping and conditional crowd edges as target building."""
    nm, ns = len(mag_edges) - 1, len(size_edges) - 1
    fi = np.clip(np.digitize(mag, mag_edges) - 1, 0, nm - 1)
    si = np.clip(np.digitize(size, size_edges) - 1, 0, ns - 1)
    if crowd_edges is None:
        return (fi * ns + si).astype(np.int32), nm * ns
    if crowd_edges.ndim == 1:
        nc = len(crowd_edges) - 1
        ci = np.clip(np.digitize(crowd, crowd_edges) - 1, 0, nc - 1)
    elif crowd_edges.ndim == 3 and crowd_edges.shape[:2] == (nm, ns):
        nc = crowd_edges.shape[2] - 1
        ci = np.empty(len(crowd), dtype=np.int32)
        for f in range(nm):
            for s in range(ns):
                use = (fi == f) & (si == s)
                ci[use] = np.clip(
                    np.digitize(crowd[use], crowd_edges[f, s]) - 1, 0, nc - 1
                )
    else:
        raise ValueError(f"unsupported crowd edge shape {crowd_edges.shape}")
    return ((fi * ns + si) * nc + ci).astype(np.int32), nm * ns * nc


def summarize_reweight(values: np.ndarray, ih: np.ndarray, ic: np.ndarray, ncell: int) -> dict:
    finite = np.isfinite(values)
    nh = np.bincount(ih[finite], minlength=ncell).astype(np.int64)
    nc = np.bincount(ic, minlength=ncell).astype(np.int64)
    total = np.bincount(ih[finite], weights=values[finite], minlength=ncell)
    common = (nh > 0) & (nc > 0)
    cell_mean = np.divide(total, nh, out=np.full(ncell, np.nan), where=nh > 0)
    own = float(values[finite].mean())
    rw = float(np.sum(cell_mean[common] * nc[common]) / np.sum(nc[common]))
    ratio = np.divide(
        nc[common] / nc[common].sum(),
        nh[common] / nh[common].sum(),
    )
    return {
        "own_mean": own,
        "constgold_reweighted_mean": rw,
        "population_shift": rw - own,
        "n_half": int(finite.sum()),
        "n_constgold": int(len(ic)),
        "n_cells": int(ncell),
        "n_common_cells": int(common.sum()),
        "halfshear_common_fraction": float(nh[common].sum() / nh.sum()),
        "constgold_common_fraction": float(nc[common].sum() / nc.sum()),
        "weight_ratio_median": float(np.median(ratio)),
        "weight_ratio_p95": float(np.quantile(ratio, 0.95)),
        "weight_ratio_max": float(ratio.max()),
        "min_half_cell": int(nh[common].min()),
        "min_constgold_cell": int(nc[common].min()),
    }


def grid_specs(args: argparse.Namespace) -> list[dict]:
    base = np.load(args.target_baseline)
    s4 = np.load(args.target_size4)
    m10 = np.load(args.target_mag10)
    m12 = np.load(args.target_mag12)

    def spec(name: str, mag, size, crowd=None) -> dict:
        return {
            "name": name,
            "mag": np.asarray(mag, float),
            "size": np.asarray(size, float),
            "crowd": None if crowd is None else np.asarray(crowd, float),
        }

    # The hybrid fine grids use the no-size conditional R_blend edges within
    # each magnitude bin, repeated over four size bins.  This probes all three
    # continuous variables more finely without estimating any new edges.
    size4 = np.asarray(s4["edges_size"], float)
    c10 = np.repeat(np.asarray(m10["edges_crowd"], float), len(size4) - 1, axis=1)
    c12 = np.repeat(np.asarray(m12["edges_crowd"], float), len(size4) - 1, axis=1)
    return [
        spec("mag6", base["edges_flux"], [base["edges_size"][0], base["edges_size"][-1]]),
        spec("mag6_size6", base["edges_flux"], base["edges_size"]),
        spec("mag6_size6_rblend5", base["edges_flux"], base["edges_size"], base["edges_crowd"]),
        spec("mag6_size4_rblend8", s4["edges_flux"], size4, s4["edges_crowd"]),
        spec("mag10_size4_rblend16", m10["edges_flux"], size4, c10),
        spec("mag12_size4_rblend16", m12["edges_flux"], size4, c12),
    ]


def mean_sem(values: list[float]) -> dict:
    a = np.asarray(values, float)
    return {
        "mean": float(a.mean()),
        "seed_sem": float(a.std(ddof=1) / np.sqrt(len(a))) if len(a) > 1 else None,
        "seed_values": a.tolist(),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features-half", required=True)
    ap.add_argument("--features-const", required=True)
    ap.add_argument("--constgold-glob", required=True)
    ap.add_argument("--baseline-template", required=True)
    ap.add_argument("--variant-dir", required=True)
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, required=True)
    ap.add_argument("--target-baseline", required=True)
    ap.add_argument("--target-size4", required=True)
    ap.add_argument("--target-mag10", required=True)
    ap.add_argument("--target-mag12", required=True)
    ap.add_argument("--closure-json", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    if os.path.exists(args.output):
        raise FileExistsError(f"refusing to overwrite {args.output}")

    print("Loading half-shear conditioner features", flush=True)
    half_features = pd.read_feather(
        args.features_half, columns=KEYS + ["r_input_p", "Re_input_p", "r_blend"]
    )
    if half_features.duplicated(KEYS).any():
        raise RuntimeError("duplicate half-shear feature keys")

    first_path = args.baseline_template.format(seed=args.seeds[0])
    first = attach_score(half_features, first_path, f"R_flow_s{args.seeds[0]}")
    truth = first["r_sim_self"].to_numpy(float)

    print("Loading constgold population", flush=True)
    const_paths = glob.glob(args.constgold_glob)
    if len(const_paths) != 10:
        raise RuntimeError(f"expected ten constgold shards, found {len(const_paths)}")
    const_score = read_unique(const_paths, KEYS + ["R_blend"])
    const_features = pd.read_feather(
        args.features_const, columns=KEYS + ["r_input_p", "Re_input_p"]
    )
    const = const_score.merge(const_features, on=KEYS, how="left", validate="one_to_one")
    if const[["r_input_p", "Re_input_p"]].isna().any().any():
        raise RuntimeError("incomplete constgold primary-feature join")

    hm = half_features["r_input_p"].to_numpy(float)
    hs = half_features["Re_input_p"].to_numpy(float)
    hr = half_features["r_blend"].to_numpy(float)
    cm = const["r_input_p"].to_numpy(float)
    cs = const["Re_input_p"].to_numpy(float)
    cr = const["R_blend"].to_numpy(float)
    if not all(np.isfinite(x).all() for x in (hm, hs, hr, cm, cs, cr)):
        raise RuntimeError("non-finite conditioning feature")

    model_values: dict[str, list[tuple[int, np.ndarray]]] = {"v22": []}
    for arm in args.arms:
        model_values[arm] = []
    for seed in args.seeds:
        print(f"Loading half-shear scores for seed {seed}", flush=True)
        bpath = args.baseline_template.format(seed=seed)
        b = attach_score(half_features, bpath, f"R_flow_s{seed}")
        if not np.array_equal(b[KEYS].to_numpy(), first[KEYS].to_numpy()):
            raise RuntimeError("baseline key order changed after deterministic join")
        if not np.allclose(b["r_sim_self"], truth, equal_nan=True):
            raise RuntimeError("half-shear truth differs by seed")
        model_values["v22"].append((seed, b[f"R_flow_s{seed}"].to_numpy(float) - truth))
        del b
        for arm in args.arms:
            path = os.path.join(args.variant_dir, f"{arm}_s{seed}.feather")
            v = attach_score(half_features, path, f"R_flow_s{seed}")
            if not np.array_equal(v[KEYS].to_numpy(), first[KEYS].to_numpy()):
                raise RuntimeError(f"variant key order changed for {arm} seed {seed}")
            model_values[arm].append((seed, v[f"R_flow_s{seed}"].to_numpy(float) - truth))
            del v

    with open(args.closure_json, encoding="utf-8") as handle:
        closure = json.load(handle)
    result = {
        "status": "diagnostic only; no correction fitted or applied",
        "question": (
            "Does constgold occupancy in (magnitude,size,R_blend) turn the half-shear "
            "flow-minus-self residual into the constgold total closure residual?"
        ),
        "seeds": args.seeds,
        "constgold_reference": {},
        "grids": {},
    }
    # Four-seed total closure residuals reconstructed with the exact bin counts
    # from the existing constgold screen.
    for arm in args.arms:
        bins = closure["constgold"][arm]["profile_by_R_blend"]["bins"]
        denom = sum(row["n"] for row in bins)
        result["constgold_reference"][arm] = {
            "total_closure_residual": float(
                sum(row["n"] * row["flow_minus_demand_variant"]["mean"] for row in bins)
                / denom
            ),
            "m": closure["constgold"][arm]["m_variant"],
        }
    bins = closure["constgold"][args.arms[0]]["profile_by_R_blend"]["bins"]
    denom = sum(row["n"] for row in bins)
    result["constgold_reference"]["v22"] = {
        "total_closure_residual": float(
            sum(row["n"] * row["flow_minus_demand_baseline"]["mean"] for row in bins)
            / denom
        )
    }

    for spec in grid_specs(args):
        name = spec["name"]
        print(f"Reweighting on {name}", flush=True)
        ih, ncell = cell_index(hm, hs, hr, spec["mag"], spec["size"], spec["crowd"])
        ic, ncell_c = cell_index(cm, cs, cr, spec["mag"], spec["size"], spec["crowd"])
        if ncell_c != ncell:
            raise RuntimeError("cell count mismatch")
        truth_summary = summarize_reweight(truth, ih, ic, ncell)
        models = {}
        for model, seed_arrays in model_values.items():
            per_seed = []
            for seed, residual in seed_arrays:
                row = summarize_reweight(residual, ih, ic, ncell)
                row["seed"] = seed
                per_seed.append(row)
            models[model] = {
                "own_mean": mean_sem([row["own_mean"] for row in per_seed]),
                "constgold_reweighted_mean": mean_sem(
                    [row["constgold_reweighted_mean"] for row in per_seed]
                ),
                "population_shift": mean_sem([row["population_shift"] for row in per_seed]),
                "per_seed": per_seed,
            }
        result["grids"][name] = {"truth": truth_summary, "models": models}
        print(
            f"  truth shift={truth_summary['population_shift']:+.6f}; "
            + "; ".join(
                f"{model} residual {models[model]['own_mean']['mean']:+.6f} -> "
                f"{models[model]['constgold_reweighted_mean']['mean']:+.6f}"
                for model in models
            ),
            flush=True,
        )

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
    print(f"Wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
