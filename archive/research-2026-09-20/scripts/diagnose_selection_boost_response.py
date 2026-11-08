"""Test whether the flow's self-response error tracks measurement boost.

The fixed-g0 cohort selects on *measured* magnitude (`MAG_AUTO < 25.8`) while
the flow conditions only on *true* properties.  Objects whose measured
magnitude is much brighter than their true magnitude are therefore
indistinguishable, to the flow, from unboosted objects at the same truth --
yet only the boosted ones enter the cohort in the faint truth bins.

This diagnostic bins the self-response residual `R_self_model - R_self_measured`
on a grid of true magnitude by boost (`r_input - MAG_AUTO`) and asks whether
the residual is a constant offset within a truth bin or slopes with boost.
A flat, nonzero residual points at the training objective; a sloping residual
points at the conditioning set.

It reports nothing about `m`: it audits one additive term, the flow
self-response, against its own measured labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

TRUTH_MAG_EDGES = (-np.inf, 23.0, 24.0, 25.0, 25.8, 26.5, np.inf)
BOOST_EDGES = (-np.inf, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, np.inf)
PIXEL_SCALE = 0.2
MAG_ZERO_POINT = 30.0


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_ready(value):
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def edge_labels(edges):
    labels = []
    for low, high in zip(edges[:-1], edges[1:]):
        lo = "-inf" if np.isneginf(low) else f"{low:g}"
        hi = "inf" if np.isposinf(high) else f"{high:g}"
        labels.append(f"({lo},{hi}]")
    return tuple(labels)


TRUTH_MAG_LABELS = edge_labels(TRUTH_MAG_EDGES)
BOOST_LABELS = edge_labels(BOOST_EDGES)


def match_keys(left_case, left_index, right_case, right_index):
    def key(case, index):
        return (np.asarray(case, dtype=np.uint64) << np.uint64(32)) | np.asarray(
            index, dtype=np.uint64
        )

    left = key(left_case, left_index)
    right = key(right_case, right_index)
    if len(np.unique(left)) != len(left) or len(np.unique(right)) != len(right):
        raise RuntimeError("response keys are not unique")
    _, li, ri = np.intersect1d(left, right, assume_unique=True, return_indices=True)
    order = np.argsort(left[li], kind="stable")
    return li[order], ri[order]


def measured_self_response(domain_root: Path, case: int) -> pd.DataFrame:
    """Finite-difference self-response from the matched g=0 and g=0.05 legs."""
    with np.load(domain_root / f"flow/g0/case{case:03d}.npz") as zero, np.load(
        domain_root / f"flow/g05/case{case:03d}.npz"
    ) as sheared:
        i0, ig = match_keys(
            zero["case"], zero["input_index"], sheared["case"], sheared["input_index"]
        )
        unmatched = {
            "g0_rows_without_g05": int(len(zero["case"]) - len(i0)),
            "g05_rows_without_g0": int(len(sheared["case"]) - len(ig)),
        }
        gamma = sheared["gamma"][ig].astype(np.float64)
        delta = (sheared["target"][ig, :2] - zero["target"][i0, :2]).astype(np.float64)
        h2 = np.einsum("ij,ij->i", gamma, gamma)
        keep = (
            np.isfinite(delta).all(axis=1) & np.isfinite(gamma).all(axis=1) & (h2 > 0)
        )
        unmatched["rejected_non_finite_or_zero_shear"] = int((~keep).sum())
        i0 = i0[keep]
        target0 = zero["target"][i0].astype(np.float64)
        frame = pd.DataFrame(
            {
                "case": np.full(int(keep.sum()), case, dtype=np.int32),
                "input_index": zero["input_index"][i0].astype(np.int64),
                "R_self_measured": np.einsum("ij,ij->i", delta[keep], gamma[keep])
                / h2[keep],
                "g0_MAG_AUTO": MAG_ZERO_POINT - 2.5 * np.log10(target0[:, 3]),
                "g0_FLUX_RADIUS_arcsec": target0[:, 2] * PIXEL_SCALE,
            }
        )
    return frame, unmatched


def attach_truth(frame: pd.DataFrame, input_path: Path) -> pd.DataFrame:
    truth = pd.read_feather(
        input_path, columns=["index_input", "r_input", "Re_input"]
    ).rename(
        columns={
            "index_input": "input_index",
            "r_input": "true_r_magnitude",
            "Re_input": "true_Re_semimajor_arcsec",
        }
    )
    merged = frame.merge(
        truth, on="input_index", how="left", validate="one_to_one", indicator=True
    )
    missing = int((merged["_merge"] != "both").sum())
    if missing:
        raise RuntimeError(f"{missing} self responses lack a truth row")
    return merged.drop(columns="_merge")


def attach_model(frame: pd.DataFrame, model: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    merged = frame.merge(
        model, on=["case", "input_index"], how="inner", validate="one_to_one"
    )
    dropped = len(frame) - len(merged)
    if not np.isfinite(merged["R_self_model"].to_numpy(np.float64)).all():
        raise RuntimeError("non-finite modelled self response")
    return merged, dropped


def assign_bins(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["boost"] = frame["true_r_magnitude"] - frame["g0_MAG_AUTO"]
    frame["truth_bin"] = pd.cut(
        frame["true_r_magnitude"], bins=list(TRUTH_MAG_EDGES), labels=list(TRUTH_MAG_LABELS)
    )
    frame["boost_bin"] = pd.cut(
        frame["boost"], bins=list(BOOST_EDGES), labels=list(BOOST_LABELS)
    )
    if frame["truth_bin"].isna().any() or frame["boost_bin"].isna().any():
        raise RuntimeError("a row fell outside the binning grid")
    frame["residual"] = frame["R_self_model"] - frame["R_self_measured"]
    return frame


def case_cell_sums(frame: pd.DataFrame) -> dict:
    """Per-case sums for every grid cell, marginal, and the whole cohort."""
    out = {}
    columns = ["R_self_measured", "R_self_model", "residual", "boost"]

    def record(key, part):
        if part.empty:
            return
        out[key] = {
            "n": int(len(part)),
            **{c: float(part[c].sum()) for c in columns},
            "boost_sq": float((part["boost"] ** 2).sum()),
            "residual_boost": float((part["residual"] * part["boost"]).sum()),
        }

    for truth_label, truth_part in frame.groupby("truth_bin", observed=True):
        record(f"truth={truth_label}", truth_part)
        for boost_label, cell in truth_part.groupby("boost_bin", observed=True):
            record(f"truth={truth_label}|boost={boost_label}", cell)
    for boost_label, boost_part in frame.groupby("boost_bin", observed=True):
        record(f"boost={boost_label}", boost_part)
    record("all", frame)
    return out


FIELDS = ("n", "R_self_measured", "R_self_model", "residual", "boost", "boost_sq",
          "residual_boost")


def stack_cases(per_case: list[dict]) -> tuple[tuple[str, ...], np.ndarray]:
    """Pack per-case cell sums into a dense (case, key, field) array."""
    keys = tuple(sorted({k for entry in per_case for k in entry}))
    index = {key: i for i, key in enumerate(keys)}
    table = np.zeros((len(per_case), len(keys), len(FIELDS)), dtype=np.float64)
    for c, entry in enumerate(per_case):
        for key, values in entry.items():
            table[c, index[key]] = [values[f] for f in FIELDS]
    return keys, table


def pooled_statistics(totals: np.ndarray) -> dict:
    """Means and the residual-on-boost slope from pooled sums, vectorized over keys."""
    n = totals[:, 0]
    present = n > 0
    safe_n = np.where(present, n, 1.0)
    mean_boost = totals[:, 4] / safe_n
    var_boost = totals[:, 5] / safe_n - mean_boost**2
    mean_residual = totals[:, 3] / safe_n
    cov = totals[:, 6] / safe_n - mean_residual * mean_boost
    with np.errstate(divide="ignore", invalid="ignore"):
        slope = np.where(var_boost > 0, cov / var_boost, np.nan)
    return {
        "present": present,
        "n": n,
        "mean_R_self_measured": totals[:, 1] / safe_n,
        "mean_R_self_model": totals[:, 2] / safe_n,
        "mean_residual": mean_residual,
        "mean_boost": mean_boost,
        "residual_boost_slope": slope,
    }


def combine(per_case: list[dict]) -> dict:
    keys, table = stack_cases(per_case)
    stats = pooled_statistics(table.sum(axis=0))
    cases_with_rows = (table[:, :, 0] > 0).sum(axis=0)
    summary = {}
    for i, key in enumerate(keys):
        if not stats["present"][i]:
            continue
        slope = stats["residual_boost_slope"][i]
        summary[key] = {
            "n": int(stats["n"][i]),
            "cases_with_rows": int(cases_with_rows[i]),
            "mean_R_self_measured": float(stats["mean_R_self_measured"][i]),
            "mean_R_self_model": float(stats["mean_R_self_model"][i]),
            "mean_residual": float(stats["mean_residual"][i]),
            "mean_boost": float(stats["mean_boost"][i]),
            "residual_boost_slope": None if np.isnan(slope) else float(slope),
        }
    return summary


def bootstrap(per_case: list[dict], n_boot: int, seed: int) -> dict:
    keys, table = stack_cases(per_case)
    rng = np.random.default_rng(seed)
    n_cases = table.shape[0]
    residual = np.full((n_boot, len(keys)), np.nan)
    slope = np.full((n_boot, len(keys)), np.nan)
    for b in range(n_boot):
        pick = rng.integers(0, n_cases, size=n_cases)
        stats = pooled_statistics(table[pick].sum(axis=0))
        residual[b] = np.where(stats["present"], stats["mean_residual"], np.nan)
        slope[b] = np.where(stats["present"], stats["residual_boost_slope"], np.nan)

    def spread(draws):
        counts = np.sum(~np.isnan(draws), axis=0)
        with np.errstate(invalid="ignore"):
            values = np.nanstd(draws, axis=0, ddof=1)
        return values, counts

    residual_se, residual_counts = spread(residual)
    slope_se, slope_counts = spread(slope)
    return {
        key: {
            "mean_residual": None
            if residual_counts[i] < 2
            else float(residual_se[i]),
            "residual_boost_slope": None
            if slope_counts[i] < 2
            else float(slope_se[i]),
            "bootstrap_draws_with_rows": int(residual_counts[i]),
        }
        for i, key in enumerate(keys)
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain-root", type=Path, required=True)
    parser.add_argument("--input-pattern", required=True)
    parser.add_argument("--self-model", type=Path, required=True)
    parser.add_argument("--case", type=int, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n-boot", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260915)
    return parser.parse_args()


def main():
    args = parse_args()
    import os

    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("run under the scheduler; SLURM_JOB_ID is unset")
    if args.output.exists():
        raise RuntimeError(f"refusing to overwrite {args.output}")

    model = pd.read_feather(args.self_model)
    per_case = []
    case_reports = {}
    for case in args.case:
        frame, unmatched = measured_self_response(args.domain_root, case)
        frame = attach_truth(frame, Path(args.input_pattern.format(case=case)))
        frame, dropped = attach_model(frame, model.loc[model["case"] == case])
        frame = assign_bins(frame)
        per_case.append(case_cell_sums(frame))
        case_reports[case] = {
            "rows": int(len(frame)),
            "rows_without_model": int(dropped),
            **unmatched,
            "mean_R_self_measured": float(frame["R_self_measured"].mean()),
            "mean_R_self_model": float(frame["R_self_model"].mean()),
            "mean_boost": float(frame["boost"].mean()),
        }
        print(
            f"SELECTION_BOOST_DONE case={case} rows={len(frame)} "
            f"measured={frame['R_self_measured'].mean():.6f} "
            f"model={frame['R_self_model'].mean():.6f} "
            f"boost={frame['boost'].mean():.4f}",
            flush=True,
        )

    summary = combine(per_case)
    errors = bootstrap(per_case, args.n_boot, args.bootstrap_seed)
    for key, entry in summary.items():
        entry["standard_error"] = errors.get(key, {})

    result = {
        "format_version": 1,
        "purpose": (
            "Bin the flow self-response residual on true magnitude by measurement "
            "boost, to separate a constant conditional-mean offset from a "
            "boost-dependent selection effect."
        ),
        "cases": list(args.case),
        "grid": {
            "true_r_magnitude_edges": [float(e) for e in TRUTH_MAG_EDGES],
            "boost_edges": [float(e) for e in BOOST_EDGES],
            "boost_definition": "true_r_magnitude - g0_MAG_AUTO",
        },
        "inputs": {
            "domain_root": str(args.domain_root),
            "self_model": str(args.self_model),
            "self_model_sha256": file_sha256(args.self_model),
        },
        "bootstrap": {"n_boot": args.n_boot, "seed": args.bootstrap_seed},
        "case_reports": case_reports,
        "summary": summary,
        "limitations": [
            "Audits the flow self-response only; R_blend and m are untouched.",
            "Case bootstrap over the requested cases only.",
            "Boost uses the g=0 leg MAG_AUTO, the same quantity the cohort selects on.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as handle:
        json.dump(json_ready(result), handle, indent=2, sort_keys=True)
    print(f"SELECTION_BOOST_COMPLETE output={args.output}", flush=True)


if __name__ == "__main__":
    main()
