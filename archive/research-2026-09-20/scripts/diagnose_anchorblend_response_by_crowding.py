#!/usr/bin/env python3
"""Read the emulator's blending accuracy as a function of how crowded a primary is.

Input is the per-primary dump from ``dump_anchorblend_scene_primaries.py``.
Because the anchor is unsheared and every neighbour is sheared, the measured
anchor response *is* the blending response, so ``m = R_sim / R_emulator - 1``
here tests the summed emulator on its own, with no flow term anywhere in it.

Primaries are binned on a predicted column -- ``R_scene_abs`` by default, the
sum of per-pair response magnitudes, because the signed sum reaches zero by
cancellation rather than by blending being absent.  The bin edges are global
quantiles of the weighted population, so the bins are a fixed partition of the
cohort and not re-chosen per case.

Each bin's response is formed per simulation case (pooling the two anchor
strata with their inverse-sampling weights), then averaged over cases with
equal weight and bootstrapped over cases -- the same estimator and the same
error unit the production run used.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

H = 0.02
LEGS = ("plus", "minus")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parts", type=Path, required=True,
                        help="directory of case{case}_stratum{stratum}.feather")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--branch", choices=("matched", "unmatched"), default="matched")
    parser.add_argument("--bin-column", default="R_scene_abs")
    parser.add_argument("--bins", type=int, default=8)
    parser.add_argument("--edge", type=float, action="append", default=None,
                        help="explicit inner bin edge; repeat. Overrides --bins")
    parser.add_argument("--primary-magnitude-min", type=float, default=None,
                        help="keep primaries with true r_input_p >= this")
    parser.add_argument("--primary-magnitude-max", type=float, default=None,
                        help="keep primaries with true r_input_p < this")
    parser.add_argument("--replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260915)
    return parser.parse_args(argv)


def load(parts: Path) -> pd.DataFrame:
    files = sorted(parts.glob("case*_stratum*.feather"))
    if not files:
        raise FileNotFoundError(f"no parts under {parts}")
    frame = pd.concat((pd.read_feather(path) for path in files), ignore_index=True)
    if frame.duplicated(["case", "stratum", "input_index"]).any():
        raise RuntimeError("a primary appears twice in the same case and stratum")
    return frame


def bin_edges(values: np.ndarray, weights: np.ndarray, bins: int) -> np.ndarray:
    """Weighted quantile edges, open at both ends."""
    if bins < 3:
        raise ValueError("use at least three bins")
    order = np.argsort(values, kind="stable")
    cumulative = np.cumsum(weights[order])
    targets = cumulative[-1] * np.arange(1, bins) / bins
    inner = values[order][np.searchsorted(cumulative, targets)]
    if not np.all(np.diff(inner) > 0):
        raise ValueError(f"{bins} quantiles of the bin column are not strictly increasing")
    return np.concatenate(([-np.inf], inner, [np.inf]))


def leg_rows(frame: pd.DataFrame, branch: str, leg: str) -> np.ndarray:
    """Which rows this branch scores in this leg."""
    if branch == "matched":
        return frame["matched"].to_numpy(bool)
    return np.isfinite(frame[f"e1_{leg}"].to_numpy(float))


def case_responses(frame: pd.DataFrame, branch: str, n_bins: int) -> dict[str, np.ndarray]:
    """Per (case, bin) measured and predicted response, plus its object weight.

    The two strata pool by their inverse-sampling weights inside a case, which
    is exactly what the production run's weighted sufficient statistics did.
    """
    cases = np.unique(frame["case"].to_numpy())
    shape = (len(cases), n_bins)
    sums = {leg: {name: np.zeros(shape) for name in ("e1", "e2", "response", "weight")}
            for leg in LEGS}
    index = pd.Series(np.arange(len(cases)), index=cases)
    case_of = index.reindex(frame["case"].to_numpy()).to_numpy()
    bin_of = frame["bin"].to_numpy()
    weight = frame["weight"].to_numpy(float)
    for leg in LEGS:
        active = leg_rows(frame, branch, leg)
        flat = case_of[active] * n_bins + bin_of[active]
        w = weight[active]
        for name, values in (
            ("e1", frame.loc[active, f"e1_{leg}"].to_numpy(float)),
            ("e2", frame.loc[active, f"e2_{leg}"].to_numpy(float)),
            ("response", frame.loc[active, "R_scene"].to_numpy(float)),
        ):
            sums[leg][name] += np.bincount(flat, w * values, shape[0] * shape[1]).reshape(shape)
        sums[leg]["weight"] += np.bincount(flat, w, shape[0] * shape[1]).reshape(shape)
    #  A primary scored in both legs is still one object, so the occupancy and
    #  the blending it carries are counted once over the union of the legs.
    scored = leg_rows(frame, branch, "plus") | leg_rows(frame, branch, "minus")
    flat = case_of[scored] * n_bins + bin_of[scored]
    objects = np.bincount(flat, None, shape[0] * shape[1]).reshape(shape).astype(float)
    blend_weight = np.bincount(
        flat, weight[scored] * np.abs(frame.loc[scored, "R_scene"].to_numpy(float)),
        shape[0] * shape[1]).reshape(shape)
    empty = min(sums[leg]["weight"].min() for leg in LEGS)
    if empty <= 0:
        raise RuntimeError("a (case, bin) cell is empty in one leg; use fewer bins")
    means = {leg: {name: sums[leg][name] / sums[leg]["weight"]
                   for name in ("e1", "e2", "response")} for leg in LEGS}
    return {
        "cases": cases,
        "R_sim": (means["plus"]["e1"] - means["minus"]["e1"]) / (2.0 * H),
        "R_sim_null": (means["plus"]["e2"] - means["minus"]["e2"]) / (2.0 * H),
        "R_emulator": 0.5 * (means["plus"]["response"] + means["minus"]["response"]),
        "objects": objects,
        "blend_weight": blend_weight,
    }


def summarize(per_case: dict, labels: list[str], replicates: int, seed: int) -> list[dict]:
    """Equal-weighted case mean per bin, with a case bootstrap on ``m``."""
    sim, model = per_case["R_sim"], per_case["R_emulator"]
    n_cases = len(per_case["cases"])
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n_cases, size=(replicates, n_cases))
    boot = sim[draws].mean(axis=1) / model[draws].mean(axis=1) - 1.0
    share = per_case["blend_weight"].sum(axis=0) / per_case["blend_weight"].sum()
    rows = []
    for index, label in enumerate(labels):
        measured = float(sim[:, index].mean())
        predicted = float(model[:, index].mean())
        rows.append({
            "bin": label,
            "objects": int(per_case["objects"][:, index].sum()),
            "R_sim": measured,
            "R_sim_case_sem": float(sim[:, index].std(ddof=1) / np.sqrt(n_cases)),
            "R_emulator": predicted,
            "R_emulator_case_sem": float(model[:, index].std(ddof=1) / np.sqrt(n_cases)),
            "R_sim_null_21": float(per_case["R_sim_null"][:, index].mean()),
            "m_percent": 100.0 * (measured / predicted - 1.0),
            "m_percent_standard_error": float(100.0 * boot[:, index].std(ddof=1)),
            "share_of_total_blending_percent": float(100.0 * share[index]),
        })
    return rows


def main(argv=None) -> None:
    args = parse_args(argv)
    frame = load(args.parts)
    keep = leg_rows(frame, args.branch, "plus") | leg_rows(frame, args.branch, "minus")
    #  The primary's own magnitude is a truth property, so restricting on it
    #  partitions the cohort without touching either leg's selection.
    magnitude = frame["r_input_p"].to_numpy(float)
    if args.primary_magnitude_min is not None:
        keep &= magnitude >= args.primary_magnitude_min
    if args.primary_magnitude_max is not None:
        keep &= magnitude < args.primary_magnitude_max
    if not keep.any():
        raise RuntimeError("the primary-magnitude restriction keeps nothing")
    scored = frame.loc[keep].copy()
    values = scored[args.bin_column].to_numpy(float)
    weights = scored["weight"].to_numpy(float)
    if args.edge:
        edges = np.concatenate(([-np.inf], np.sort(args.edge), [np.inf]))
    else:
        edges = bin_edges(values, weights, args.bins)
    scored["bin"] = np.searchsorted(edges[1:-1], values, side="right")
    labels = [f"bin{index:02d}" for index in range(len(edges) - 1)]
    per_case = case_responses(scored, args.branch, len(labels))
    rows = summarize(per_case, labels, args.replicates, args.seed)
    whole = scored.assign(bin=0)
    pooled = summarize(case_responses(whole, args.branch, 1), ["all"],
                       args.replicates, args.seed)
    quantiles = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0]
    result = {
        "format_version": 1,
        "branch": args.branch,
        "bin_column": args.bin_column,
        "bin_edges": [None if not np.isfinite(e) else float(e) for e in edges],
        "primary_magnitude_range": [args.primary_magnitude_min,
                                    args.primary_magnitude_max],
        "cases": [int(per_case["cases"].min()), int(per_case["cases"].max())],
        "n_cases": int(len(per_case["cases"])),
        "bootstrap": {"unit": "equal-weighted simulation case",
                      "replicates": int(args.replicates), "seed": int(args.seed)},
        "distribution": {
            name: {
                "weighted_mean": float(np.average(scored[name], weights=weights)),
                "quantiles": dict(zip(
                    [str(q) for q in quantiles],
                    [float(v) for v in np.quantile(scored[name], quantiles)])),
            }
            for name in ("R_scene", "R_scene_abs", "r_input_p", "pairs")
        },
        "bins": rows,
        "all": pooled[0],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(pd.DataFrame(rows).to_string(index=False), flush=True)
    print("\npooled:", json.dumps(pooled[0], indent=2), flush=True)


if __name__ == "__main__":
    main()
