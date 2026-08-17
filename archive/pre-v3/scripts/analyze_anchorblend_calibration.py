"""Fit and validate a global R_blend scale without reading constgold.

Cases below ``--split-case`` are the development set.  They determine the
single multiplicative scale.  Cases at or above the split are untouched until
the validation report is formed.  Case bootstrap errors preserve the sky-scene
as the independent sampling unit.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from scripts.fit_anchorblend_isotonic import read_disjoint_inputs


def case_table(frame: pd.DataFrame, truth: str, prediction: str) -> pd.DataFrame:
    finite = np.isfinite(frame[[truth, prediction]].to_numpy(float)).all(axis=1)
    clean = frame.loc[finite, ["case", truth, prediction]]
    return clean.groupby("case", sort=True)[[truth, prediction]].mean()


def scale_and_residual(dev: np.ndarray, test: np.ndarray) -> tuple[float, float, float]:
    """Return dev scale, held-out residual, and independently implied test scale."""
    scale = float(dev[:, 0].mean() / dev[:, 1].mean())
    residual = float(test[:, 0].mean() - scale * test[:, 1].mean())
    test_scale = float(test[:, 0].mean() / test[:, 1].mean())
    return scale, residual, test_scale


def bootstrap(dev: np.ndarray, test: np.ndarray, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    result = np.empty((n, 3), dtype=float)
    for i in range(n):
        d = dev[rng.integers(0, len(dev), len(dev))]
        t = test[rng.integers(0, len(test), len(test))]
        result[i] = scale_and_residual(d, t)
    return result


def bootstrap_scale(data: np.ndarray, n: int, seed: int) -> np.ndarray:
    """Case-bootstrap the all-case deployment scale."""
    rng = np.random.default_rng(seed)
    result = np.empty(n, dtype=float)
    for i in range(n):
        sample = data[rng.integers(0, len(data), len(data))]
        result[i] = sample[:, 0].mean() / sample[:, 1].mean()
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "input", nargs="+",
        help="one or more disjoint anchor-response feather files",
    )
    ap.add_argument("--tag", default="lsst_r_extnbr_v21")
    ap.add_argument("--split-case", type=int, default=50)
    ap.add_argument("--bootstrap", type=int, default=20_000)
    ap.add_argument("--seed", type=int, default=25876)
    ap.add_argument("--output-json")
    args = ap.parse_args()

    truth = "R_blend_truth"
    prediction = f"R_blend_{args.tag}"
    table = case_table(read_disjoint_inputs(args.input), truth, prediction)
    dev = table.loc[table.index < args.split_case].to_numpy(float)
    test = table.loc[table.index >= args.split_case].to_numpy(float)
    if len(dev) < 10 or len(test) < 10:
        raise RuntimeError(f"need >=10 independent cases per split; got {len(dev)} and {len(test)}")

    scale, residual, test_scale = scale_and_residual(dev, test)
    draws = bootstrap(dev, test, args.bootstrap, args.seed)
    errors = draws.std(axis=0, ddof=1)
    test_truth, test_pred = test.mean(axis=0)
    deployment_scale = float(table[truth].mean() / table[prediction].mean())
    deployment_scale_se = float(
        bootstrap_scale(table.to_numpy(float), args.bootstrap, args.seed + 1).std(ddof=1)
    )
    result = {
        "tag": args.tag,
        "split_case": args.split_case,
        "n_dev_cases": len(dev),
        "n_test_cases": len(test),
        "dev_truth_mean": float(dev[:, 0].mean()),
        "dev_prediction_mean": float(dev[:, 1].mean()),
        "dev_scale": scale,
        "dev_scale_se": float(errors[0]),
        "test_truth_mean": float(test_truth),
        "test_prediction_mean": float(test_pred),
        "test_corrected_mean": float(scale * test_pred),
        "test_residual": residual,
        "test_residual_se": float(errors[1]),
        "test_implied_scale": test_scale,
        "test_implied_scale_se": float(errors[2]),
        "deployment_scale": deployment_scale,
        "deployment_scale_se": deployment_scale_se,
        "deployment_fit_cases": len(table),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.output_json:
        with open(args.output_json, "x", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, sort_keys=True)
            handle.write("\n")


if __name__ == "__main__":
    main()
