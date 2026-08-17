"""Whole-case bootstrap retrains of the frozen V2.2 R_blend regression.

Replicate zero reproduces the nominal cases-40--199 recipe.  Positive
replicates resample those 160 rendered cases with replacement while preserving
the original row split, feature transform, XGBoost parameters, and early
stopping rule.  Case multiplicities are represented as row weights, which is
equivalent to duplicating each case for the squared-error objective without
materializing a much larger dataframe.

Only half-shear response labels are read.  Constgold and anchor truth are not
opened by this script.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc
import xgboost as xgb
from sklearn.model_selection import train_test_split


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BE = "/home/z/Zekang.Zhang/blendemu"
for path in (ROOT, BE, os.path.join(BE, "scripts")):
    if path not in sys.path:
        sys.path.insert(0, path)

from blendemu import data_utils  # noqa: E402
from blendemu.config import load_config  # noqa: E402
import train_emulator as TE  # noqa: E402


BASE_TAG = "lsst_r_extnbr_v22"
CASE_MIN = 40
CASE_MAX = 199
N_CASES = CASE_MAX - CASE_MIN + 1
NEED = [
    "case", "r_input_p", "r_input_s", "Re_input_p", "Re_input_s",
    "sersic_n_input_p", "sersic_n_input_s", "distance", "delta_et1",
]


def weighted_mean_std(values: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
    """Mean and duplicated-row sample standard deviation."""
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    total = float(weights.sum())
    if total <= 1.0:
        raise ValueError("bootstrap has fewer than two weighted rows")
    mean = float(np.dot(weights, values) / total)
    variance = float(np.dot(weights, np.square(values - mean)) / (total - 1.0))
    return mean, float(np.sqrt(variance))


def weighted_r2(prediction: np.ndarray, data: xgb.DMatrix) -> tuple[str, float]:
    label = data.get_label().astype(np.float64, copy=False)
    weight = data.get_weight().astype(np.float64, copy=False)
    if not len(weight):
        weight = np.ones(len(label), dtype=np.float64)
    total = float(weight.sum())
    mean = float(np.dot(weight, label) / total)
    ss_res = float(np.dot(weight, np.square(label - prediction)))
    ss_tot = float(np.dot(weight, np.square(label - mean)))
    return "r2", 1.0 - ss_res / ss_tot


def load_selected(cfg: dict) -> pd.DataFrame:
    catalogue = os.path.join(
        cfg["simulation"]["output_path"], "response_catalogue_train.feather",
    )
    parts: list[pd.DataFrame] = []
    seen: set[int] = set()
    raw_rows = 0
    with ipc.open_file(catalogue) as reader:
        missing = set(NEED) - set(reader.schema.names)
        if missing:
            raise KeyError(f"response catalogue lacks {sorted(missing)}")
        for batch_index in range(reader.num_record_batches):
            frame = pa.Table.from_batches([reader.get_batch(batch_index)]).select(NEED).to_pandas()
            frame = frame[frame.case.between(CASE_MIN, CASE_MAX)]
            if not len(frame):
                continue
            raw_rows += len(frame)
            seen.update(frame.case.astype(int).unique().tolist())
            selected = data_utils.source_select_reg(
                frame, cuts=cfg["training"]["regression_cuts"],
            )
            selected = selected[np.isfinite(selected.delta_et1.to_numpy(float))]
            if len(selected):
                parts.append(selected)
    expected = set(range(CASE_MIN, CASE_MAX + 1))
    if seen != expected:
        raise RuntimeError(
            f"training cases drifted: missing={sorted(expected-seen)} extra={sorted(seen-expected)}"
        )
    dataset = pd.concat(parts, ignore_index=True)
    print(
        f"catalogue={catalogue} cases={CASE_MIN}--{CASE_MAX} "
        f"raw={raw_rows:,} selected={len(dataset):,}", flush=True,
    )
    return dataset


def case_multiplicities(replicate: int, seed: int) -> np.ndarray:
    if replicate == 0:
        return np.ones(N_CASES, dtype=np.int16)
    rng = np.random.default_rng(seed + replicate)
    sampled = rng.integers(0, N_CASES, size=N_CASES)
    return np.bincount(sampled, minlength=N_CASES).astype(np.int16)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--replicate", type=int, required=True)
    parser.add_argument("--bootstrap-seed", type=int, default=202608120)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    if args.replicate < 0:
        raise ValueError("replicate must be non-negative; zero is the nominal reproduction")

    os.makedirs(args.output_dir, exist_ok=True)
    model_path = os.path.join(args.output_dir, f"regression_rep{args.replicate:02d}.json")
    summary_path = os.path.join(args.output_dir, f"summary_rep{args.replicate:02d}.json")
    for path in (model_path, summary_path):
        if os.path.exists(path):
            raise FileExistsError(f"refusing to overwrite {path}")

    cfg = load_config(args.config)
    training = cfg["training"]
    if training["model_tag"] != BASE_TAG:
        raise RuntimeError(f"expected {BASE_TAG}, got {training['model_tag']}")
    base_metadata_path = os.path.join(BE, "models", f"emulator_metadata_{BASE_TAG}.json")
    with open(base_metadata_path, encoding="utf-8") as handle:
        base_task = json.load(handle)["tasks"]["regression"]
    if list(training["features"]) != list(base_task["features"]):
        raise RuntimeError("feature list differs from frozen V2.2")

    dataset = load_selected(cfg)
    rescale = training["rescale"]
    dataset = data_utils.rescale(
        dataset, pixel_rms=rescale["pixel_rms"], pixel_size=rescale["pixel_size"],
        zero_mag=rescale["zero_mag"], psf_fwhm=rescale["psf_fwhm"],
        moffat_beta=rescale["moffat_beta"],
    )
    shear = TE._catalogue_shear_scale(cfg, "response")
    if not np.isclose(shear, 0.2, rtol=0.0, atol=1e-12):
        raise RuntimeError(f"unexpected response shear {shear}")
    response = dataset.delta_et1.to_numpy(np.float64) / shear
    case = dataset.case.to_numpy(np.int16)
    multiplicity = case_multiplicities(args.replicate, args.bootstrap_seed)
    row_weight = multiplicity[case - CASE_MIN].astype(np.float64)
    response_mean, response_std = weighted_mean_std(response, row_weight)
    target = (response - response_mean) / response_std

    indices = np.arange(len(dataset), dtype=np.int64)
    train_index, validation_index = train_test_split(
        indices, test_size=training["test_size"], random_state=training["random_state"],
    )
    features = list(training["features"])
    x_train = dataset.iloc[train_index][features]
    x_validation = dataset.iloc[validation_index][features]
    train_weight = row_weight[train_index]
    validation_weight = row_weight[validation_index]
    if args.replicate == 0:
        dm_train = xgb.DMatrix(x_train, target[train_index])
        dm_validation = xgb.DMatrix(x_validation, target[validation_index])
    else:
        dm_train = xgb.DMatrix(x_train, target[train_index], weight=train_weight)
        dm_validation = xgb.DMatrix(
            x_validation, target[validation_index], weight=validation_weight,
        )

    if args.replicate == 0:
        expected = base_task["standardization"]
        if not np.isclose(response_mean, expected["mean"], rtol=0.0, atol=2e-12):
            raise RuntimeError(f"nominal mean mismatch {response_mean} != {expected['mean']}")
        if not np.isclose(response_std, expected["std"], rtol=0.0, atol=2e-12):
            raise RuntimeError(f"nominal std mismatch {response_std} != {expected['std']}")

    params = dict(base_task["params"])
    params.update(
        objective="reg:squarederror", n_jobs=-1,
        device=os.environ.get("XGB_DEVICE", "cpu"), tree_method="hist",
        booster="gbtree", disable_default_eval_metric=1,
    )
    print(
        f"replicate={args.replicate} unique_cases={(multiplicity>0).sum()} "
        f"max_case_multiplicity={multiplicity.max()} label={response_mean:+.8f}+-{response_std:.8f}",
        flush=True,
    )
    history: dict = {}
    start = time.time()
    booster = xgb.train(
        params, dm_train, evals=[(dm_train, "train"), (dm_validation, "eval")],
        evals_result=history, num_boost_round=2000, verbose_eval=100,
        custom_metric=weighted_r2, maximize=True,
        callbacks=[TE._early_stopping_callback(cfg, "regression", "r2", maximize=True)],
    )
    trees = data_utils.get_xgb_iteration_range(booster)[1]
    booster.save_model(model_path)
    payload = {
        "replicate": args.replicate,
        "kind": "nominal_reproduction" if args.replicate == 0 else "whole_case_bootstrap",
        "bootstrap_seed": args.bootstrap_seed,
        "case_window": [CASE_MIN, CASE_MAX],
        "n_source_cases": N_CASES,
        "n_unique_cases": int((multiplicity > 0).sum()),
        "case_multiplicities": {
            str(CASE_MIN + index): int(value)
            for index, value in enumerate(multiplicity)
        },
        "selected_rows": int(len(dataset)),
        "weighted_rows": int(row_weight.sum()),
        "train_rows_physical": int(len(train_index)),
        "validation_rows_physical": int(len(validation_index)),
        "weighted_train_rows": float(train_weight.sum()),
        "weighted_validation_rows": float(validation_weight.sum()),
        "response_mean": response_mean,
        "response_std": response_std,
        "features": features,
        "best_iteration": int(booster.best_iteration),
        "best_trees": int(trees),
        "best_score": float(booster.best_score),
        "fit_seconds": float(time.time() - start),
        "model_path": model_path,
        "constgold_opened": False,
    }
    with open(summary_path, "x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    print("V22_CASE_BOOTSTRAP_TRAIN_DONE", flush=True)


if __name__ == "__main__":
    main()
