"""Measure label and emulator-prediction means on the exact retraining split."""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import xgboost as xgb


BE = "/home/z/Zekang.Zhang/blendemu"
for path in (BE, os.path.join(BE, "scripts")):
    if path not in sys.path:
        sys.path.insert(0, path)

from blendemu.config import load_config  # noqa: E402
import retrain_extnbr as RE  # noqa: E402


def load_model(tag: str) -> tuple[xgb.Booster, float, float]:
    meta_path = os.path.join(BE, "models", f"emulator_metadata_{tag}.json")
    with open(meta_path, encoding="utf-8") as handle:
        task = json.load(handle)["tasks"]["regression"]
    model = xgb.Booster({"device": "cpu", "n_jobs": -1})
    model.load_model(os.path.join(BE, "models", task["model_file"]))
    return model, float(task["standardization"]["mean"]), float(task["standardization"]["std"])


def physical_prediction(model: xgb.Booster, dmatrix: xgb.DMatrix, mean: float, std: float) -> np.ndarray:
    return model.predict(dmatrix).astype(np.float64) * std + mean


def summarize(label: np.ndarray, source: np.ndarray, candidate: np.ndarray) -> dict[str, float | int]:
    return {
        "rows": int(len(label)),
        "label_mean": float(np.mean(label, dtype=np.float64)),
        "source_prediction_mean": float(np.mean(source, dtype=np.float64)),
        "candidate_prediction_mean": float(np.mean(candidate, dtype=np.float64)),
        "source_minus_label": float(np.mean(source - label, dtype=np.float64)),
        "candidate_minus_label": float(np.mean(candidate - label, dtype=np.float64)),
        "candidate_minus_source": float(np.mean(candidate - source, dtype=np.float64)),
    }


def combine(parts: list[dict[str, float | int]]) -> dict[str, float | int]:
    n = sum(int(part["rows"]) for part in parts)
    out: dict[str, float | int] = {"rows": n}
    for key in parts[0]:
        if key != "rows":
            out[key] = sum(float(part[key]) * int(part["rows"]) for part in parts) / n
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--source-tag", required=True)
    ap.add_argument("--candidate-tag", required=True)
    ap.add_argument("--minimum-case", type=int, default=0)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    if args.minimum_case < 0:
        raise SystemExit("--minimum-case must be non-negative")
    os.environ["HELDOUT_MIN_CASE"] = str(args.minimum_case)
    cfg = load_config(args.config)
    dm_train, dm_validation, x_train, x_validation, y_train, y_validation, y_mean, y_std = (
        RE.load_regression_data_lowmem(cfg)
    )
    del x_train, x_validation
    label_train = np.asarray(y_train, dtype=np.float64) * y_std + y_mean
    label_validation = np.asarray(y_validation, dtype=np.float64) * y_std + y_mean
    del y_train, y_validation

    source, source_mean, source_std = load_model(args.source_tag)
    candidate, candidate_mean, candidate_std = load_model(args.candidate_tag)
    candidate_meta_delta = max(abs(candidate_mean - y_mean), abs(candidate_std - y_std))
    if candidate_meta_delta > 1e-10:
        raise SystemExit(
            "candidate standardization disagrees with reconstructed training data: "
            f"metadata=({candidate_mean}, {candidate_std}) data=({y_mean}, {y_std})"
        )

    source_train = physical_prediction(source, dm_train, source_mean, source_std)
    candidate_train = physical_prediction(candidate, dm_train, candidate_mean, candidate_std)
    train = summarize(label_train, source_train, candidate_train)
    del label_train, source_train, candidate_train

    source_validation = physical_prediction(source, dm_validation, source_mean, source_std)
    candidate_validation = physical_prediction(candidate, dm_validation, candidate_mean, candidate_std)
    validation = summarize(label_validation, source_validation, candidate_validation)
    result = {
        "config": os.path.abspath(args.config),
        "source_tag": args.source_tag,
        "candidate_tag": args.candidate_tag,
        "minimum_training_case": args.minimum_case,
        "train": train,
        "validation": validation,
        "global": combine([train, validation]),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps(result, indent=2, allow_nan=False))
    print("EMU_GLOBAL_MEAN_DONE")


if __name__ == "__main__":
    main()
