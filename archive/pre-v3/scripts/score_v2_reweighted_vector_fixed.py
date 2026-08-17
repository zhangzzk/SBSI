#!/usr/bin/env python3
"""Score frozen V2 pair-response models by the half-shear vector beta metric.

This is evaluation only.  It streams half-shear cases 0--39, applies the exact
larger V2 regression support, and uses the same case-balanced vector-slope
definition that selected the source V2.2 recipe.  No fitting or selection is
performed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc
import xgboost as xgb


ROOT = Path(__file__).resolve().parents[1]
BLENDEMU = Path("/home/z/Zekang.Zhang/blendemu")
for path in (ROOT, BLENDEMU, BLENDEMU / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from blendemu import data_utils  # noqa: E402
from blendemu.config import (  # noqa: E402
    load_config,
    resolve_simulation_set,
    simulation_shear_scale,
)
from tune_v22_reweighted_vector_optuna import VectorEvaluation  # noqa: E402


FEATURES = [
    "Re_input_p_scaled",
    "Re_input_s_scaled",
    "r_input_p_scaled",
    "r_input_s_scaled",
    "sersic_n_input_p",
    "sersic_n_input_s",
    "distance_scaled",
]
RAW_FEATURES = [
    "Re_input_p",
    "Re_input_s",
    "r_input_p",
    "r_input_s",
    "sersic_n_input_p",
    "sersic_n_input_s",
    "distance",
]
NEEDED = [
    "case",
    "input_index",
    "shear_angle",
    "delta_et1",
    "delta_et2",
    *RAW_FEATURES,
]
V2_TAG = "lsst_r_extnbr_indom_tuned"
V2_CUTS = [[13.0, 29.0], [18.0, 26.0], [0.0, 10.0], [0.3, 1.5], [0.0, 10.0]]
CASE_WINDOW = (0, 39)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(clean(payload), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)


def load_booster(path: Path) -> xgb.Booster:
    booster = xgb.Booster({
        "device": os.environ.get("XGB_DEVICE", "cuda"),
        "n_jobs": int(os.environ.get("SLURM_CPUS_PER_TASK", "8")),
    })
    booster.load_model(path)
    booster.set_param({"device": os.environ.get("XGB_DEVICE", "cuda")})
    if booster.feature_names != FEATURES:
        raise RuntimeError(f"model feature order drifted for {path}")
    return booster


def selected_batch(
    reader: ipc.RecordBatchFileReader,
    batch_index: int,
    cuts: list[list[float]],
) -> pd.DataFrame:
    frame = (
        pa.Table.from_batches([reader.get_batch(batch_index)])
        .select(NEEDED)
        .to_pandas()
    )
    case = frame.case.to_numpy(np.int64, copy=False)
    use = (case >= CASE_WINDOW[0]) & (case <= CASE_WINDOW[1])
    frame = frame.loc[use]
    if frame.empty:
        return frame
    frame = data_utils.source_select_reg(frame, cuts=cuts)
    if frame.empty:
        return frame
    numeric = frame[NEEDED].to_numpy(np.float64, copy=False)
    if not np.isfinite(numeric).all():
        raise RuntimeError("selected V2 evaluation rows contain non-finite values")
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--training-summary", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    config_path = Path(args.config).resolve()
    summary_path = Path(args.training_summary).resolve()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    cfg = load_config(str(config_path))
    training = cfg["training"]
    if training["model_tag"] != V2_TAG:
        raise RuntimeError("configuration is not the frozen V2 domain")
    if list(training["features"]) != FEATURES:
        raise RuntimeError("configuration feature order drifted")
    cuts = [[float(value) for value in pair] for pair in training["regression_cuts"]]
    if cuts != V2_CUTS or summary["domain"]["regression_cuts"] != V2_CUTS:
        raise RuntimeError("V2 evaluation cuts differ from training provenance")
    if summary["features"] != FEATURES or summary["domain"]["fit_cases"] != [40, 199]:
        raise RuntimeError("training summary provenance drifted")

    model_paths = {
        "fresh_base": Path(summary["artifacts"]["base_model"]),
        "weighted": Path(summary["artifacts"]["weighted_model"]),
    }
    expected_hashes = {
        "fresh_base": summary["artifacts"]["base_model_sha256"],
        "weighted": summary["artifacts"]["weighted_model_sha256"],
    }
    metadata_path = BLENDEMU / "models" / f"emulator_metadata_{V2_TAG}.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    task = metadata["tasks"]["regression"]
    historical_path = BLENDEMU / "models" / task["model_file"]
    model_paths["historical_v2_indom_tuned"] = historical_path
    for name in ("fresh_base", "weighted"):
        if sha256(model_paths[name]) != expected_hashes[name]:
            raise RuntimeError(f"{name} model hash differs from training summary")
    if sha256(metadata_path) != summary["provenance"]["source_v2_metadata_sha256"]:
        raise RuntimeError("V2 metadata hash differs from training summary")
    if list(task["features"]) != FEATURES or task["cuts"] != V2_CUTS:
        raise RuntimeError("historical V2 metadata support drifted")

    standardization = task["standardization"]
    mean = float(standardization["mean"])
    scale = float(standardization["std"])
    if not np.isclose(mean, summary["training"]["target_mean"], rtol=0.0, atol=1e-12):
        raise RuntimeError("target mean differs between model and V2 metadata")
    if not np.isclose(scale, summary["training"]["target_scale"], rtol=0.0, atol=1e-12):
        raise RuntimeError("target scale differs between model and V2 metadata")

    boosters = {name: load_booster(path) for name, path in model_paths.items()}
    catalogue = Path(cfg["simulation"]["output_path"]) / "response_catalogue_train.feather"
    simulation = resolve_simulation_set(
        cfg["simulation"],
        "response",
        overrides=cfg.get("catalogues", {}).get("response", {}),
    )
    shear = float(simulation_shear_scale(simulation, label="simulation.response"))
    if not np.isclose(shear, 0.2, rtol=0.0, atol=1e-12):
        raise RuntimeError(f"unexpected half-shear amplitude {shear}")

    chunks: dict[str, list[np.ndarray]] = {
        "case": [],
        "input_index": [],
        "shear_angle": [],
        "label": [],
        "null": [],
        **{name: [] for name in boosters},
    }
    counts = np.zeros(CASE_WINDOW[1] + 1, dtype=np.int64)
    with ipc.open_file(catalogue) as reader:
        missing = set(NEEDED) - set(reader.schema.names)
        if missing:
            raise KeyError(f"response catalogue lacks {sorted(missing)}")
        for batch_index in range(reader.num_record_batches):
            frame = selected_batch(reader, batch_index, cuts)
            if frame.empty:
                continue
            scaled = data_utils.rescale(
                frame.copy(),
                pixel_rms=training["rescale"]["pixel_rms"],
                pixel_size=training["rescale"]["pixel_size"],
                zero_mag=training["rescale"]["zero_mag"],
                psf_fwhm=training["rescale"]["psf_fwhm"],
                moffat_beta=training["rescale"]["moffat_beta"],
            )
            x_scaled = scaled[FEATURES].to_numpy(np.float32)
            matrix = xgb.DMatrix(x_scaled, feature_names=FEATURES)
            case = frame.case.to_numpy(np.int16)
            chunks["case"].append(case)
            chunks["input_index"].append(frame.input_index.to_numpy(np.int64))
            chunks["shear_angle"].append(frame.shear_angle.to_numpy(np.float32))
            chunks["label"].append(
                (frame.delta_et1.to_numpy(np.float64) / shear).astype(np.float32)
            )
            chunks["null"].append(
                (frame.delta_et2.to_numpy(np.float64) / shear).astype(np.float32)
            )
            for name, booster in boosters.items():
                prediction = booster.predict(matrix).astype(np.float64) * scale + mean
                chunks[name].append(prediction.astype(np.float32))
            counts += np.bincount(case.astype(np.int64), minlength=len(counts))
            if (batch_index + 1) % 100 == 0:
                print(
                    f"batch {batch_index + 1}/{reader.num_record_batches}: "
                    f"selected={counts.sum():,}",
                    flush=True,
                )

    arrays = {name: np.concatenate(parts) for name, parts in chunks.items()}
    if np.any(counts <= 0) or len(arrays["case"]) != int(counts.sum()):
        raise RuntimeError("V2 evaluation case coverage is incomplete")
    vector = VectorEvaluation(
        case=arrays["case"],
        primary=arrays["input_index"],
        shear_angle_degree=arrays["shear_angle"],
        label=arrays["label"],
        null=arrays["null"],
    )
    results: dict[str, Any] = {}
    for name in boosters:
        metrics = vector.score(arrays[name], CASE_WINDOW)
        beta = metrics["slope_measured_on_predicted"]
        metrics["score_abs_beta_minus_one"] = abs(float(beta["mean"]) - 1.0)
        results[name] = metrics

    payload = {
        "schema_version": 1,
        "kind": "frozen V2 half-shear vector-score evaluation",
        "metric": {
            "beta": "case-balanced projection slope of measured scene-response vector on summed predicted pair-response vector",
            "score": "abs(beta - 1)",
        },
        "evaluation": {
            "case_window": list(CASE_WINDOW),
            "domain": "V2 rectangular primary domain",
            "regression_cuts": V2_CUTS,
            "n_pairs": int(len(arrays["case"])),
            "case_counts": counts.tolist(),
            "shear": shear,
            "role": "retrospective transfer diagnostic; no fitting or selection",
            "independence_note": "these case IDs selected the source V2.2 recipe, so this is not an untouched confirmation block",
        },
        "models": {
            name: {"path": str(path), "sha256": sha256(path)}
            for name, path in model_paths.items()
        },
        "results": results,
        "provenance": {
            "config": str(config_path),
            "training_summary": str(summary_path),
            "training_summary_sha256": sha256(summary_path),
            "source_catalogue": str(catalogue),
            "source_catalogue_size_bytes": int(catalogue.stat().st_size),
            "source_v2_metadata": str(metadata_path),
            "source_v2_metadata_sha256": sha256(metadata_path),
            "constgold_opened_by_this_evaluator": False,
            "coherent_anchor_truth_opened": False,
        },
    }
    write_json(output, payload)
    print(json.dumps(clean(payload), indent=2, sort_keys=True), flush=True)
    print("V2_REWEIGHTED_VECTOR_FIXED_SCORE_DONE", flush=True)


if __name__ == "__main__":
    main()
