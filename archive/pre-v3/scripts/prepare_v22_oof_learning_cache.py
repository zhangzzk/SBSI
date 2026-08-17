#!/usr/bin/env python3
"""Build a compact, immutable cache for V2.2 cross-fitted learning tests.

The source is the existing half-shear response catalogue.  Rows are restricted
to the exact frozen V2.2 regression support.  Cases 40--199 are the model-
development population; cases 0--39 remain external half-shear evaluation.

Large arrays are written under the project filesystem.  The script makes two
streaming passes over the 26-GB Feather catalogue so memory use stays modest.
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
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]
BLENDEMU = Path("/home/z/Zekang.Zhang/blendemu")
for path in (ROOT, BLENDEMU, BLENDEMU / "scripts"):
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)

from blendemu import data_utils  # noqa: E402
from blendemu.config import (  # noqa: E402
    load_config,
    resolve_simulation_set,
    simulation_shear_scale,
)


SOURCE_TAG = "lsst_r_extnbr_v22"
CASE_MIN = 0
CASE_MAX = 199
FIT_CASE_MIN = 40
N_FOLDS = 4
FOLD_SEED = 20260814
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
EXPECTED = {
    "all_rows": 47_310_214,
    "fit_rows": 37_852_393,
    "official_train_rows": 30_281_914,
    "official_validation_rows": 7_570_479,
}


def strict_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def file_sha256(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


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
    frame = frame.loc[(case >= CASE_MIN) & (case <= CASE_MAX)]
    if frame.empty:
        return frame
    frame = data_utils.source_select_reg(frame, cuts=cuts)
    if frame.empty:
        return frame
    numeric = frame[NEEDED].to_numpy(np.float64, copy=False)
    if not np.isfinite(numeric).all():
        bad = int((~np.isfinite(numeric)).any(axis=1).sum())
        raise RuntimeError(f"selected V2.2 batch contains {bad} non-finite rows")
    return frame


def count_rows(catalogue: Path, cuts: list[list[float]]) -> np.ndarray:
    counts = np.zeros(CASE_MAX + 1, dtype=np.int64)
    with ipc.open_file(catalogue) as reader:
        missing = set(NEEDED) - set(reader.schema.names)
        if missing:
            raise KeyError(f"response catalogue lacks {sorted(missing)}")
        for batch_index in range(reader.num_record_batches):
            frame = selected_batch(reader, batch_index, cuts)
            if frame.empty:
                continue
            counts += np.bincount(
                frame.case.to_numpy(np.int64), minlength=CASE_MAX + 1
            )
            if (batch_index + 1) % 100 == 0:
                print(
                    f"count pass batch {batch_index + 1}/{reader.num_record_batches}: "
                    f"selected={counts.sum():,}",
                    flush=True,
                )
    return counts


def fold_assignment() -> np.ndarray:
    output = np.full(CASE_MAX + 1, -1, dtype=np.int8)
    cases = np.arange(FIT_CASE_MIN, CASE_MAX + 1, dtype=np.int16)
    rng = np.random.default_rng(FOLD_SEED)
    cases = cases[rng.permutation(len(cases))]
    for index, case in enumerate(cases):
        output[int(case)] = np.int8(index % N_FOLDS)
    fold_counts = np.bincount(output[FIT_CASE_MIN:].astype(int), minlength=N_FOLDS)
    if not np.array_equal(fold_counts, np.full(N_FOLDS, 40)):
        raise RuntimeError(f"unbalanced case folds: {fold_counts.tolist()}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite cache directory {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.parent / f".{output.name}.{os.getpid()}.tmp"
    if temporary.exists():
        raise FileExistsError(f"temporary path already exists: {temporary}")
    temporary.mkdir()

    try:
        cfg = load_config(args.config)
        training = cfg["training"]
        if training["model_tag"] != SOURCE_TAG:
            raise RuntimeError(
                f"expected config tag {SOURCE_TAG}, got {training['model_tag']}"
            )
        metadata_path = BLENDEMU / "models" / f"emulator_metadata_{SOURCE_TAG}.json"
        model_path = BLENDEMU / "models" / f"regression_model_{SOURCE_TAG}.json"
        with metadata_path.open(encoding="utf-8") as handle:
            source_metadata = json.load(handle)
        task = source_metadata["tasks"]["regression"]
        model_features = list(task["features"])
        if list(training["features"]) != model_features:
            raise RuntimeError("configuration features differ from frozen V2.2 metadata")
        if training["regression_cuts"] != task["cuts"]:
            raise RuntimeError("configuration cuts differ from frozen V2.2 metadata")

        catalogue = Path(cfg["simulation"]["output_path"]) / "response_catalogue_train.feather"
        counts = count_rows(catalogue, training["regression_cuts"])
        n_rows = int(counts.sum())
        fit_rows = int(counts[FIT_CASE_MIN:].sum())
        if n_rows != EXPECTED["all_rows"] or fit_rows != EXPECTED["fit_rows"]:
            raise RuntimeError(
                "V2.2 selected-row count drifted: "
                f"all={n_rows:,} fit={fit_rows:,} expected={EXPECTED}"
            )
        if np.any(counts <= 0):
            raise RuntimeError("at least one expected case has no supported pairs")

        arrays = {
            "case": np.lib.format.open_memmap(
                temporary / "case.npy", mode="w+", dtype=np.int16, shape=(n_rows,)
            ),
            "input_index": np.lib.format.open_memmap(
                temporary / "input_index.npy", mode="w+", dtype=np.int64, shape=(n_rows,)
            ),
            "shear_angle": np.lib.format.open_memmap(
                temporary / "shear_angle.npy", mode="w+", dtype=np.float32, shape=(n_rows,)
            ),
            "label": np.lib.format.open_memmap(
                temporary / "label.npy", mode="w+", dtype=np.float32, shape=(n_rows,)
            ),
            "null": np.lib.format.open_memmap(
                temporary / "null.npy", mode="w+", dtype=np.float32, shape=(n_rows,)
            ),
            "x_raw": np.lib.format.open_memmap(
                temporary / "x_raw.npy",
                mode="w+",
                dtype=np.float32,
                shape=(n_rows, len(RAW_FEATURES)),
            ),
            "x_scaled": np.lib.format.open_memmap(
                temporary / "x_scaled.npy",
                mode="w+",
                dtype=np.float32,
                shape=(n_rows, len(model_features)),
            ),
            "v22_prediction": np.lib.format.open_memmap(
                temporary / "v22_prediction.npy",
                mode="w+",
                dtype=np.float32,
                shape=(n_rows,),
            ),
        }

        booster = xgb.Booster({"device": "cpu", "n_jobs": -1})
        booster.load_model(model_path)
        standardization = task["standardization"]
        y_mean = float(standardization["mean"])
        y_std = float(standardization["std"])
        simulation = resolve_simulation_set(
            cfg["simulation"],
            "response",
            overrides=cfg.get("catalogues", {}).get("response", {}),
        )
        shear = float(
            simulation_shear_scale(simulation, label="simulation.response")
        )
        if not np.isclose(shear, 0.2, rtol=0.0, atol=1.0e-12):
            raise RuntimeError(f"unexpected response shear {shear}")

        cursor = 0
        filled_counts = np.zeros_like(counts)
        with ipc.open_file(catalogue) as reader:
            for batch_index in range(reader.num_record_batches):
                frame = selected_batch(
                    reader, batch_index, training["regression_cuts"]
                )
                if frame.empty:
                    continue
                n_batch = len(frame)
                stop = cursor + n_batch
                scaled = data_utils.rescale(
                    frame.copy(),
                    pixel_rms=training["rescale"]["pixel_rms"],
                    pixel_size=training["rescale"]["pixel_size"],
                    zero_mag=training["rescale"]["zero_mag"],
                    psf_fwhm=training["rescale"]["psf_fwhm"],
                    moffat_beta=training["rescale"]["moffat_beta"],
                )
                x_scaled = scaled[model_features].to_numpy(np.float32)
                prediction = booster.inplace_predict(x_scaled).astype(np.float64)
                prediction = prediction * y_std + y_mean
                arrays["case"][cursor:stop] = frame.case.to_numpy(np.int16)
                arrays["input_index"][cursor:stop] = frame.input_index.to_numpy(np.int64)
                arrays["shear_angle"][cursor:stop] = frame.shear_angle.to_numpy(np.float32)
                arrays["label"][cursor:stop] = (
                    frame.delta_et1.to_numpy(np.float64) / shear
                ).astype(np.float32)
                arrays["null"][cursor:stop] = (
                    frame.delta_et2.to_numpy(np.float64) / shear
                ).astype(np.float32)
                arrays["x_raw"][cursor:stop] = frame[RAW_FEATURES].to_numpy(np.float32)
                arrays["x_scaled"][cursor:stop] = x_scaled
                arrays["v22_prediction"][cursor:stop] = prediction.astype(np.float32)
                filled_counts += np.bincount(
                    frame.case.to_numpy(np.int64), minlength=CASE_MAX + 1
                )
                cursor = stop
                if (batch_index + 1) % 100 == 0:
                    print(
                        f"write pass batch {batch_index + 1}/{reader.num_record_batches}: "
                        f"filled={cursor:,}/{n_rows:,}",
                        flush=True,
                    )
        if cursor != n_rows or not np.array_equal(filled_counts, counts):
            raise RuntimeError("streamed fill does not reproduce count pass")
        for array in arrays.values():
            array.flush()
        del arrays

        case = np.load(temporary / "case.npy", mmap_mode="r")
        fit_index = np.flatnonzero(case >= FIT_CASE_MIN)
        local = np.arange(len(fit_index), dtype=np.int32)
        official_train, official_validation = train_test_split(
            local,
            test_size=float(training["test_size"]),
            random_state=int(training["random_state"]),
        )
        official_mask = np.lib.format.open_memmap(
            temporary / "official_train.npy",
            mode="w+",
            dtype=np.bool_,
            shape=(n_rows,),
        )
        official_mask[:] = False
        official_mask[fit_index[official_train]] = True
        official_mask.flush()
        train_rows = int(official_mask.sum())
        validation_rows = fit_rows - train_rows
        if (
            train_rows != EXPECTED["official_train_rows"]
            or validation_rows != EXPECTED["official_validation_rows"]
        ):
            raise RuntimeError(
                f"official split drifted: train={train_rows:,} validation={validation_rows:,}"
            )
        del local, official_train, official_validation, official_mask

        folds = fold_assignment()
        np.save(temporary / "fold_by_case.npy", folds)
        fold_rows = np.asarray([
            counts[np.flatnonzero(folds == fold)].sum() for fold in range(N_FOLDS)
        ], dtype=np.int64)

        label = np.load(temporary / "label.npy", mmap_mode="r")
        prediction = np.load(temporary / "v22_prediction.npy", mmap_mode="r")
        fit = np.asarray(case) >= FIT_CASE_MIN
        physical = {
            "label_mean_fit": float(np.asarray(label[fit], dtype=np.float64).mean()),
            "label_std_fit_ddof1": float(np.asarray(label[fit], dtype=np.float64).std(ddof=1)),
            "prediction_mean_fit": float(
                np.asarray(prediction[fit], dtype=np.float64).mean()
            ),
            "residual_mean_fit": float(
                np.asarray(label[fit] - prediction[fit], dtype=np.float64).mean()
            ),
        }
        if not np.isclose(
            physical["label_mean_fit"], y_mean, rtol=0.0, atol=2.0e-9
        ):
            raise RuntimeError("cached fit label mean does not reproduce V2.2 metadata")

        payload = {
            "schema_version": 1,
            "source_tag": SOURCE_TAG,
            "source_catalogue": str(catalogue),
            "source_catalogue_size_bytes": int(catalogue.stat().st_size),
            "source_metadata": str(metadata_path),
            "source_metadata_sha256": file_sha256(metadata_path),
            "source_model": str(model_path),
            "source_model_sha256": file_sha256(model_path),
            "config": str(Path(args.config).resolve()),
            "case_window": [CASE_MIN, CASE_MAX],
            "fit_case_window": [FIT_CASE_MIN, CASE_MAX],
            "evaluation_case_window": [CASE_MIN, FIT_CASE_MIN - 1],
            "n_rows": n_rows,
            "n_fit_rows": fit_rows,
            "n_evaluation_rows": n_rows - fit_rows,
            "case_counts": {str(case_id): int(value) for case_id, value in enumerate(counts)},
            "official_random_row_split": {
                "random_state": int(training["random_state"]),
                "test_size": float(training["test_size"]),
                "train_rows": train_rows,
                "validation_rows": validation_rows,
            },
            "case_crossfit": {
                "n_folds": N_FOLDS,
                "seed": FOLD_SEED,
                "fold_rows": fold_rows.tolist(),
                "fold_cases": {
                    str(fold): np.flatnonzero(folds == fold).astype(int).tolist()
                    for fold in range(N_FOLDS)
                },
            },
            "raw_features": RAW_FEATURES,
            "model_features": model_features,
            "source_standardization": {"mean": y_mean, "std": y_std},
            "shear": shear,
            "physical_reproduction": physical,
            "arrays": {
                "case": "case.npy",
                "input_index": "input_index.npy",
                "shear_angle": "shear_angle.npy",
                "label": "label.npy",
                "null": "null.npy",
                "x_raw": "x_raw.npy",
                "x_scaled": "x_scaled.npy",
                "v22_prediction": "v22_prediction.npy",
                "official_train": "official_train.npy",
                "fold_by_case": "fold_by_case.npy",
            },
            "constgold_opened": False,
            "anchor_truth_opened": False,
        }
        strict_json(temporary / "metadata.json", payload)
        os.replace(temporary, output)
        print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
        print("V22_OOF_CACHE_DONE", flush=True)
    except Exception:
        print(f"failed cache remains for inspection at {temporary}", flush=True)
        raise


if __name__ == "__main__":
    main()
