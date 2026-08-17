"""Retrain V2.2 against its actual per-primary aggregate response vector.

The half-shear catalogue repeats one measured primary response in the coordinate
frame of every sheared neighbour.  Ordinary row MSE treats those correlated
projections as independent scalar labels.  This control instead minimizes

    1/2 |sum_j R(x_ij) u_ij - Delta e_i / g|^2

for each primary i.  It changes only the training objective: pair features,
domain, source cases, preprocessing, tree parameters and tree count remain the
V2.2 recipe.  Groups, not rows, are assigned to the validation split.
"""
from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc
import xgboost as xgb


SBSI_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BE = "/home/z/Zekang.Zhang/blendemu"
for _path in (SBSI_ROOT, os.path.join(SBSI_ROOT, "scripts"), BE, os.path.join(BE, "scripts")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from blendemu import data_utils  # noqa: E402
from blendemu.config import load_config  # noqa: E402
import train_emulator as TE  # noqa: E402
from retrain_emulator_v22 import EXPECTED_CUTS  # noqa: E402


FROZEN_TAG = "lsst_r_extnbr_v22"
OUTPUT_TAG = os.environ.get("GROUP_VECTOR_TAG", "lsst_r_extnbr_v22_groupvec")
RAW_FEATURES = [
    "Re_input_p", "Re_input_s", "r_input_p", "r_input_s",
    "sersic_n_input_p", "sersic_n_input_s", "distance",
]
NEED = [
    "case", "input_index", "shear_angle", "delta_et1", "delta_et2",
    *RAW_FEATURES,
]


@dataclass
class Groups:
    group_id: np.ndarray
    cosine: np.ndarray
    sine: np.ndarray
    target1: np.ndarray
    target2: np.ndarray
    offset: float
    hessian: np.ndarray


def stable_validation_mask(case: np.ndarray, primary: np.ndarray) -> np.ndarray:
    """Deterministic approximately 20% split, constant within each primary."""
    key = (
        case.astype(np.uint64) * np.uint64(0x9E3779B185EBCA87)
        + primary.astype(np.uint64) * np.uint64(0xC2B2AE3D27D4EB4F)
    )
    key ^= key >> np.uint64(30)
    key *= np.uint64(0xBF58476D1CE4E5B9)
    key ^= key >> np.uint64(27)
    return (key % np.uint64(5)) == 0


def make_groups(frame: pd.DataFrame, shear: float, y_mean: float, y_std: float) -> Groups:
    case = frame.case.to_numpy(np.int64, copy=False)
    primary = frame.input_index.to_numpy(np.int64, copy=False)
    change = np.empty(len(frame), dtype=bool)
    change[0] = True
    change[1:] = (case[1:] != case[:-1]) | (primary[1:] != primary[:-1])
    starts = np.flatnonzero(change)
    start_keys = case[starts].astype(np.int64) * np.int64(10_000_000) + primary[starts]
    if len(np.unique(start_keys)) != len(starts):
        raise RuntimeError("primary rows are not contiguous in the response catalogue")
    counts = np.diff(np.r_[starts, len(frame)])
    group_id = np.repeat(np.arange(len(starts), dtype=np.int32), counts)

    phase = np.deg2rad(2.0 * frame.shear_angle.to_numpy(float, copy=False))
    cosine = np.cos(phase).astype(np.float32)
    sine = np.sin(phase).astype(np.float32)
    label = frame.delta_et1.to_numpy(float, copy=False) / shear
    null = frame.delta_et2.to_numpy(float, copy=False) / shear
    global1 = label * cosine - null * sine
    global2 = label * sine + null * cosine
    target1 = (global1[starts] / y_std).astype(np.float64)
    target2 = (global2[starts] / y_std).astype(np.float64)
    # Exact construction repeats the same global vector in each neighbour frame.
    probe = np.linspace(0, len(frame) - 1, min(len(frame), 200_000), dtype=int)
    if (np.max(np.abs(global1[probe] / y_std - target1[group_id[probe]])) > 1e-4
            or np.max(np.abs(global2[probe] / y_std - target2[group_id[probe]])) > 1e-4):
        raise RuntimeError("projected rows do not reconstruct one response vector per primary")
    return Groups(
        group_id=group_id, cosine=cosine, sine=sine,
        target1=target1, target2=target2, offset=y_mean / y_std,
        hessian=np.ones(len(frame), dtype=np.float32),
    )


def aggregate_residual(prediction: np.ndarray, groups: Groups):
    physical_standardized = prediction.astype(np.float64, copy=False) + groups.offset
    n_groups = len(groups.target1)
    sum1 = np.bincount(
        groups.group_id, weights=physical_standardized * groups.cosine,
        minlength=n_groups,
    )
    sum2 = np.bincount(
        groups.group_id, weights=physical_standardized * groups.sine,
        minlength=n_groups,
    )
    return sum1 - groups.target1, sum2 - groups.target2


def objective_for(groups_by_matrix: dict[int, Groups]):
    def objective(prediction: np.ndarray, matrix: xgb.DMatrix):
        groups = groups_by_matrix[id(matrix)]
        residual1, residual2 = aggregate_residual(prediction, groups)
        gradient = (
            residual1[groups.group_id] * groups.cosine
            + residual2[groups.group_id] * groups.sine
        ).astype(np.float32)
        return gradient, groups.hessian
    return objective


def group_rmse(prediction: np.ndarray, groups: Groups) -> float:
    residual1, residual2 = aggregate_residual(prediction, groups)
    return float(np.sqrt(np.mean(residual1 * residual1 + residual2 * residual2)))


def main() -> None:
    cfg = load_config(os.environ["CONFIG_PATH"])
    cuts = np.asarray(cfg["training"]["regression_cuts"], dtype=float)
    if cuts.shape != (5, 2) or not np.array_equal(cuts, np.asarray(EXPECTED_CUTS)):
        raise SystemExit(f"V2.2 regression cuts drifted: {cuts.tolist()}")
    cfg["training"]["model_tag"] = OUTPUT_TAG
    model_path = os.path.join(BE, "models", f"regression_model_{OUTPUT_TAG}.json")
    metadata_path = os.path.join(BE, "models", f"emulator_metadata_{OUTPUT_TAG}.json")
    for path in (model_path, metadata_path):
        if os.path.exists(path):
            raise FileExistsError(f"refusing to overwrite {path}")

    tr = cfg["training"]
    rc = tr["rescale"]
    catalogue = os.path.join(cfg["simulation"]["output_path"], "response_catalogue_train.feather")
    case_min = int(os.environ.get("HELDOUT_MIN_CASE", "40"))
    parts = []
    raw_rows = 0
    with ipc.open_file(catalogue) as reader:
        missing = set(NEED) - set(reader.schema.names)
        if missing:
            raise KeyError(f"response catalogue lacks {sorted(missing)}")
        for batch_index in range(reader.num_record_batches):
            frame = pa.Table.from_batches([reader.get_batch(batch_index)]).select(NEED).to_pandas()
            frame = frame.loc[frame.case.to_numpy(int) >= case_min]
            if frame.empty:
                continue
            raw_rows += len(frame)
            finite = np.isfinite(frame[NEED[2:]].to_numpy(float)).all(axis=1)
            frame = data_utils.source_select_reg(frame.loc[finite], cuts=tr["regression_cuts"])
            if not frame.empty:
                parts.append(frame)
    dataset = pd.concat(parts, ignore_index=True)
    del parts
    print(f"raw rows case>={case_min}: {raw_rows:,}; selected rows: {len(dataset):,}", flush=True)
    shear = TE._catalogue_shear_scale(cfg, "response")
    scalar_label = dataset.delta_et1.to_numpy(float, copy=False) / shear
    y_mean = float(np.mean(scalar_label))
    y_std = float(np.std(scalar_label, ddof=1))
    print(f"row standardization mean={y_mean:.8f} std={y_std:.8f}", flush=True)

    validation = stable_validation_mask(
        dataset.case.to_numpy(np.int64, copy=False),
        dataset.input_index.to_numpy(np.int64, copy=False),
    )
    train_frame = dataset.loc[~validation].reset_index(drop=True)
    validation_frame = dataset.loc[validation].reset_index(drop=True)
    del dataset, scalar_label, validation
    print(f"group-preserving rows train={len(train_frame):,} validation={len(validation_frame):,}", flush=True)
    train_groups = make_groups(train_frame, shear, y_mean, y_std)
    validation_groups = make_groups(validation_frame, shear, y_mean, y_std)
    print(
        f"primaries train={len(train_groups.target1):,} "
        f"validation={len(validation_groups.target1):,}", flush=True,
    )

    train_frame = data_utils.rescale(
        train_frame, pixel_rms=rc["pixel_rms"], pixel_size=rc["pixel_size"],
        zero_mag=rc["zero_mag"], psf_fwhm=rc["psf_fwhm"],
        moffat_beta=rc["moffat_beta"],
    )
    validation_frame = data_utils.rescale(
        validation_frame, pixel_rms=rc["pixel_rms"], pixel_size=rc["pixel_size"],
        zero_mag=rc["zero_mag"], psf_fwhm=rc["psf_fwhm"],
        moffat_beta=rc["moffat_beta"],
    )
    features = tr["features"]
    x_train = train_frame[features]
    x_validation = validation_frame[features]
    # Row labels are unused by the custom objective but keep the DMatrix schema conventional.
    dm_train = xgb.DMatrix(x_train, label=np.zeros(len(x_train), dtype=np.float32))
    dm_validation = xgb.DMatrix(x_validation, label=np.zeros(len(x_validation), dtype=np.float32))
    del train_frame, validation_frame

    frozen_meta = json.load(open(
        os.path.join(BE, "models", f"emulator_metadata_{FROZEN_TAG}.json"),
        encoding="utf-8",
    ))
    frozen_task = frozen_meta["tasks"]["regression"]
    params = dict(frozen_task["params"])
    params.update({
        "objective": "reg:squarederror", "n_jobs": -1,
        "device": os.environ.get("XGB_DEVICE", "cpu"), "tree_method": "hist",
        "booster": "gbtree", "disable_default_eval_metric": 1,
        "base_score": 0.0,
    })
    gamma_override = os.environ.get("GROUP_VECTOR_GAMMA")
    min_child_override = os.environ.get("GROUP_VECTOR_MIN_CHILD_WEIGHT")
    if gamma_override is not None:
        params["gamma"] = float(gamma_override)
    if min_child_override is not None:
        params["min_child_weight"] = float(min_child_override)
    n_trees = int(os.environ.get("GROUP_VECTOR_N_TREES", "271"))
    objective = objective_for({id(dm_train): train_groups, id(dm_validation): validation_groups})
    initial_train = group_rmse(np.zeros(len(x_train), dtype=np.float32), train_groups)
    initial_validation = group_rmse(
        np.zeros(len(x_validation), dtype=np.float32), validation_groups,
    )
    print(
        f"initial standardized vector RMSE train={initial_train:.6f} "
        f"validation={initial_validation:.6f}", flush=True,
    )
    started = time.time()
    booster = xgb.train(
        params, dm_train, obj=objective, num_boost_round=n_trees,
        verbose_eval=False,
    )
    elapsed = time.time() - started
    train_prediction = booster.predict(dm_train)
    validation_prediction = booster.predict(dm_validation)
    final_train = group_rmse(train_prediction, train_groups)
    final_validation = group_rmse(validation_prediction, validation_groups)
    print(
        f"trained {n_trees} trees in {elapsed:.1f}s; vector RMSE "
        f"train={final_train:.6f} validation={final_validation:.6f}", flush=True,
    )

    boundary = np.asarray([[x_train[f].min(), x_train[f].max()] for f in features])
    booster.save_model(model_path)
    metadata_curve = TE._fname(tr["model_dir"], "regression_train_curve.npz", cfg)
    np.savez(
        metadata_curve,
        initial_train_vector_rmse=initial_train,
        initial_validation_vector_rmse=initial_validation,
        final_train_vector_rmse=final_train,
        final_validation_vector_rmse=final_validation,
    )
    TE._update_metadata(
        cfg, "regression", model_path, features, boundary, params,
        standardization=(y_mean, y_std), train_curve_path=metadata_curve,
        metrics={
            "best_iteration": n_trees - 1, "best_trees": n_trees,
            "score_name": "per_primary_group_vector_rmse",
            "train_rows": int(len(x_train)),
            "validation_rows": int(len(x_validation)),
            "train_groups": int(len(train_groups.target1)),
            "validation_groups": int(len(validation_groups.target1)),
            "initial_train_vector_rmse": initial_train,
            "initial_validation_vector_rmse": initial_validation,
            "final_train_vector_rmse": final_train,
            "final_validation_vector_rmse": final_validation,
            "group_objective": {
                "formula": "0.5*norm(sum_j R(x_ij)*u_ij - delta_e_i/g)^2",
                "row_labels_used_as_independent_targets": False,
                "validation_split_unit": "case,input_index primary group",
                "source_cases": [case_min, 199],
                "fixed_tree_count": n_trees,
                "parameter_overrides": {
                    "gamma": None if gamma_override is None else float(gamma_override),
                    "min_child_weight": (
                        None if min_child_override is None else float(min_child_override)
                    ),
                },
            },
        },
    )
    metadata = json.load(open(metadata_path, encoding="utf-8"))
    for task_name in ("classification", "self_response"):
        metadata["tasks"][task_name] = frozen_meta["tasks"][task_name]
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(f"saved model={model_path} metadata={metadata_path}", flush=True)
    print("V22_GROUP_VECTOR_TRAIN_DONE", flush=True)


if __name__ == "__main__":
    main()
