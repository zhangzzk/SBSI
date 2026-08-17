"""Retrain only the V2.2 pairwise blend-response task at forward g=0.05.

The V2.2 cuts, features, XGBoost parameters, and row-level early-stopping split
are inherited unchanged.  Catalogue paths, the case window, and the isolated
output tag can be overridden through ``G005_*`` environment variables.  The
defaults reproduce the original 60-case ablation.  Constgold and coherent-anchor
truth are never read.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc
import xgboost as xgb
from sklearn.metrics import r2_score
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
TAG = os.environ.get("G005_MODEL_TAG", "lsst_r_extnbr_v22_g005")
DEFAULT_CATALOGUE = (
    "/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876/"
    "response_catalogue_g005_train.feather"
)
CATALOGUES = [
    path for path in os.environ.get("G005_CATALOGUES", DEFAULT_CATALOGUE).split(":")
    if path
]
SHEAR = 0.05
TRAIN_CASE_MIN = int(os.environ.get("G005_TRAIN_CASE_MIN", "40"))
TRAIN_CASE_MAX = int(os.environ.get("G005_TRAIN_CASE_MAX", "99"))
EXPECTED_TRAIN_CASES = int(os.environ.get("G005_EXPECTED_TRAIN_CASES", "60"))
EXPECTED_CUTS = np.asarray(
    [[13, 29], [18, 25.8], [0, 10], [0.5, 1.5], [0, 10]], float,
)
NEED = [
    "case", "r_input_p", "r_input_s", "Re_input_p", "Re_input_s",
    "sersic_n_input_p", "sersic_n_input_s", "distance", "delta_et1",
]


def load_training_data(cfg):
    training = cfg["training"]
    rescale = training["rescale"]
    parts = []
    raw_rows = 0
    cases = set()
    source_case_sets = []
    for catalogue in CATALOGUES:
        source_cases = set()
        with ipc.open_file(catalogue) as reader:
            missing = set(NEED) - set(reader.schema.names)
            if missing:
                raise KeyError(f"g=0.05 response catalogue lacks {sorted(missing)}: {catalogue}")
            for batch_index in range(reader.num_record_batches):
                frame = pa.Table.from_batches([reader.get_batch(batch_index)]).select(NEED).to_pandas()
                case = frame["case"].to_numpy(int)
                frame = frame[(case >= TRAIN_CASE_MIN) & (case <= TRAIN_CASE_MAX)]
                if not len(frame):
                    continue
                raw_rows += len(frame)
                source_cases.update(frame["case"].astype(int).unique().tolist())
                selected = data_utils.source_select_reg(frame, cuts=training["regression_cuts"])
                selected = selected[np.isfinite(selected["delta_et1"].to_numpy(float))]
                if len(selected):
                    parts.append(selected)
        overlap = cases & source_cases
        if overlap:
            raise RuntimeError(
                f"catalogue case windows overlap for cases {sorted(overlap)[:10]}"
            )
        cases.update(source_cases)
        source_case_sets.append(sorted(source_cases))
    if not parts:
        raise RuntimeError("no g=0.05 training rows survived")
    dataset = pd.concat(parts, ignore_index=True)
    expected_cases = set(range(TRAIN_CASE_MIN, TRAIN_CASE_MAX + 1))
    if cases != expected_cases or len(cases) != EXPECTED_TRAIN_CASES:
        raise RuntimeError(
            f"unexpected g=0.05 training cases {min(cases)}--{max(cases)} "
            f"({len(cases)}); missing={sorted(expected_cases - cases)[:10]} "
            f"extra={sorted(cases - expected_cases)[:10]} expected_count={EXPECTED_TRAIN_CASES}"
        )
    print(
        f"catalogues={CATALOGUES}\nsource case sets="
        f"{[(s[0], s[-1], len(s)) for s in source_case_sets if s]}\n"
        f"training cases={min(cases)}--{max(cases)} "
        f"({len(cases)} cases) raw={raw_rows:,} selected={len(dataset):,}",
        flush=True,
    )
    dataset = data_utils.rescale(
        dataset, pixel_rms=rescale["pixel_rms"], pixel_size=rescale["pixel_size"],
        zero_mag=rescale["zero_mag"], psf_fwhm=rescale["psf_fwhm"],
        moffat_beta=rescale["moffat_beta"],
    )
    response = dataset["delta_et1"].to_numpy(float) / SHEAR
    response_mean = float(response.mean())
    response_std = float(response.std(ddof=1))
    target = data_utils.standardize(response, response_mean, response_std)
    features = training["features"]
    x_train, x_validation, y_train, y_validation = train_test_split(
        dataset[features], target, test_size=training["test_size"],
        random_state=training["random_state"],
    )
    print(
        f"label mean={response_mean:+.8f} std={response_std:.8f}; "
        f"train={len(x_train):,} validation={len(x_validation):,}",
        flush=True,
    )
    return (
        xgb.DMatrix(x_train, y_train), xgb.DMatrix(x_validation, y_validation),
        x_train, x_validation, y_train, y_validation, response_mean, response_std,
        sorted(cases), raw_rows, len(dataset),
    )


def physical_summary(name, booster, frame, target, mean, std):
    prediction = data_utils.reverse_standardize(
        booster.predict(
            xgb.DMatrix(frame),
            iteration_range=data_utils.get_xgb_iteration_range(booster),
        ), mean, std,
    )
    label = data_utils.reverse_standardize(np.asarray(target), mean, std)
    residual = prediction - label
    print(
        f"{name}: N={len(label):,} label={label.mean():+.8f} "
        f"prediction={prediction.mean():+.8f} residual={residual.mean():+.8f} "
        f"residual_std={residual.std(ddof=1):.8f} R2={r2_score(label, prediction):.8f}",
        flush=True,
    )


def main():
    cfg = load_config(os.environ["CONFIG_PATH"])
    training = cfg["training"]
    if training["model_tag"] != TAG:
        raise SystemExit(f"wrong output tag {training['model_tag']!r}")
    if not np.array_equal(np.asarray(training["regression_cuts"], float), EXPECTED_CUTS):
        raise SystemExit(f"V2.2 regression cuts drifted: {training['regression_cuts']}")
    if cfg["simulation"]["response"]["shear_values"] != [0.0, SHEAR]:
        raise SystemExit("configuration is not the predeclared 0 -> 0.05 response ablation")

    base_metadata_path = os.path.join(BE, f"models/emulator_metadata_{BASE_TAG}.json")
    with open(base_metadata_path, encoding="utf-8") as handle:
        base_metadata = json.load(handle)
    base_task = base_metadata["tasks"]["regression"]
    if list(training["features"]) != list(base_task["features"]):
        raise SystemExit("V2.2 feature list drifted")

    loaded = load_training_data(cfg)
    (
        dm_train, dm_validation, x_train, x_validation, y_train, y_validation,
        response_mean, response_std, cases, raw_rows, selected_rows,
    ) = loaded
    params = dict(base_task["params"])
    params.update(
        objective="reg:squarederror", n_jobs=-1,
        device=os.environ.get("XGB_DEVICE", "cpu"), tree_method="hist",
        booster="gbtree", disable_default_eval_metric=1,
    )
    print(f"parameters inherited from {BASE_TAG}: {params}", flush=True)
    metric = lambda prediction, data: ("r2", r2_score(data.get_label(), prediction))
    history = {}
    start = time.time()
    booster = xgb.train(
        params, dm_train, evals=[(dm_train, "train"), (dm_validation, "eval")],
        evals_result=history, num_boost_round=2000, verbose_eval=100,
        custom_metric=metric, maximize=True,
        callbacks=[TE._early_stopping_callback(cfg, "regression", "r2", maximize=True)],
    )
    trees = data_utils.get_xgb_iteration_range(booster)[1]
    print(
        f"fit seconds={time.time() - start:.1f} best_iter={booster.best_iteration} "
        f"trees={trees}", flush=True,
    )
    physical_summary(
        "train", booster, x_train, y_train, response_mean, response_std,
    )
    physical_summary(
        "row-validation", booster, x_validation, y_validation,
        response_mean, response_std,
    )

    model_path = TE._fname(training["model_dir"], "regression_model.json", cfg)
    curve_path = TE._fname(training["model_dir"], "regression_train_curve.npz", cfg)
    metadata_path = TE._fname(training["model_dir"], "emulator_metadata.json", cfg)
    for path in (model_path, curve_path, metadata_path):
        if os.path.exists(path):
            raise FileExistsError(f"refusing to overwrite {path}")
    boundary = np.asarray([
        [x_train[feature].min(), x_train[feature].max()]
        for feature in training["features"]
    ])
    booster.save_model(model_path)
    np.savez(
        curve_path, train_r2=history["train"]["r2"],
        eval_r2=history["eval"]["r2"],
    )
    metadata_path = TE._update_metadata(
        cfg, "regression", model_path, training["features"], boundary, params,
        standardization=(response_mean, response_std), train_curve_path=curve_path,
        metrics={
            "best_iteration": booster.best_iteration,
            "best_trees": trees,
            "best_score": booster.best_score,
            "score_name": "r2",
            "params_inherited_from": BASE_TAG,
            "train_rows": len(x_train),
            "row_validation_rows": len(x_validation),
            "source_catalogues": CATALOGUES,
            "response_shear": SHEAR,
            "training_cases": cases,
            "heldout_cases": list(range(TRAIN_CASE_MIN)),
            "raw_training_rows": raw_rows,
            "selected_training_rows": selected_rows,
            "uncertainty_unit": "case",
            "sbsi_domain": {
                "domain": "v2.2", "primary_mag_max": 25.8,
                "primary_re_min_arcsec": 0.5,
            },
        },
    )
    with open(metadata_path, encoding="utf-8") as handle:
        metadata = json.load(handle)
    for task in ("classification", "self_response"):
        metadata["tasks"][task] = base_metadata["tasks"][task]
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(
        f"saved model={model_path}\nmetadata={metadata_path}\n"
        "V22_G005_EMULATOR_TRAIN_DONE",
        flush=True,
    )


if __name__ == "__main__":
    main()
