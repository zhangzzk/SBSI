#!/usr/bin/env python3
"""Retrain the frozen half-shear-selected correction recipe on c40--199."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import xgboost as xgb

from scripts.train_v22_grouped_rscene_hparam_candidate import (
    atomic_model,
    residual_summary,
)
from scripts.v22_grouped_rscene_common import (
    CASE_MAX,
    CONDITIONAL_FEATURES,
    FIT_CASE_MIN,
    conditional_matrix,
    load_scene_metadata,
    load_source_metadata,
    mmap_array,
    sha256,
    strict_json,
)
from scripts.v22_grouped_rscene_hparam_common import (
    candidate_by_id,
    physical_prediction,
    select_case_rows,
    xgb_params,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-cache", required=True)
    parser.add_argument("--scene-cache", required=True)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--output-model", required=True)
    parser.add_argument("--output-summary", required=True)
    args = parser.parse_args()

    selection_path = Path(args.selection).resolve()
    with selection_path.open(encoding="utf-8") as handle:
        selection_payload = json.load(handle)
    selected_id = selection_payload["selection"]["selected_candidate_id"]
    if selected_id is None:
        raise RuntimeError("hyperparameter evaluation did not select a candidate")
    selected_id = int(selected_id)
    config = candidate_by_id(selected_id)
    selected_rows = [
        row for row in selection_payload["selection"]["candidate_rows"]
        if row["candidate_id"] == selected_id
    ]
    if len(selected_rows) != 1:
        raise RuntimeError("selected candidate row is missing or duplicated")
    best_rounds = int(selected_rows[0]["best_rounds"])
    if best_rounds <= 0:
        raise RuntimeError("selected tree count is invalid")
    if selection_payload["provenance"]["external_final_cases_20_39_opened"]:
        raise RuntimeError("selection opened the reserved half-shear test block")
    if selection_payload["provenance"]["coherent_anchor_truth_opened"]:
        raise RuntimeError("selection opened coherent-anchor truth")
    if selection_payload["provenance"]["constgold_opened"]:
        raise RuntimeError("selection opened ConstGold")

    source_cache = Path(args.source_cache).resolve()
    scene_cache = Path(args.scene_cache).resolve()
    model_path = Path(args.output_model).resolve()
    summary_path = Path(args.output_summary).resolve()
    model_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    if model_path.exists() or summary_path.exists():
        raise FileExistsError("refusing to overwrite final tuned outputs")

    source_meta = load_source_metadata(source_cache)
    scene_meta = load_scene_metadata(scene_cache)
    if scene_meta["source_metadata_sha256"] != sha256(source_cache / "metadata.json"):
        raise RuntimeError("scene/source cache provenance differs")
    case = mmap_array(source_cache, source_meta, "case")
    official_train = mmap_array(source_cache, source_meta, "official_train")
    label = mmap_array(source_cache, source_meta, "label")
    pair_prediction = mmap_array(source_cache, source_meta, "v22_prediction")
    x_scaled = mmap_array(source_cache, source_meta, "x_scaled")
    x_raw = mmap_array(source_cache, source_meta, "x_raw")
    scene_prediction = mmap_array(scene_cache, scene_meta, "scene_prediction")
    scene_n_pairs = mmap_array(scene_cache, scene_meta, "scene_n_pairs")
    scene_bin = mmap_array(scene_cache, scene_meta, "scene_bin")

    fit_window = (FIT_CASE_MIN, CASE_MAX)
    index = select_case_rows(
        case, official_train, fit_window, official_validation_only=True
    )
    features = conditional_matrix(
        x_scaled,
        x_raw,
        pair_prediction,
        scene_prediction,
        scene_n_pairs,
        index,
    )
    physical_target = (
        np.asarray(label[index], dtype=np.float64)
        - np.asarray(pair_prediction[index], dtype=np.float64)
    )
    target_scale = float(source_meta["source_standardization"]["std"])
    target = (physical_target / target_scale).astype(np.float32)
    matrix = xgb.DMatrix(
        features, label=target, feature_names=CONDITIONAL_FEATURES
    )
    params = xgb_params(config)
    started = time.time()
    booster = xgb.train(
        params,
        matrix,
        num_boost_round=best_rounds,
        verbose_eval=False,
    )
    elapsed = time.time() - started
    if booster.num_boosted_rounds() != best_rounds:
        raise RuntimeError("final booster tree count differs from frozen selection")
    correction = physical_prediction(booster, features, target_scale)
    diagnostics = residual_summary(
        physical_target,
        correction,
        np.asarray(scene_bin[index], dtype=np.uint8),
    )
    atomic_model(booster, model_path)

    payload = {
        "schema_version": 1,
        "kind": "half-shear-selected final V2.2 additive pair-residual correction",
        "base_model": "lsst_r_extnbr_v22",
        "feature_names": CONDITIONAL_FEATURES,
        "target": "half-shear pair label minus frozen V2.2 pair prediction",
        "target_scale": target_scale,
        "configuration": config,
        "selected_candidate_id": selected_id,
        "selected_candidate_name": config["name"],
        "selection": str(selection_path),
        "selection_sha256": sha256(selection_path),
        "model": str(model_path),
        "model_sha256": sha256(model_path),
        "source_cache": str(source_cache),
        "source_metadata_sha256": sha256(source_cache / "metadata.json"),
        "scene_cache": str(scene_cache),
        "scene_metadata_sha256": sha256(scene_cache / "metadata.json"),
        "training": {
            "case_window": list(fit_window),
            "split": "official frozen-V2.2 random-row validation rows only",
            "n_rows": int(len(index)),
            "trees": best_rounds,
            "tree_count_source": "best iteration selected without c20--39, anchors, or ConstGold",
            "params": params,
            "fit_seconds": elapsed,
        },
        "in_sample_diagnostics": diagnostics,
        "protocol": {
            "external_final_cases_20_39_opened": False,
            "coherent_anchor_truth_opened": False,
            "constgold_opened": False,
        },
    }
    strict_json(summary_path, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    print(
        f"V22_RSCENE_HPARAM_FINAL_DONE candidate={selected_id} "
        f"name={config['name']} trees={best_rounds}",
        flush=True,
    )


if __name__ == "__main__":
    main()
