#!/usr/bin/env python3
"""Train one fixed base-plus-correction stack with full-neighbour context.

This is a controlled follow-up to the corrected-own-scene Optuna study.  It
reuses that study's selected base and correction hyperparameters without any
new tuning.  Pair labels and all supervised losses remain restricted to the
sheared half-neighbour response catalogue.  The correction's scene features,
and the coordinate used to assess scene-conditioned residuals, instead sum the
model over the deployed all-object neighbour list around each primary.

Five rotating case-disjoint folds provide one clean OOF prediction per labelled
pair.  In each rotation 80 cases fit the base, 80 different cases fit the
correction, and 40 cases are scored.  A deployment stack is then trained on all
200 cases, with its correction target built from five-fold base-OOF predictions.
No coherent anchors or ConstGold data are opened.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import xgboost as xgb

from scripts.tune_crossfit_base_ownscene_optuna import atomic_npy
from scripts.tune_crossfit_sequential_rblend_optuna import (
    CrossfitExperiment,
    N_CASES,
    N_FOLDS,
)
from scripts.tune_sequential_rblend_optuna import (
    BASE_FEATURES,
    CONDITIONAL_FEATURES,
    R2_GAP_WEIGHT,
    SCENE_WEIGHT,
    atomic_json,
    atomic_model,
    json_clean,
    r2_score,
    xgb_common,
)
from scripts.v22_grouped_rscene_common import (
    assign_scene_bins,
    case_offsets,
    make_scene_edges,
    sha256,
)


N_SCENE_BINS = 20


class FullNeighbourExperiment(CrossfitExperiment):
    """Sequential cross-fit whose scene coordinates use all neighbours."""

    def __init__(
        self,
        source_cache: Path,
        scene_cache: Path,
        full_cache: Path,
    ) -> None:
        super().__init__(source_cache, scene_cache)
        self.full_cache = full_cache
        with (full_cache / "metadata.json").open(encoding="utf-8") as handle:
            self.full_meta = json.load(handle)
        if self.full_meta["source_metadata_sha256"] != sha256(
            source_cache / "metadata.json"
        ):
            raise RuntimeError("full-neighbour cache does not match source cache")
        if self.full_meta["case_window"] != [0, N_CASES - 1]:
            raise RuntimeError("full-neighbour cache lacks cases 0--199")
        if self.full_meta["base_features"] != BASE_FEATURES:
            raise RuntimeError("full-neighbour base features drifted")
        neighbour = self.full_meta["neighbour_definition"]
        if float(neighbour["r_max_arcsec"]) != 10.0 or int(neighbour["k"]) != 20:
            raise RuntimeError("full-neighbour cache is not deployed r_max/k")
        self.source_offsets = case_offsets(self.source_meta)
        scene_case_count = np.bincount(
            self.scene_case.astype(np.int64), minlength=N_CASES
        )
        self.scene_offsets = np.concatenate(([0], np.cumsum(scene_case_count)))
        if int(self.scene_offsets[-1]) != self.n_scenes:
            raise RuntimeError("scene case offsets do not close")
        self.full_count = np.empty(self.n_scenes, dtype=np.int16)
        for case in range(N_CASES):
            full = self.load_full(case, static_only=True)
            mapping = self.full_to_source_scene_mapping(case, full["scene_primary"])
            s0, s1 = self.scene_offsets[case : case + 2]
            self.full_count[s0:s1] = full["scene_count"][mapping]
        print(
            "Loaded full-neighbour context: "
            f"pairs={self.full_meta['n_pairs']:,}, "
            f"mean_all_neighbours={self.full_count.mean():.3f}, "
            f"mean_labelled_neighbours={self.scene_counts.mean():.3f}, "
            f"zero_all_neighbour_scenes={np.count_nonzero(self.full_count == 0):,}",
            flush=True,
        )

    def full_path(self, name: str, case: int) -> Path:
        pattern = self.full_meta["arrays"][name]
        return self.full_cache / pattern.format(case=case)

    def load_full(
        self, case: int, *, static_only: bool = False
    ) -> dict[str, np.ndarray]:
        names = ("scene_primary", "scene_count") if static_only else (
            "x_scaled",
            "x_aux",
            "pair_scene",
            "scene_primary",
            "scene_count",
        )
        return {
            name: np.load(self.full_path(name, case), mmap_mode="r")
            for name in names
        }

    def full_to_source_scene_mapping(
        self, case: int, full_primary: np.ndarray
    ) -> np.ndarray:
        s0, s1 = self.scene_offsets[case : case + 2]
        active_primary = np.asarray(
            self.primary[self.scene_starts[s0:s1]], dtype=np.int64
        )
        full_primary = np.asarray(full_primary, dtype=np.int64)
        if len(active_primary) != len(full_primary):
            raise RuntimeError(f"case {case}: full/labelled scene count differs")
        mapping = np.searchsorted(full_primary, active_primary)
        if (
            np.any(mapping >= len(full_primary))
            or not np.array_equal(full_primary[mapping], active_primary)
        ):
            raise RuntimeError(f"case {case}: full/labelled primary IDs differ")
        return mapping.astype(np.int32)

    def physical_full_base_prediction(
        self, booster: xgb.Booster, x_scaled: np.ndarray
    ) -> np.ndarray:
        prediction = booster.inplace_predict(
            np.asarray(x_scaled, dtype=np.float32)
        ).astype(np.float64)
        prediction = prediction * self.target_scale + self.target_mean
        if not np.isfinite(prediction).all():
            raise RuntimeError("base returned non-finite full-neighbour prediction")
        return prediction.astype(np.float32)

    def full_base_context(
        self,
        booster: xgb.Booster,
        cases: np.ndarray,
        *,
        keep_pair_cases: set[int],
    ) -> dict[int, dict[str, np.ndarray]]:
        output: dict[int, dict[str, np.ndarray]] = {}
        for case_value in np.sort(np.asarray(cases, dtype=np.int64)):
            case = int(case_value)
            full = self.load_full(case)
            pair_prediction = self.physical_full_base_prediction(
                booster, full["x_scaled"]
            )
            n_scene = len(full["scene_primary"])
            scene_prediction = np.bincount(
                np.asarray(full["pair_scene"], dtype=np.int64),
                weights=pair_prediction.astype(np.float64),
                minlength=n_scene,
            ).astype(np.float32)
            record = {
                "scene_primary": np.asarray(full["scene_primary"]),
                "scene_count": np.asarray(full["scene_count"]),
                "scene_prediction": scene_prediction,
            }
            if case in keep_pair_cases:
                record["pair_prediction"] = pair_prediction
            output[case] = record
        return output

    def conditional_features_context(
        self,
        index: np.ndarray,
        pair_prediction: np.ndarray,
        context: dict[int, dict[str, np.ndarray]],
    ) -> np.ndarray:
        prediction = np.asarray(pair_prediction, dtype=np.float32)
        if prediction.shape != (len(index),):
            raise RuntimeError("active conditional pair length mismatch")
        scaled = np.asarray(self.x_scaled[index], dtype=np.float32)
        raw = np.asarray(self.x_raw[index], dtype=np.float32)
        output = np.empty((len(index), len(CONDITIONAL_FEATURES)), dtype=np.float32)
        output[:, :7] = scaled
        output[:, 7] = prediction
        output[:, 8] = -0.4 * (raw[:, 3] - raw[:, 2])
        output[:, 9] = np.log10(raw[:, 1] / raw[:, 0])
        selected_case = np.asarray(self.case[index], dtype=np.int16)
        selected_primary = np.asarray(self.primary[index], dtype=np.int64)
        for case_value in np.unique(selected_case):
            case = int(case_value)
            lo = int(np.searchsorted(selected_case, case_value, side="left"))
            hi = int(np.searchsorted(selected_case, case_value, side="right"))
            full_primary = context[case]["scene_primary"]
            mapping = np.searchsorted(full_primary, selected_primary[lo:hi])
            if (
                np.any(mapping >= len(full_primary))
                or not np.array_equal(
                    full_primary[mapping], selected_primary[lo:hi]
                )
            ):
                raise RuntimeError(f"case {case}: active row lacks full scene")
            output[lo:hi, 10] = context[case]["scene_prediction"][mapping]
            output[lo:hi, 11] = np.log1p(
                context[case]["scene_count"][mapping].astype(np.float32)
            )
        if not np.isfinite(output).all():
            raise RuntimeError("active full-context feature is non-finite")
        return output

    def correction_matrix_context(
        self,
        index: np.ndarray,
        base_prediction: np.ndarray,
        context: dict[int, dict[str, np.ndarray]],
    ) -> xgb.DMatrix:
        features = self.conditional_features_context(
            index, base_prediction, context
        )
        target = (
            np.asarray(self.label[index], dtype=np.float64) - base_prediction
        ) / self.target_scale
        matrix = xgb.DMatrix(
            features,
            label=target.astype(np.float32),
            feature_names=CONDITIONAL_FEATURES,
        )
        del features, target
        return matrix

    def global_context_features(
        self,
        pair_prediction: np.ndarray,
        scene_prediction: np.ndarray,
    ) -> np.ndarray:
        pair_prediction = np.asarray(pair_prediction, dtype=np.float32)
        scene_prediction = np.asarray(scene_prediction, dtype=np.float32)
        if pair_prediction.shape != (self.n_rows,):
            raise RuntimeError("global pair prediction length mismatch")
        if scene_prediction.shape != (self.n_scenes,):
            raise RuntimeError("global scene prediction length mismatch")
        output = np.empty(
            (self.n_rows, len(CONDITIONAL_FEATURES)), dtype=np.float32
        )
        output[:, :7] = np.asarray(self.x_scaled, dtype=np.float32)
        output[:, 7] = pair_prediction
        raw = np.asarray(self.x_raw, dtype=np.float32)
        output[:, 8] = -0.4 * (raw[:, 3] - raw[:, 2])
        output[:, 9] = np.log10(raw[:, 1] / raw[:, 0])
        output[:, 10] = np.repeat(scene_prediction, self.scene_counts)
        output[:, 11] = np.log1p(
            np.repeat(self.full_count, self.scene_counts).astype(np.float32)
        )
        if not np.isfinite(output).all():
            raise RuntimeError("global full-context feature is non-finite")
        return output

    def corrected_full_coordinate(
        self,
        correction: xgb.Booster,
        context: dict[int, dict[str, np.ndarray]],
        cases: np.ndarray,
    ) -> dict[int, np.ndarray]:
        output: dict[int, np.ndarray] = {}
        for case_value in np.sort(np.asarray(cases, dtype=np.int64)):
            case = int(case_value)
            full = self.load_full(case)
            base_pair = context[case].get("pair_prediction")
            if base_pair is None:
                raise RuntimeError(f"case {case}: full pair prediction not retained")
            pair_scene = np.asarray(full["pair_scene"], dtype=np.int64)
            features = np.empty(
                (len(base_pair), len(CONDITIONAL_FEATURES)), dtype=np.float32
            )
            features[:, :7] = np.asarray(full["x_scaled"], dtype=np.float32)
            features[:, 7] = base_pair
            features[:, 8:10] = np.asarray(full["x_aux"], dtype=np.float32)
            features[:, 10] = context[case]["scene_prediction"][pair_scene]
            features[:, 11] = np.log1p(
                context[case]["scene_count"][pair_scene].astype(np.float32)
            )
            correction_pair = correction.inplace_predict(features).astype(np.float64)
            correction_pair *= self.target_scale
            combined = base_pair.astype(np.float64) + correction_pair
            n_scene = len(context[case]["scene_primary"])
            output[case] = np.bincount(
                pair_scene, weights=combined, minlength=n_scene
            ).astype(np.float32)
            del features, correction_pair, combined
        return output

    def put_scene_coordinate(
        self,
        destination: np.ndarray,
        by_case: dict[int, np.ndarray],
    ) -> None:
        for case, full_coordinate in by_case.items():
            full = self.load_full(case, static_only=True)
            mapping = self.full_to_source_scene_mapping(case, full["scene_primary"])
            s0, s1 = self.scene_offsets[case : case + 2]
            destination[s0:s1] = full_coordinate[mapping]

    def full_coordinate_metrics(
        self,
        active_pair_prediction: np.ndarray,
        full_scene_coordinate: np.ndarray,
    ) -> dict[str, Any]:
        active_scene_prediction = np.add.reduceat(
            np.asarray(active_pair_prediction, dtype=np.float64), self.scene_starts
        )
        coordinate = np.asarray(full_scene_coordinate, dtype=np.float64)
        if coordinate.shape != (self.n_scenes,) or not np.isfinite(coordinate).all():
            raise RuntimeError("full scene coordinate is incomplete")
        residual = self.truth_scene - active_scene_prediction
        edges = make_scene_edges(coordinate, n_quantile_bins=N_SCENE_BINS)
        scene_bin = assign_scene_bins(coordinate, edges).astype(np.int64)
        flat = self.scene_case.astype(np.int64) * N_SCENE_BINS + scene_bin
        count = np.bincount(
            flat, minlength=N_CASES * N_SCENE_BINS
        ).reshape(N_CASES, N_SCENE_BINS)
        total = np.bincount(
            flat, weights=residual, minlength=N_CASES * N_SCENE_BINS
        ).reshape(N_CASES, N_SCENE_BINS)
        if np.any(count == 0):
            raise RuntimeError("full-neighbour bins leave an empty case/bin cell")
        curve = (total / count).mean(axis=0)
        scene_count = np.bincount(scene_bin, minlength=N_SCENE_BINS)
        prediction_mean = np.bincount(
            scene_bin, weights=coordinate, minlength=N_SCENE_BINS
        ) / scene_count
        active_prediction_mean = np.bincount(
            scene_bin, weights=active_scene_prediction, minlength=N_SCENE_BINS
        ) / scene_count
        truth_mean = np.bincount(
            scene_bin, weights=self.truth_scene, minlength=N_SCENE_BINS
        ) / scene_count
        count_case = np.bincount(self.scene_case, minlength=N_CASES)
        mean_case = np.bincount(
            self.scene_case, weights=residual, minlength=N_CASES
        ) / count_case
        output: dict[str, Any] = {
            "coordinate": "sum of final corrected predictions over all deployed neighbours",
            "residual": "sum over sheared labelled neighbours of (label-prediction)",
            "edges": edges,
            "edge_convention": "lower-inclusive, upper-exclusive",
            "curve": curve,
            "curve_rms": float(np.sqrt(np.mean(np.square(curve)))),
            "coordinate_mean": prediction_mean,
            "active_prediction_mean": active_prediction_mean,
            "truth_mean": truth_mean,
            "scene_count": scene_count,
            "case_balanced_mean": float(mean_case.mean()),
            "case_sem": float(mean_case.std(ddof=1) / np.sqrt(N_CASES)),
            "row_weighted_mean": float(residual.mean()),
        }
        for name, mask in {
            "tail_gt_0p1": coordinate > 0.1,
            "tail_gt_0p2": coordinate > 0.2,
        }.items():
            local_case = self.scene_case[mask]
            local_residual = residual[mask]
            local_count = np.bincount(local_case, minlength=N_CASES)
            local_total = np.bincount(
                local_case, weights=local_residual, minlength=N_CASES
            )
            present = local_count > 0
            local_mean = local_total[present] / local_count[present]
            output[name] = {
                "n_scenes": int(mask.sum()),
                "n_cases": int(present.sum()),
                "case_balanced_mean": float(local_mean.mean()),
                "case_sem": float(
                    local_mean.std(ddof=1) / np.sqrt(len(local_mean))
                ),
            }
        return output

    def fit_crossfit(
        self, stage_params: dict[str, Any]
    ) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
        started = time.time()
        combined_oof = np.full(self.n_rows, np.nan, dtype=np.float32)
        full_coordinate_oof = np.full(self.n_scenes, np.nan, dtype=np.float32)
        train_r2_by_fold: list[float] = []
        fold_records: list[dict[str, Any]] = []
        base_params = xgb_common() | stage_params["base"]
        correction_params = xgb_common() | stage_params["correction"]
        base_rounds = int(stage_params["base_rounds"])
        correction_rounds = int(stage_params["correction_rounds"])

        for outer_fold in range(N_FOLDS):
            fold_started = time.time()
            (
                base_index,
                correction_index,
                outer_index,
                base_folds,
                correction_folds,
            ) = self.split_indices(outer_fold)
            dbase_train = self.base_matrix(base_index)
            dbase_correction = self.base_matrix(correction_index)
            dbase_outer = self.base_matrix(outer_index)
            base = xgb.train(
                base_params, dbase_train, num_boost_round=base_rounds,
                verbose_eval=False,
            )
            base_correction = self.physical_base_prediction(base, dbase_correction)
            base_outer = self.physical_base_prediction(base, dbase_outer)
            outer_cases = np.flatnonzero(self.case_fold == outer_fold)
            correction_cases = np.flatnonzero(
                np.isin(self.case_fold, correction_folds)
            )
            context = self.full_base_context(
                base,
                np.concatenate((correction_cases, outer_cases)),
                keep_pair_cases=set(outer_cases.astype(int)),
            )
            del dbase_train, dbase_correction, dbase_outer, base
            gc.collect()

            dcorrection_train = self.correction_matrix_context(
                correction_index, base_correction, context
            )
            dcorrection_outer = self.correction_matrix_context(
                outer_index, base_outer, context
            )
            correction = xgb.train(
                correction_params,
                dcorrection_train,
                num_boost_round=correction_rounds,
                verbose_eval=False,
            )
            correction_train = (
                correction.predict(dcorrection_train).astype(np.float64)
                * self.target_scale
            )
            correction_outer = (
                correction.predict(dcorrection_outer).astype(np.float64)
                * self.target_scale
            )
            combined_train = base_correction + correction_train
            combined_outer = base_outer + correction_outer
            fold_train_r2 = r2_score(
                self.label[correction_index], combined_train
            )
            fold_outer_r2 = r2_score(self.label[outer_index], combined_outer)
            train_r2_by_fold.append(fold_train_r2)
            combined_oof[outer_index] = combined_outer.astype(np.float32)
            by_case = self.corrected_full_coordinate(
                correction, context, outer_cases
            )
            self.put_scene_coordinate(full_coordinate_oof, by_case)
            fold_records.append({
                "outer_fold": outer_fold,
                "outer_cases": outer_cases,
                "base_folds": base_folds,
                "correction_folds": correction_folds,
                "n_base_rows": len(base_index),
                "n_correction_rows": len(correction_index),
                "n_outer_rows": len(outer_index),
                "base_rounds": base_rounds,
                "correction_rounds": correction_rounds,
                "correction_fit_r2": fold_train_r2,
                "outer_r2": fold_outer_r2,
                "fit_seconds": time.time() - fold_started,
            })
            print(
                f"FOLD_DONE outer={outer_fold} r2={fold_outer_r2:+.7f} "
                f"trees={base_rounds}+{correction_rounds} "
                f"seconds={fold_records[-1]['fit_seconds']:.1f}",
                flush=True,
            )
            del (
                dcorrection_train,
                dcorrection_outer,
                correction,
                base_correction,
                base_outer,
                correction_train,
                correction_outer,
                combined_train,
                combined_outer,
                context,
                by_case,
                base_index,
                correction_index,
                outer_index,
            )
            gc.collect()

        if not np.isfinite(combined_oof).all():
            raise RuntimeError("combined active OOF prediction is incomplete")
        if not np.isfinite(full_coordinate_oof).all():
            raise RuntimeError("full-neighbour OOF coordinate is incomplete")
        pair = self.pair_metrics(combined_oof)
        full_scene = self.full_coordinate_metrics(
            combined_oof, full_coordinate_oof
        )
        fixed_scene = self.scene_metrics(combined_oof)
        train_r2 = float(np.mean(train_r2_by_fold))
        pair_score = float(pair["r2"]) - R2_GAP_WEIGHT * abs(
            train_r2 - float(pair["r2"])
        )
        scene_standardized = float(full_scene["curve_rms"]) / self.target_scale
        metrics = {
            "train_r2_mean_over_folds": train_r2,
            "oof_r2": pair["r2"],
            "r2_train_oof_gap_abs": abs(train_r2 - float(pair["r2"])),
            "historical_pair_score": pair_score,
            "oof_fullneighbour_scene_curve_rms": full_scene["curve_rms"],
            "oof_fullneighbour_scene_curve_rms_standardized": scene_standardized,
            "descriptive_pair_plus_scene_score": (
                pair_score - SCENE_WEIGHT * scene_standardized
            ),
            "oof_pair_metrics": pair,
            "oof_fullneighbour_scene_metrics": full_scene,
            "oof_fixed_v22_active_scene_metrics": fixed_scene,
            "folds": fold_records,
            "fit_seconds": time.time() - started,
        }
        return metrics, combined_oof, full_coordinate_oof

    def fit_deployment(
        self, stage_params: dict[str, Any], output_dir: Path
    ) -> dict[str, Any]:
        base_rounds = int(stage_params["base_rounds"])
        correction_rounds = int(stage_params["correction_rounds"])
        base_params = xgb_common() | stage_params["base"]
        correction_params = xgb_common() | stage_params["correction"]
        all_index = np.arange(self.n_rows, dtype=np.int32)

        base_oof = np.full(self.n_rows, np.nan, dtype=np.float32)
        scene_base_oof = np.full(self.n_scenes, np.nan, dtype=np.float32)
        for outer_fold in range(N_FOLDS):
            outer_index = self.fold_index[outer_fold]
            train_index = np.sort(np.concatenate([
                self.fold_index[fold]
                for fold in range(N_FOLDS)
                if fold != outer_fold
            ]))
            dtrain = self.base_matrix(train_index)
            douter = self.base_matrix(outer_index)
            base = xgb.train(
                base_params, dtrain, num_boost_round=base_rounds,
                verbose_eval=False,
            )
            base_oof[outer_index] = self.physical_base_prediction(
                base, douter
            ).astype(np.float32)
            outer_cases = np.flatnonzero(self.case_fold == outer_fold)
            context = self.full_base_context(
                base, outer_cases, keep_pair_cases=set()
            )
            for case in outer_cases.astype(int):
                full = self.load_full(case, static_only=True)
                mapping = self.full_to_source_scene_mapping(
                    case, full["scene_primary"]
                )
                s0, s1 = self.scene_offsets[case : case + 2]
                scene_base_oof[s0:s1] = context[case]["scene_prediction"][mapping]
            del dtrain, douter, base, train_index, outer_index, context
            gc.collect()
            print(f"DEPLOYMENT_BASE_OOF_FOLD_DONE outer={outer_fold}", flush=True)
        if not np.isfinite(base_oof).all() or not np.isfinite(scene_base_oof).all():
            raise RuntimeError("deployment base OOF context is incomplete")

        correction_features = self.global_context_features(
            base_oof, scene_base_oof
        )
        correction_target = (
            np.asarray(self.label, dtype=np.float64) - base_oof
        ) / self.target_scale
        dcorrection = xgb.DMatrix(
            correction_features,
            label=correction_target.astype(np.float32),
            feature_names=CONDITIONAL_FEATURES,
        )
        correction = xgb.train(
            correction_params,
            dcorrection,
            num_boost_round=correction_rounds,
            verbose_eval=False,
        )
        del correction_features, correction_target, dcorrection, base_oof
        gc.collect()

        dbase = self.base_matrix(all_index)
        base = xgb.train(
            base_params, dbase, num_boost_round=base_rounds,
            verbose_eval=False,
        )
        base_prediction = self.physical_base_prediction(base, dbase)
        all_cases = np.arange(N_CASES)
        context = self.full_base_context(
            base, all_cases, keep_pair_cases=set(all_cases.astype(int))
        )
        scene_base = np.empty(self.n_scenes, dtype=np.float32)
        for case in range(N_CASES):
            full = self.load_full(case, static_only=True)
            mapping = self.full_to_source_scene_mapping(
                case, full["scene_primary"]
            )
            s0, s1 = self.scene_offsets[case : case + 2]
            scene_base[s0:s1] = context[case]["scene_prediction"][mapping]
        active_features = self.global_context_features(base_prediction, scene_base)
        active_correction = correction.inplace_predict(
            active_features
        ).astype(np.float64) * self.target_scale
        combined = base_prediction + active_correction
        by_case = self.corrected_full_coordinate(correction, context, all_cases)
        full_coordinate = np.empty(self.n_scenes, dtype=np.float32)
        self.put_scene_coordinate(full_coordinate, by_case)
        in_sample = {
            "pair": self.pair_metrics(combined),
            "fullneighbour_scene": self.full_coordinate_metrics(
                combined, full_coordinate
            ),
            "fixed_v22_active_scene": self.scene_metrics(combined),
        }
        base_path = output_dir / "fixed_base_all200.json"
        correction_path = output_dir / "fixed_correction_all200.json"
        atomic_model(base, base_path)
        atomic_model(correction, correction_path)
        del (
            dbase,
            base,
            correction,
            base_prediction,
            active_features,
            active_correction,
            combined,
            context,
            by_case,
            full_coordinate,
            all_index,
        )
        gc.collect()
        return {
            "base_rounds": base_rounds,
            "correction_rounds": correction_rounds,
            "base_model": str(base_path),
            "base_sha256": sha256(base_path),
            "correction_model": str(correction_path),
            "correction_sha256": sha256(correction_path),
            "correction_target": (
                "labelled sheared-pair residual to five-fold base-OOF prediction"
            ),
            "correction_scene_context": (
                "sum of base predictions over all deployed neighbours"
            ),
            "in_sample_deployment_diagnostic_not_used_for_selection": in_sample,
        }


def load_fixed_params(summary_path: Path) -> dict[str, Any]:
    with summary_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    params = payload["best_params"]
    expected = {"base", "correction", "base_rounds", "correction_rounds"}
    if set(params) != expected:
        raise RuntimeError("parameter summary does not contain the expected stack")
    if len(params["base"]) != 8 or len(params["correction"]) != 8:
        raise RuntimeError("fixed parameter stages are incomplete")
    return params


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-cache", required=True)
    parser.add_argument("--scene-cache", required=True)
    parser.add_argument("--full-neighbour-cache", required=True)
    parser.add_argument("--parameter-summary", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    if summary_path.exists():
        print(f"Fixed model already complete: {summary_path}", flush=True)
        return

    parameter_summary = Path(args.parameter_summary).resolve()
    stage_params = load_fixed_params(parameter_summary)
    experiment = FullNeighbourExperiment(
        Path(args.source_cache).resolve(),
        Path(args.scene_cache).resolve(),
        Path(args.full_neighbour_cache).resolve(),
    )
    metrics, oof_pair, oof_coordinate = experiment.fit_crossfit(stage_params)
    pair_path = output_dir / "oof_active_pair_prediction.npy"
    coordinate_path = output_dir / "oof_fullneighbour_scene_coordinate.npy"
    atomic_npy(pair_path, oof_pair)
    atomic_npy(coordinate_path, oof_coordinate)
    del oof_pair, oof_coordinate
    gc.collect()
    deployment = experiment.fit_deployment(stage_params, output_dir)
    payload = {
        "schema_version": 1,
        "kind": "fixed sequential R_blend model with all-neighbour scene context",
        "training_change": (
            "all neighbours define scene context/bin coordinate; only sheared "
            "half-neighbour rows carry labels and enter supervised pair losses"
        ),
        "tuning_performed": False,
        "fixed_parameter_source": str(parameter_summary),
        "fixed_parameter_source_sha256": sha256(parameter_summary),
        "fixed_params": stage_params,
        "crossfit": {
            "n_cases": N_CASES,
            "n_folds": N_FOLDS,
            "case_fold": experiment.case_fold,
            "per_rotation": {
                "base_fit_cases": 80,
                "correction_fit_cases": 80,
                "outer_scored_cases": 40,
                "roles_are_case_disjoint": True,
            },
        },
        "features": {
            "base": BASE_FEATURES,
            "correction": CONDITIONAL_FEATURES,
            "scene_sum_population": "all deployed neighbours",
            "supervised_population": "sheared labelled neighbours only",
        },
        "source": {
            "label_cache": str(experiment.source_cache),
            "label_cache_metadata_sha256": sha256(
                experiment.source_cache / "metadata.json"
            ),
            "full_neighbour_cache": str(experiment.full_cache),
            "full_neighbour_cache_metadata_sha256": sha256(
                experiment.full_cache / "metadata.json"
            ),
            "mean_labelled_neighbours": float(experiment.scene_counts.mean()),
            "mean_all_neighbours": float(experiment.full_count.mean()),
            "constgold_opened": False,
            "anchor_truth_opened": False,
        },
        "oof_metrics": metrics,
        "oof_active_pair_prediction": str(pair_path),
        "oof_fullneighbour_scene_coordinate": str(coordinate_path),
        "deployment_fit_all_200_cases": deployment,
    }
    atomic_json(summary_path, payload)
    print(json.dumps(json_clean(payload), indent=2, sort_keys=True), flush=True)
    print("CROSSFIT_SEQUENTIAL_FULLNEIGHBOUR_FIXED_DONE", flush=True)


if __name__ == "__main__":
    main()
