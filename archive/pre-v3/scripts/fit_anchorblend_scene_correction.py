"""Fit and validate one scene-level correction to V2.2 on anchor data only.

The candidate predicts ``truth_local10 - V2.2`` from intrinsic primary and
deployed-pair scene summaries.  Five case-blocked development models are
ensembled; cases 250--299 and their exact-key coherent response are untouched
validation sets.  No constgold quantity is read and no deployable model is
written unless all anchor-only gates pass (this script only writes diagnostics).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

BE = "/home/z/Zekang.Zhang/blendemu"
if BE not in sys.path:
    sys.path.insert(0, BE)

from blendemu import data_utils, nz_utils  # noqa: E402
from blendemu.inference import BlendingPredictor  # noqa: E402


COND = dict(pixel_size=0.2, zero_point=30.0, psf_fwhm=0.73,
            moffat_beta=2.224, pixel_rms=0.312)
PAIR_FEATURES = [
    "Re_input_p", "Re_input_s", "r_input_p", "r_input_s",
    "sersic_n_input_p", "sersic_n_input_s", "distance",
]
KEY = ["case", "input_index"]
TILE = "tile180.0_-0.5"
SHELLS = [(0.0, 1.0, "d0_1"), (1.0, 3.0, "d1_3"), (3.0, 10.0, "d3_10")]
MODEL_FEATURES = [
    "r_primary", "log_Re_primary", "log_n_primary", "R_v22",
    "log1p_n_pairs", "min_distance", "log1p_max_flux_ratio",
    *[f"log1p_n_{name}" for _, _, name in SHELLS],
    *[f"R_v22_{name}" for _, _, name in SHELLS],
    *[f"log1p_flux_ratio_{name}" for _, _, name in SHELLS],
]


def sem(values) -> float:
    values = np.asarray(values, float)
    return float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else float("nan")


def stat(values) -> dict:
    values = np.asarray(values, float)
    return {"mean": float(values.mean()), "case_sem": sem(values)}


def input_frame(base: str, case: int, sign: float) -> pd.DataFrame:
    path = os.path.join(
        base, f"case{case}_{str(float(sign))}", "real0", "catalogues", "input",
        f"gals_info_{TILE}.feather",
    )
    frame = pd.read_feather(path)
    return frame.rename(columns={column: column.replace("_input", "") for column in frame})


def exact_common(response10: list[str], response15: list[str],
                 case_min: int, case_max: int) -> pd.DataFrame:
    parts = []
    value = [
        "R_blend_truth", "R_blend_null", "R_blend_lsst_r_extnbr_v22",
    ]
    for layer, (path10, path15) in enumerate(zip(response10, response15)):
        r10 = pd.read_feather(path10, columns=KEY + value)
        r15 = pd.read_feather(path15, columns=KEY)
        joined = r10.merge(r15, on=KEY, validate="one_to_one")
        joined = joined[joined.case.between(case_min, case_max)].copy()
        joined["layer"] = layer
        parts.append(joined)
    out = pd.concat(parts, ignore_index=True)
    if out.duplicated(KEY).any():
        raise RuntimeError("response keys overlap across layers")
    return out


def aggregate_scene_features(base: str, observed: pd.DataFrame,
                             predictor: BlendingPredictor, sign: float) -> pd.DataFrame:
    cuts, r_max, k = predictor._select("regression")
    parts = []
    replay_max = 0.0
    for case, case_observed in observed.groupby("case", sort=True):
        case = int(case)
        ids = case_observed.input_index.to_numpy(np.int64)
        frame = input_frame(base, case, sign)
        primary = frame[frame["index"].isin(ids)].copy()
        if len(primary) != len(ids):
            raise RuntimeError(f"case {case}: missing anchor input")
        raw = nz_utils.make_reg_features(
            primary, frame, r_max=float(r_max) / 3600.0, k=int(k),
        )
        primary_id = next(
            column for column in raw if column.startswith("index") and column.endswith("_p")
        )
        pairs = data_utils.source_select_reg(raw, cuts=cuts).copy()
        pairs["distance"] *= 3600.0
        pairs["prediction"] = predictor.predict_on_pairs(
            pairs[PAIR_FEATURES], task="response", warn_extrapolation=False,
        ).response.to_numpy(float)
        pairs["flux_ratio"] = np.power(
            10.0, -0.4 * (pairs.r_input_s.to_numpy(float) - pairs.r_input_p.to_numpy(float)),
        )
        anchors = pd.Index(ids, name="input_index")
        group = pairs.groupby(primary_id, sort=False)
        scene = pd.DataFrame(index=anchors)
        scene["n_pairs"] = group.size().reindex(anchors, fill_value=0).to_numpy(int)
        scene["R_v22"] = group.prediction.sum().reindex(anchors, fill_value=0.0).to_numpy(float)
        scene["min_distance"] = group.distance.min().reindex(anchors, fill_value=10.0).to_numpy(float)
        scene["max_flux_ratio"] = group.flux_ratio.max().reindex(
            anchors, fill_value=0.0,
        ).to_numpy(float)
        distance = pairs.distance.to_numpy(float)
        for lo, hi, name in SHELLS:
            shell = pairs[(distance >= lo) & (distance < hi)]
            shell_group = shell.groupby(primary_id, sort=False)
            scene[f"n_{name}"] = shell_group.size().reindex(
                anchors, fill_value=0,
            ).to_numpy(int)
            scene[f"R_v22_{name}"] = shell_group.prediction.sum().reindex(
                anchors, fill_value=0.0,
            ).to_numpy(float)
            scene[f"flux_ratio_{name}"] = shell_group.flux_ratio.sum().reindex(
                anchors, fill_value=0.0,
            ).to_numpy(float)
        by_id = primary.set_index("index", verify_integrity=True)
        scene["r_primary"] = by_id.loc[anchors, "r"].to_numpy(float)
        scene["log_Re_primary"] = np.log(np.clip(
            by_id.loc[anchors, "Re"].to_numpy(float), 1e-4, None,
        ))
        scene["log_n_primary"] = np.log(np.clip(
            by_id.loc[anchors, "sersic_n"].to_numpy(float), 1e-4, None,
        ))
        scene["log1p_n_pairs"] = np.log1p(scene.n_pairs.to_numpy(float))
        scene["log1p_max_flux_ratio"] = np.log1p(scene.max_flux_ratio.to_numpy(float))
        for _, _, name in SHELLS:
            scene[f"log1p_n_{name}"] = np.log1p(scene[f"n_{name}"].to_numpy(float))
            scene[f"log1p_flux_ratio_{name}"] = np.log1p(
                scene[f"flux_ratio_{name}"].to_numpy(float)
            )
        stored = case_observed.set_index("input_index").loc[
            anchors, "R_blend_lsst_r_extnbr_v22"
        ].to_numpy(float)
        error = float(np.max(np.abs(stored - scene.R_v22.to_numpy(float))))
        replay_max = max(replay_max, error)
        if not np.allclose(stored, scene.R_v22, rtol=1e-7, atol=5e-7):
            raise RuntimeError(f"case {case}: V2.2 replay failed, max abs={error:.3e}")
        scene = scene.reset_index()
        scene["case"] = case
        parts.append(scene[KEY + MODEL_FEATURES])
        print(
            f"case {case}: anchors={len(scene):,} pairs={len(pairs):,} "
            f"replay={error:.2e}", flush=True,
        )
    features = pd.concat(parts, ignore_index=True)
    if not np.isfinite(features[MODEL_FEATURES].to_numpy(float)).all():
        raise RuntimeError("non-finite scene feature")
    print(f"scene replay max abs={replay_max:.3e}", flush=True)
    return observed.merge(features, on=KEY, validate="one_to_one")


def fit_case_blocked_ensemble(frame: pd.DataFrame, development_max: int,
                              seed: int,
                              model_features=None,
                              ) -> tuple[np.ndarray, np.ndarray, float]:
    if model_features is None:
        model_features = MODEL_FEATURES
    development = frame.case.to_numpy(int) <= development_max
    validation = ~development
    x = frame[model_features].to_numpy(np.float32)
    target = (
        frame.R_blend_truth.to_numpy(float)
        - frame.R_blend_lsst_r_extnbr_v22.to_numpy(float)
    )
    cases = frame.case.to_numpy(int)
    dev_cases = np.unique(cases[development])
    fold_by_case = {case: index % 5 for index, case in enumerate(dev_cases)}
    fold = np.asarray([fold_by_case.get(case, -1) for case in cases], int)
    oof = np.full(len(frame), np.nan)
    validation_predictions = []
    for heldout_fold in range(5):
        train = development & (fold != heldout_fold)
        heldout = development & (fold == heldout_fold)
        count_by_case = pd.Series(cases[train]).value_counts()
        weights = np.asarray([1.0 / count_by_case[case] for case in cases[train]], float)
        weights *= len(weights) / weights.sum()
        model = HistGradientBoostingRegressor(
            loss="squared_error", learning_rate=0.05, max_iter=200,
            max_leaf_nodes=15, min_samples_leaf=1000, l2_regularization=10.0,
            random_state=seed + heldout_fold,
        ).fit(x[train], target[train], sample_weight=weights)
        oof[heldout] = model.predict(x[heldout])
        validation_predictions.append(model.predict(x[validation]))
    if not np.isfinite(oof[development]).all():
        raise RuntimeError("incomplete out-of-fold development predictions")
    validation_models = np.vstack(validation_predictions)
    prediction = np.full(len(frame), np.nan)
    prediction[development] = oof[development]
    prediction[validation] = validation_models.mean(axis=0)
    stability = float(np.std(validation_models.mean(axis=1), ddof=1))
    return prediction, target, stability


def response_summary(frame: pd.DataFrame, correction: np.ndarray,
                     prediction_edges: np.ndarray) -> dict:
    gap = (
        frame.R_blend_lsst_r_extnbr_v22.to_numpy(float)
        - frame.R_blend_truth.to_numpy(float)
    )
    work = pd.DataFrame({
        "case": frame.case.to_numpy(int), "raw": gap,
        "corrected": gap + correction,
        "constant": gap + float(frame.attrs["constant_correction"]),
        "prediction": frame.R_blend_lsst_r_extnbr_v22.to_numpy(float),
    })
    case = work.groupby("case", sort=True)[["raw", "corrected", "constant"]].mean()
    conditional = {}
    bins = np.searchsorted(prediction_edges, work.prediction.to_numpy(float), side="right") - 1
    bins = np.clip(bins, 0, len(prediction_edges) - 2)
    work["bin"] = bins
    for index in range(len(prediction_edges) - 1):
        subset = work[work.bin == index]
        by_case = subset.groupby("case", sort=True)[["raw", "corrected", "constant"]].mean()
        conditional[str(index)] = {
            "edges": [
                None if not np.isfinite(value) else float(value)
                for value in (prediction_edges[index], prediction_edges[index + 1])
            ],
            "n_rows": int(len(subset)),
            "raw": stat(by_case.raw), "corrected": stat(by_case.corrected),
            "constant": stat(by_case.constant),
        }
    return {
        "n_rows": int(len(frame)), "n_cases": int(frame.case.nunique()),
        "raw": stat(case.raw), "corrected": stat(case.corrected),
        "constant": stat(case.constant), "conditional_prediction_quintiles": conditional,
        "max_abs_conditional_raw": float(max(abs(v["raw"]["mean"]) for v in conditional.values())),
        "max_abs_conditional_corrected": float(max(
            abs(v["corrected"]["mean"]) for v in conditional.values()
        )),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", required=True)
    ap.add_argument("--response10", action="append", required=True)
    ap.add_argument("--response15", action="append", required=True)
    ap.add_argument("--coherent-response", required=True)
    ap.add_argument("--case-min", type=int, default=200)
    ap.add_argument("--case-max", type=int, default=299)
    ap.add_argument("--development-max", type=int, default=249)
    ap.add_argument("--sign", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=311)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    if os.path.exists(args.output):
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if len(args.response10) != len(args.response15):
        raise RuntimeError("response10/15 argument counts differ")

    predictor = BlendingPredictor.load(
        os.path.join(BE, "models"), tag="lsst_r_extnbr_v22",
        conditions=COND, device="cpu",
    )
    observed = exact_common(args.response10, args.response15, args.case_min, args.case_max)
    frame = aggregate_scene_features(args.base, observed, predictor, args.sign)
    correction, target, stability = fit_case_blocked_ensemble(
        frame, args.development_max, args.seed,
    )
    development = frame.case.to_numpy(int) <= args.development_max
    validation = ~development
    constant_correction = float(target[development].mean())
    frame.attrs["constant_correction"] = constant_correction
    edges = np.quantile(
        frame.loc[development, "R_blend_lsst_r_extnbr_v22"].to_numpy(float),
        np.linspace(0.0, 1.0, 6),
    )
    edges[0] = -np.inf; edges[-1] = np.inf
    random_dev_frame = frame.loc[development].copy()
    random_dev_frame.attrs["constant_correction"] = constant_correction
    random_dev = response_summary(random_dev_frame, correction[development], edges)
    random_val_frame = frame.loc[validation].copy()
    random_val_frame.attrs["constant_correction"] = constant_correction
    random_val = response_summary(random_val_frame, correction[validation], edges)

    coherent = pd.read_feather(
        args.coherent_response,
        columns=KEY + ["R_blend_truth", "R_blend_lsst_r_extnbr_v22"],
    )
    coherent = coherent[coherent.case.between(args.case_min, args.case_max)]
    coherent_val = frame.loc[validation, KEY + MODEL_FEATURES].merge(
        coherent, on=KEY, validate="one_to_one",
    )
    correction_by_key = pd.Series(
        correction[validation],
        index=pd.MultiIndex.from_frame(frame.loc[validation, KEY]),
    )
    coherent_correction = correction_by_key.reindex(
        pd.MultiIndex.from_frame(coherent_val[KEY])
    ).to_numpy(float)
    coherent_val.attrs["constant_correction"] = constant_correction
    coherent_summary = response_summary(coherent_val, coherent_correction, edges)

    random_uncertainty = float(np.hypot(
        random_val["corrected"]["case_sem"], stability,
    ))
    coherent_uncertainty = float(np.hypot(
        coherent_summary["corrected"]["case_sem"], stability,
    ))
    gates = {
        "random_validation_global_smaller": bool(
            abs(random_val["corrected"]["mean"]) < abs(random_val["raw"]["mean"])
        ),
        "random_validation_global_zero_at_2_combined_sem": bool(
            abs(random_val["corrected"]["mean"]) <= 2.0 * random_uncertainty
        ),
        "random_validation_max_conditional_smaller": bool(
            random_val["max_abs_conditional_corrected"] < random_val["max_abs_conditional_raw"]
        ),
        "coherent_validation_global_smaller": bool(
            abs(coherent_summary["corrected"]["mean"])
            < abs(coherent_summary["raw"]["mean"])
        ),
        "coherent_validation_global_zero_at_2_combined_sem": bool(
            abs(coherent_summary["corrected"]["mean"]) <= 2.0 * coherent_uncertainty
        ),
        "coherent_validation_max_conditional_smaller": bool(
            coherent_summary["max_abs_conditional_corrected"]
            < coherent_summary["max_abs_conditional_raw"]
        ),
        "training_stability_below_0p003": bool(stability < 0.003),
    }
    payload = {
        "candidate": "case-blocked HGB scene correction to V2.2 response",
        "features": MODEL_FEATURES,
        "development_cases": [args.case_min, args.development_max],
        "validation_cases": [args.development_max + 1, args.case_max],
        "n_layers": len(args.response10), "n_exact_common_rows": int(len(frame)),
        "constant_development_correction": constant_correction,
        "ensemble_validation_mean_stability_sd": stability,
        "prediction_quintile_edges": [
            None if not np.isfinite(value) else float(value) for value in edges
        ],
        "random_local10_development": random_dev,
        "random_local10_validation": random_val,
        "coherent_validation": coherent_summary,
        "random_validation_corrected_uncertainty_including_training": random_uncertainty,
        "coherent_validation_corrected_uncertainty_including_training": coherent_uncertainty,
        "gates": gates, "gate_passed": bool(all(gates.values())),
        "deployable_model_written": False, "constgold_opened": False,
    }
    with open(args.output, "x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    print("ANCHORBLEND_SCENE_CORRECTION_DONE", flush=True)


if __name__ == "__main__":
    main()
