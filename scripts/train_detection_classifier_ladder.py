#!/usr/bin/env python3
"""Train and validate an exact-neighbour detection-classifier ladder.

The caller supplies BlendEMU-produced truth and CrossMatch catalogues.  SBSI
constructs exact in-aperture neighbours, trains the classifiers, and evaluates
both ordinary held-case classification metrics and the centered forward
detection-selection response.  ConstGold is intentionally not consumed here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from sbsi.coordinates import ellipticity_from_axis_ratio_angle
from sbsi.detection_classifier import (
    AXIS_RATIO_FEATURES,
    BASE_DETECTION_FEATURES,
    PAIR_FRAME_SHAPE_FEATURES,
    build_detection_feature_frame,
    choose_neighbour_ladder,
    detection_selection_response,
    render_catalogue_shape_shear,
)


CONDITIONS = {
    "pixel_size": 0.2,
    "zero_point": 30.0,
    "psf_fwhm": 0.73,
    "moffat_beta": 2.224,
    "pixel_rms": 0.312,
}
CURRENT_XGB_PARAMS = {
    "objective": "binary:logistic",
    "booster": "gbtree",
    "learning_rate": 0.16154830815056245,
    "max_depth": 9,
    "min_child_weight": 161,
    "subsample": 0.959173065337887,
    "colsample_bytree": 0.9556674659210342,
    "gamma": 0.017059395190662907,
    "reg_alpha": 0.003239447934442709,
    "reg_lambda": 0.9463720390498181,
    "eval_metric": "logloss",
    "tree_method": "hist",
}


def parse_cases(value: str) -> list[int]:
    out: list[int] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            lower, upper = (int(part) for part in item.split("-", 1))
            if upper < lower:
                raise argparse.ArgumentTypeError(f"invalid case range {item!r}")
            out.extend(range(lower, upper + 1))
        else:
            out.append(int(item))
    if not out or len(set(out)) != len(out):
        raise argparse.ArgumentTypeError("case list must be nonempty and unique")
    return out


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def catalogue_paths(root: Path, case: int, shear: str) -> tuple[Path, Path]:
    truth = root / f"gals{case}_{shear}.feather"
    matched = (
        root
        / f"case{case}_{shear}"
        / "real0/catalogues/CrossMatch/tile180.0_-0.5_rot0_matched.feather"
    )
    for path in (truth, matched):
        if not path.is_file():
            raise FileNotFoundError(path)
    return truth, matched


def detection_labels(galaxies: pd.DataFrame, matched_path: Path) -> tuple[np.ndarray, dict]:
    if "index" not in galaxies:
        raise KeyError("truth catalogue lacks index")
    input_id = galaxies["index"].to_numpy(dtype=np.int64)
    if len(np.unique(input_id)) != len(input_id):
        raise ValueError("truth input identifiers are not unique")
    matched = pd.read_feather(matched_path, columns=["id_input"])["id_input"].to_numpy(
        dtype=np.int64
    )
    known = np.isin(matched, input_id)
    n_unmatched = int((~known).sum())
    if n_unmatched:
        raise ValueError(
            f"{matched_path} contains {n_unmatched} rows absent from the truth catalogue"
        )
    detected_id = np.unique(matched)
    label = np.isin(input_id, detected_id)
    return label, {
        "crossmatch_rows": int(len(matched)),
        "unique_detected_inputs": int(len(detected_id)),
        "duplicate_crossmatch_input_rows": int(len(matched) - len(detected_id)),
        "unmatched_crossmatch_rows": n_unmatched,
    }


def aligned_shapes(
    g0: pd.DataFrame, gg: pd.DataFrame, primary_rows: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    g1 = gg.iloc[primary_rows]["g1"].to_numpy(dtype=float)
    g2 = gg.iloc[primary_rows]["g2"].to_numpy(dtype=float)
    magnitude = np.hypot(g1, g2)
    unit1 = np.divide(g1, magnitude, out=np.zeros_like(g1), where=magnitude > 0)
    unit2 = np.divide(g2, magnitude, out=np.zeros_like(g2), where=magnitude > 0)
    e10, e20 = ellipticity_from_axis_ratio_angle(
        g0.iloc[primary_rows]["axis_ratio"].to_numpy(dtype=float),
        g0.iloc[primary_rows]["position_angle"].to_numpy(dtype=float),
    )
    e1g, e2g = ellipticity_from_axis_ratio_angle(
        gg.iloc[primary_rows]["axis_ratio"].to_numpy(dtype=float),
        gg.iloc[primary_rows]["position_angle"].to_numpy(dtype=float),
    )
    return e10 * unit1 + e20 * unit2, e1g * unit1 + e2g * unit2, magnitude


def classifier_metrics(y: np.ndarray, probability: np.ndarray) -> dict:
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        balanced_accuracy_score,
        brier_score_loss,
        log_loss,
        roc_auc_score,
    )

    y = np.asarray(y, dtype=bool)
    probability = np.asarray(probability, dtype=float)
    hard = probability >= 0.5
    return {
        "logloss": float(log_loss(y, probability)),
        "brier": float(brier_score_loss(y, probability)),
        "roc_auc": float(roc_auc_score(y, probability)),
        "average_precision": float(average_precision_score(y, probability)),
        "accuracy": float(accuracy_score(y, hard)),
        "balanced_accuracy": float(balanced_accuracy_score(y, hard)),
        "positive_fraction": float(y.mean()),
        "mean_probability": float(probability.mean()),
    }


def response_summary(case_meta: list[dict], p0: np.ndarray, pg: np.ndarray) -> dict:
    truth_case, model_case, error_case = [], [], []
    offset = 0
    for meta in case_meta:
        n = len(meta["y0"])
        case_p0, case_pg = p0[offset : offset + n], pg[offset : offset + n]
        offset += n
        truth = detection_selection_response(
            meta["e0_parallel"],
            meta["eg_parallel"],
            meta["y0"].astype(float),
            meta["yg"].astype(float),
            meta["shear"],
        )
        model = detection_selection_response(
            meta["e0_parallel"],
            meta["eg_parallel"],
            case_p0,
            case_pg,
            meta["shear"],
        )
        truth_case.append(truth)
        model_case.append(model)
        error_case.append(model - truth)
    if offset != len(p0) or len(p0) != len(pg):
        raise ValueError("validation predictions are not case-aligned")

    def mean_sem(values):
        values = np.asarray(values, dtype=float)
        sem = float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else np.nan
        return float(values.mean()), sem

    truth_mean, truth_sem = mean_sem(truth_case)
    model_mean, model_sem = mean_sem(model_case)
    error_mean, error_sem = mean_sem(error_case)
    return {
        "estimator": "centered_forward_case_mean",
        "n_cases": len(case_meta),
        "truth": truth_mean,
        "truth_case_sem": truth_sem,
        "model": model_mean,
        "model_case_sem": model_sem,
        "model_minus_truth": error_mean,
        "paired_case_sem": error_sem,
        "truth_per_case": truth_case,
        "model_per_case": model_case,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulation-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-cases", type=parse_cases, default=parse_cases("0-29"))
    parser.add_argument("--validation-cases", type=parse_cases, default=parse_cases("30-39"))
    parser.add_argument("--zero-shear-label", default="0.0")
    parser.add_argument("--forward-shear-label", default="0.05")
    parser.add_argument("--radius-arcsec", type=float, default=3.0)
    parser.add_argument("--impact-exponents", type=float, nargs="+", default=[1.0, 2.0, 4.0])
    parser.add_argument("--max-primaries-per-case", type=int, default=50_000)
    parser.add_argument("--sampling-seed", type=int, default=20260824)
    parser.add_argument("--xgb-seed", type=int, default=321)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--nthread", type=int, default=8)
    parser.add_argument("--num-boost-round", type=int, default=800)
    parser.add_argument("--early-stopping-rounds", type=int, default=30)
    parser.add_argument("--deployed-model", type=Path)
    args = parser.parse_args()

    overlap = sorted(set(args.train_cases) & set(args.validation_cases))
    if overlap:
        raise ValueError(f"train/validation cases overlap: {overlap}")
    if args.max_primaries_per_case <= 0:
        raise ValueError("max-primaries-per-case must be positive")
    output = args.output_dir
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    base = list(BASE_DETECTION_FEATURES)
    q_features = base + list(AXIS_RATIO_FEATURES)
    pair_shape_features = q_features + list(PAIR_FRAME_SHAPE_FEATURES)
    variants = {
        "nearest7": {"choice": "nearest", "features": base},
        "nearest_q": {"choice": "nearest", "features": q_features},
        "nearest_q_pairshape": {
            "choice": "nearest",
            "features": pair_shape_features,
        },
    }
    for exponent in args.impact_exponents:
        key = f"impact_a{exponent:g}"
        variants[f"{key}_q"] = {"choice": key, "features": q_features}
        variants[f"{key}_q_pairshape"] = {
            "choice": key,
            "features": pair_shape_features,
        }
    if 2.0 in args.impact_exponents:
        variants["impact_a2_q_score"] = {
            "choice": "impact_a2",
            "features": q_features + ["neighbour_log_impact"],
        }

    blocks = {
        name: {split: [] for split in ("train0", "traing", "val0", "valg")}
        for name in variants
    }
    labels = {split: [] for split in ("train0", "traing", "val0", "valg")}
    case_reports, validation_meta = [], []
    all_cases = [(case, "train") for case in args.train_cases] + [
        (case, "val") for case in args.validation_cases
    ]
    started = time.time()
    for sequence, (case, split) in enumerate(all_cases):
        truth0_path, matched0_path = catalogue_paths(
            args.simulation_root, case, args.zero_shear_label
        )
        truthg_path, matchedg_path = catalogue_paths(
            args.simulation_root, case, args.forward_shear_label
        )
        g0 = pd.read_feather(truth0_path)
        gg = pd.read_feather(truthg_path)
        if len(g0) != len(gg):
            raise ValueError(f"case {case}: truth legs have different row counts")
        for column in ("index", "RA", "DEC", "r", "Re", "sersic_n"):
            if not np.array_equal(g0[column].to_numpy(), gg[column].to_numpy()):
                raise ValueError(f"case {case}: truth legs disagree in invariant {column}")
        y0, label_report0 = detection_labels(g0, matched0_path)
        yg, label_reportg = detection_labels(gg, matchedg_path)

        eligible = np.flatnonzero(
            (gg["r"].to_numpy(dtype=float) > 18.0)
            & (gg["r"].to_numpy(dtype=float) < 28.0)
            & (gg["Re"].to_numpy(dtype=float) > 0.1)
            & (gg["Re"].to_numpy(dtype=float) < 1.5)
        )
        rng = np.random.default_rng(args.sampling_seed + case)
        if len(eligible) > args.max_primaries_per_case:
            primary_rows = np.sort(
                rng.choice(eligible, size=args.max_primaries_per_case, replace=False)
            )
        else:
            primary_rows = eligible
        gg_rendered = render_catalogue_shape_shear(gg)
        choices = choose_neighbour_ladder(
            gg_rendered,
            primary_rows,
            radius_arcsec=args.radius_arcsec,
            impact_exponents=args.impact_exponents,
        )
        frames0, framesg = {}, {}
        for choice_name, choice in choices.items():
            frames0[choice_name] = build_detection_feature_frame(
                g0, choice, conditions=CONDITIONS
            )
            framesg[choice_name] = build_detection_feature_frame(
                gg_rendered, choice, conditions=CONDITIONS
            )
            if not np.array_equal(
                frames0[choice_name]["secondary_row"].to_numpy(),
                framesg[choice_name]["secondary_row"].to_numpy(),
            ):
                raise ValueError(f"case {case}: neighbour identity changed between legs")

        output_split0 = "train0" if split == "train" else "val0"
        output_splitg = "traing" if split == "train" else "valg"
        for name, spec in variants.items():
            feature_names = spec["features"]
            blocks[name][output_split0].append(
                frames0[spec["choice"]][feature_names].to_numpy(dtype=np.float32)
            )
            blocks[name][output_splitg].append(
                framesg[spec["choice"]][feature_names].to_numpy(dtype=np.float32)
            )
        labels[output_split0].append(y0[primary_rows])
        labels[output_splitg].append(yg[primary_rows])

        e0_parallel, eg_parallel, shear = aligned_shapes(
            g0, gg_rendered, primary_rows
        )
        if split == "val":
            validation_meta.append(
                {
                    "case": case,
                    "e0_parallel": e0_parallel,
                    "eg_parallel": eg_parallel,
                    "shear": shear,
                    "y0": y0[primary_rows],
                    "yg": yg[primary_rows],
                }
            )
        nearest = choices["nearest"]
        case_reports.append(
            {
                "case": case,
                "split": split,
                "truth_rows": len(gg),
                "eligible_rows": int(len(eligible)),
                "sampled_rows": int(len(primary_rows)),
                "nonzero_shear_fraction": float((shear > 0).mean()),
                "candidate_count_mean": float(nearest.candidate_count.mean()),
                "candidate_count_max": int(nearest.candidate_count.max()),
                "candidate_count_gt4_fraction": float(
                    (nearest.candidate_count > 4).mean()
                ),
                "labels_g0": label_report0,
                "labels_g": label_reportg,
                "detection_fraction_g0": float(y0[primary_rows].mean()),
                "detection_fraction_g": float(yg[primary_rows].mean()),
                "impact_differs_from_nearest": {
                    f"a{exponent:g}": float(
                        np.mean(
                            choices[f"impact_a{exponent:g}"].secondary_row
                            != nearest.secondary_row
                        )
                    )
                    for exponent in args.impact_exponents
                },
            }
        )
        print(
            f"case {case:03d} {split}: sample={len(primary_rows):,} "
            f"<Ncand>={nearest.candidate_count.mean():.2f} "
            f"P(Ncand>4)={(nearest.candidate_count > 4).mean():.3f} "
            f"elapsed={time.time() - started:.1f}s",
            flush=True,
        )
        del g0, gg, gg_rendered, frames0, framesg, choices

    combined_labels = {
        key: np.concatenate(value) for key, value in labels.items()
    }
    y_train = np.concatenate((combined_labels["train0"], combined_labels["traing"]))
    y_val = np.concatenate((combined_labels["val0"], combined_labels["valg"]))

    import xgboost as xgb

    results = {}

    def evaluate(name, booster, feature_names, xval0, xvalg, iteration_range=None):
        d0 = xgb.DMatrix(xval0, feature_names=feature_names)
        dg = xgb.DMatrix(xvalg, feature_names=feature_names)
        predict_kwargs = {} if iteration_range is None else {"iteration_range": iteration_range}
        p0 = booster.predict(d0, **predict_kwargs)
        pg = booster.predict(dg, **predict_kwargs)
        metrics = classifier_metrics(y_val, np.concatenate((p0, pg)))
        response = response_summary(validation_meta, p0, pg)
        results[name] = {
            "features": feature_names,
            "classification": metrics,
            "detection_selection_response": response,
        }

    nearest_val0 = np.concatenate(blocks["nearest7"]["val0"])
    nearest_valg = np.concatenate(blocks["nearest7"]["valg"])
    if args.deployed_model is not None:
        if not args.deployed_model.is_file():
            raise FileNotFoundError(args.deployed_model)
        deployed = xgb.Booster()
        deployed.load_model(args.deployed_model)
        evaluate(
            "deployed_frozen",
            deployed,
            base,
            nearest_val0,
            nearest_valg,
        )
        results["deployed_frozen"]["model_file"] = str(args.deployed_model)
        results["deployed_frozen"]["model_sha256"] = file_sha256(args.deployed_model)

    for name, spec in variants.items():
        feature_names = spec["features"]
        xtrain0 = np.concatenate(blocks[name]["train0"])
        xtraing = np.concatenate(blocks[name]["traing"])
        xval0 = np.concatenate(blocks[name]["val0"])
        xvalg = np.concatenate(blocks[name]["valg"])
        x_train = np.concatenate((xtrain0, xtraing))
        x_val = np.concatenate((xval0, xvalg))
        dtrain = xgb.DMatrix(x_train, label=y_train, feature_names=feature_names)
        dval = xgb.DMatrix(x_val, label=y_val, feature_names=feature_names)
        params = dict(CURRENT_XGB_PARAMS)
        params.update(
            {
                "device": args.device,
                "nthread": args.nthread,
                "seed": args.xgb_seed,
            }
        )
        evals_result = {}
        fit_start = time.time()
        booster = xgb.train(
            params,
            dtrain,
            num_boost_round=args.num_boost_round,
            evals=[(dtrain, "train"), (dval, "validation")],
            evals_result=evals_result,
            early_stopping_rounds=args.early_stopping_rounds,
            verbose_eval=25,
        )
        model_path = output / f"{name}.json"
        booster.save_model(model_path)
        evaluate(
            name,
            booster,
            feature_names,
            xval0,
            xvalg,
            iteration_range=(0, int(booster.best_iteration) + 1),
        )
        results[name].update(
            {
                "choice": spec["choice"],
                "best_iteration": int(booster.best_iteration),
                "best_score": float(booster.best_score),
                "fit_seconds": float(time.time() - fit_start),
                "model_file": str(model_path),
                "model_sha256": file_sha256(model_path),
                "feature_boundary": [
                    [
                        float(np.nanmin(x_train[:, index])),
                        float(np.nanmax(x_train[:, index])),
                    ]
                    for index in range(x_train.shape[1])
                ],
                "params": params,
                "train_logloss_last": float(evals_result["train"]["logloss"][-1]),
                "validation_logloss_last": float(
                    evals_result["validation"]["logloss"][-1]
                ),
            }
        )
        print(
            f"{name}: logloss={results[name]['classification']['logloss']:.6f} "
            f"Rdet(model-truth)="
            f"{results[name]['detection_selection_response']['model_minus_truth']:+.6f}",
            flush=True,
        )
        del xtrain0, xtraing, xval0, xvalg, x_train, x_val, dtrain, dval

    trained_names = list(variants)
    winner = min(
        trained_names,
        key=lambda name: (
            abs(results[name]["detection_selection_response"]["model_minus_truth"]),
            results[name]["classification"]["logloss"],
        ),
    )
    payload = {
        "format_version": 1,
        "created_unix": time.time(),
        "simulation_root": str(args.simulation_root),
        "train_cases": args.train_cases,
        "validation_cases": args.validation_cases,
        "constgold_firewall": "cases 40-139 were not read",
        "zero_shear_label": args.zero_shear_label,
        "forward_shear_label": args.forward_shear_label,
        "radius_arcsec": args.radius_arcsec,
        "impact_definition": "flux_s * (Re_s / distance_ps)**a",
        "rendered_shape_definition": (
            "catalogue intrinsic q/PA composed with each row's stored g1/g2; "
            "pairshape components projected into the primary-secondary separation frame"
        ),
        "impact_exponents": args.impact_exponents,
        "max_primaries_per_case": args.max_primaries_per_case,
        "sampling_seed": args.sampling_seed,
        "xgb_seed": args.xgb_seed,
        "selection_rule": (
            "minimum absolute held-case centered detection-response mismatch; "
            "held-case logloss breaks exact ties"
        ),
        "selected_winner": winner,
        "case_reports": case_reports,
        "results": results,
        "elapsed_seconds": float(time.time() - started),
    }
    report_path = output / "report.json"
    report_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"selected={winner}\nreport={report_path}", flush=True)


if __name__ == "__main__":
    main()
