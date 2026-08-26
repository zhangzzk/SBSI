#!/usr/bin/env python3
"""Evaluate frozen detection classifiers on unmatched ConstGold shear arms."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

from sbsi.coordinates import ellipticity_from_axis_ratio_angle
from sbsi.detection_classifier import (
    build_detection_feature_frame,
    choose_representative_neighbours,
    render_catalogue_shape_shear,
)
from sbsi.selection_model import load_selection_model
from scripts.evaluate_detection_classifier_shear_stability import (
    parse_model,
    predict_logits,
    probability,
)
from scripts.train_detection_classifier_transition_pair import (
    CONDITIONS,
    FEATURES,
    INVARIANT_FEATURES,
    catalogue_paths,
    changed_morphology,
    classifier_metrics,
    detection_labels,
    file_sha256,
    mean_sem,
    parse_cases,
    transition_metrics,
)


def antithetic_detection_response(
    intrinsic_e1: np.ndarray,
    weights_minus: np.ndarray,
    weights_plus: np.ndarray,
    shear_magnitude: float,
) -> float:
    """Pure detection-selection response from independently selected +/- arms."""

    shape = np.asarray(intrinsic_e1, dtype=float)
    weight_minus = np.asarray(weights_minus, dtype=float)
    weight_plus = np.asarray(weights_plus, dtype=float)
    if not (shape.shape == weight_minus.shape == weight_plus.shape):
        raise ValueError("shape and antithetic weights must have equal shape")
    if not np.isfinite(shear_magnitude) or shear_magnitude <= 0:
        raise ValueError("shear magnitude must be positive and finite")
    valid = (
        np.isfinite(shape)
        & np.isfinite(weight_minus)
        & np.isfinite(weight_plus)
        & (weight_minus >= 0)
        & (weight_plus >= 0)
    )
    if (
        not valid.any()
        or weight_minus[valid].sum() <= 0
        or weight_plus[valid].sum() <= 0
    ):
        raise ValueError("response population or selected weight is empty")

    def shift(weight):
        return np.sum(weight * shape[valid]) / np.sum(weight) - np.mean(shape[valid])

    return float((shift(weight_plus[valid]) - shift(weight_minus[valid])) / (2 * shear_magnitude))


def summarize_response(records: list[dict]) -> dict:
    truth = [record["truth"] for record in records]
    model = [record["model"] for record in records]
    residual = [prediction - target for prediction, target in zip(model, truth)]
    truth_mean, truth_sem = mean_sem(truth)
    model_mean, model_sem = mean_sem(model)
    residual_mean, residual_sem = mean_sem(residual)
    return {
        "truth": truth_mean,
        "truth_case_sem": truth_sem,
        "model": model_mean,
        "model_case_sem": model_sem,
        "model_minus_truth": residual_mean,
        "paired_case_sem": residual_sem,
        "truth_per_case": truth,
        "model_per_case": model,
    }


def truth_summary(values: list[float]) -> dict:
    mean, sem = mean_sem(values)
    return {"response": mean, "case_sem": sem, "response_per_case": values}


def rendered_decomposition_summary(records: list[dict]) -> dict:
    output = {}
    for key in ("parent", "detected", "detected_minus_parent"):
        values = [record[key] for record in records]
        output[key] = truth_summary(values)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulation-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", type=parse_cases, default=parse_cases("40-49"))
    parser.add_argument("--shear-magnitude", type=float, default=0.02)
    parser.add_argument("--model", action="append", type=parse_model, required=True)
    parser.add_argument("--radius-arcsec", type=float, default=3.0)
    parser.add_argument("--impact-exponent", type=float, default=1.0)
    parser.add_argument("--max-primaries-per-case", type=int, default=0)
    parser.add_argument("--sampling-seed", type=int, default=20260824)
    parser.add_argument("--prediction-batch-size", type=int, default=65_536)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.shear_magnitude <= 0 or not np.isfinite(args.shear_magnitude):
        raise ValueError("shear magnitude must be positive and finite")
    if len(args.model) != len({name for name, _ in args.model}):
        raise ValueError("model names must be unique")
    if args.max_primaries_per_case < 0:
        raise ValueError("max primaries must be non-negative; zero means all")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    device = torch.device(args.device)
    bundles = {}
    model_provenance = {}
    for name, path in args.model:
        bundle = load_selection_model(path, device=device)
        if bundle.preprocessor.feature_names != FEATURES:
            raise ValueError(f"{name}: checkpoint features differ from evaluator")
        bundles[name] = bundle
        model_provenance[name] = {
            "path": str(path.resolve()),
            "sha256": file_sha256(path),
            "metadata": bundle.metadata,
        }

    minus_label = f"-{args.shear_magnitude:g}"
    plus_label = f"{args.shear_magnitude:g}"
    combined_labels = {"minus": [], "plus": [], "eligible": []}
    combined_logits = {
        name: {"minus": [], "plus": [], "response": []} for name in bundles
    }
    truth_global_cases = []
    truth_domain_cases = []
    rendered_global_cases = []
    rendered_domain_cases = []
    case_reports = []
    started = time.time()

    for case in args.cases:
        truth_minus_path, matched_minus_path = catalogue_paths(
            args.simulation_root, case, minus_label
        )
        truth_plus_path, matched_plus_path = catalogue_paths(
            args.simulation_root, case, plus_label
        )
        minus = pd.read_feather(truth_minus_path)
        plus = pd.read_feather(truth_plus_path)
        if len(minus) != len(plus):
            raise ValueError(f"case {case}: antithetic truth legs differ in length")
        for column in (
            "index",
            "RA",
            "DEC",
            "r",
            "Re",
            "sersic_n",
            "axis_ratio",
            "position_angle",
        ):
            if not np.array_equal(minus[column].to_numpy(), plus[column].to_numpy()):
                raise ValueError(f"case {case}: invariant {column} differs between arms")
        if not np.allclose(
            minus["g1"].to_numpy(dtype=float),
            -args.shear_magnitude,
            rtol=0,
            atol=1.0e-8,
        ) or not np.allclose(
            plus["g1"].to_numpy(dtype=float),
            args.shear_magnitude,
            rtol=0,
            atol=1.0e-8,
        ):
            raise ValueError(f"case {case}: stored g1 is not the requested antithetic shear")

        label_minus, report_minus = detection_labels(minus, matched_minus_path)
        label_plus, report_plus = detection_labels(plus, matched_plus_path)
        e1_intrinsic, _ = ellipticity_from_axis_ratio_angle(
            minus["axis_ratio"].to_numpy(dtype=float),
            minus["position_angle"].to_numpy(dtype=float),
        )
        truth_global_cases.append(
            antithetic_detection_response(
                e1_intrinsic,
                label_minus.astype(float),
                label_plus.astype(float),
                args.shear_magnitude,
            )
        )

        eligible_rows = np.flatnonzero(
            (minus["r"].to_numpy(dtype=float) > 18.0)
            & (minus["r"].to_numpy(dtype=float) < 28.0)
            & (minus["Re"].to_numpy(dtype=float) > 0.1)
            & (minus["Re"].to_numpy(dtype=float) < 1.5)
        )
        if args.max_primaries_per_case and len(eligible_rows) > args.max_primaries_per_case:
            rng = np.random.default_rng(args.sampling_seed + case)
            rows = np.sort(
                rng.choice(
                    eligible_rows,
                    size=args.max_primaries_per_case,
                    replace=False,
                )
            )
        else:
            rows = eligible_rows

        rendered_minus = render_catalogue_shape_shear(minus)
        rendered_plus = render_catalogue_shape_shear(plus)
        rendered_e1_minus, _ = ellipticity_from_axis_ratio_angle(
            rendered_minus["axis_ratio"].to_numpy(dtype=float),
            rendered_minus["position_angle"].to_numpy(dtype=float),
        )
        rendered_e1_plus, _ = ellipticity_from_axis_ratio_angle(
            rendered_plus["axis_ratio"].to_numpy(dtype=float),
            rendered_plus["position_angle"].to_numpy(dtype=float),
        )

        def rendered_decomposition(subset):
            parent = float(
                (rendered_e1_plus[subset].mean() - rendered_e1_minus[subset].mean())
                / (2 * args.shear_magnitude)
            )
            detected = float(
                (
                    rendered_e1_plus[subset][label_plus[subset]].mean()
                    - rendered_e1_minus[subset][label_minus[subset]].mean()
                )
                / (2 * args.shear_magnitude)
            )
            return {
                "parent": parent,
                "detected": detected,
                "detected_minus_parent": detected - parent,
            }

        rendered_global_cases.append(
            rendered_decomposition(np.ones(len(minus), dtype=bool))
        )
        domain_mask = np.zeros(len(minus), dtype=bool)
        domain_mask[rows] = True
        rendered_domain_cases.append(rendered_decomposition(domain_mask))
        choice = choose_representative_neighbours(
            rendered_plus,
            rows,
            radius_arcsec=args.radius_arcsec,
            neighbour_selection="impact",
            impact_exponent=args.impact_exponent,
        )
        frame_minus = build_detection_feature_frame(
            rendered_minus, choice, conditions=CONDITIONS
        )
        frame_plus = build_detection_feature_frame(
            rendered_plus, choice, conditions=CONDITIONS
        )
        if not np.array_equal(
            frame_minus["input_index"].to_numpy(),
            frame_plus["input_index"].to_numpy(),
        ):
            raise ValueError(f"case {case}: paired input identifiers are not aligned")
        if not np.allclose(
            frame_minus[INVARIANT_FEATURES].to_numpy(dtype=float),
            frame_plus[INVARIANT_FEATURES].to_numpy(dtype=float),
            rtol=0,
            atol=0,
            equal_nan=True,
        ):
            raise ValueError(f"case {case}: non-morphology classifier input changed")

        changed = changed_morphology(frame_minus, frame_plus)
        selected_minus = label_minus[rows]
        selected_plus = label_plus[rows]
        selected_e1 = e1_intrinsic[rows]
        domain_truth = antithetic_detection_response(
            selected_e1,
            selected_minus.astype(float),
            selected_plus.astype(float),
            args.shear_magnitude,
        )
        truth_domain_cases.append(domain_truth)
        combined_labels["minus"].append(selected_minus)
        combined_labels["plus"].append(selected_plus)
        combined_labels["eligible"].append(changed)

        both = selected_minus & selected_plus
        plus_only = ~selected_minus & selected_plus
        minus_only = selected_minus & ~selected_plus
        neither = ~selected_minus & ~selected_plus
        case_reports.append(
            {
                "case": case,
                "parent_rows": int(len(minus)),
                "domain_rows": int(len(rows)),
                "truth_global_response": truth_global_cases[-1],
                "truth_domain_response": domain_truth,
                "rendered_global_decomposition": rendered_global_cases[-1],
                "rendered_domain_decomposition": rendered_domain_cases[-1],
                "morphology_changed_rows": int(changed.sum()),
                "state_counts_domain": {
                    "00": int(neither.sum()),
                    "01_plus_only": int(plus_only.sum()),
                    "10_minus_only": int(minus_only.sum()),
                    "11": int(both.sum()),
                },
                "plus_only_fraction_of_plus_detections": float(
                    plus_only.sum() / max(selected_plus.sum(), 1)
                ),
                "minus_only_fraction_of_minus_detections": float(
                    minus_only.sum() / max(selected_minus.sum(), 1)
                ),
                "labels_minus": report_minus,
                "labels_plus": report_plus,
            }
        )

        for name, bundle in bundles.items():
            logits_minus = predict_logits(
                bundle, frame_minus, args.prediction_batch_size
            )
            logits_plus = predict_logits(
                bundle, frame_plus, args.prediction_batch_size
            )
            p_minus, p_plus = probability(logits_minus), probability(logits_plus)
            combined_logits[name]["minus"].append(logits_minus)
            combined_logits[name]["plus"].append(logits_plus)
            combined_logits[name]["response"].append(
                {
                    "truth": domain_truth,
                    "model": antithetic_detection_response(
                        selected_e1,
                        p_minus,
                        p_plus,
                        args.shear_magnitude,
                    ),
                }
            )
        print(
            f"case={case} domain={len(rows)} plus_only={int(plus_only.sum())} "
            f"minus_only={int(minus_only.sum())} Rtruth={domain_truth:+.6f}",
            flush=True,
        )

    y_minus = np.concatenate(combined_labels["minus"])
    y_plus = np.concatenate(combined_labels["plus"])
    eligible = np.concatenate(combined_labels["eligible"])
    models = {}
    for name in bundles:
        logits_minus = np.concatenate(combined_logits[name]["minus"])
        logits_plus = np.concatenate(combined_logits[name]["plus"])
        p_minus, p_plus = probability(logits_minus), probability(logits_plus)
        models[name] = {
            "classification": {
                "minus_leg": classifier_metrics(y_minus, p_minus),
                "plus_leg": classifier_metrics(y_plus, p_plus),
                "combined": classifier_metrics(
                    np.concatenate([y_minus, y_plus]),
                    np.concatenate([p_minus, p_plus]),
                ),
            },
            "transition": transition_metrics(
                y_minus, y_plus, logits_minus, logits_plus, eligible
            ),
            "fixed_intrinsic_detection_response": summarize_response(
                combined_logits[name]["response"]
            ),
        }

    report = {
        "format_version": 1,
        "created_unix": time.time(),
        "simulation_root": str(args.simulation_root.resolve()),
        "cases": args.cases,
        "shear_arms": {"minus": minus_label, "plus": plus_label},
        "estimator": "(intrinsic-e1 selection shift plus - minus) / (2g)",
        "features": FEATURES,
        "domain_cut": "18 < r < 28 and 0.1 < Re < 1.5",
        "radius_arcsec": args.radius_arcsec,
        "impact_exponent": args.impact_exponent,
        "max_primaries_per_case": args.max_primaries_per_case,
        "sampling_seed": args.sampling_seed,
        "domain_pairs": int(len(y_minus)),
        "truth_global": truth_summary(truth_global_cases),
        "truth_classifier_domain": truth_summary(truth_domain_cases),
        "truth_rendered_decomposition_global": rendered_decomposition_summary(
            rendered_global_cases
        ),
        "truth_rendered_decomposition_classifier_domain": (
            rendered_decomposition_summary(rendered_domain_cases)
        ),
        "models": models,
        "case_reports": case_reports,
        "model_provenance": model_provenance,
        "constgold_firewall": (
            "evaluation only: no fitting, calibration, early stopping, checkpoint "
            "selection, or hyperparameter choice used ConstGold"
        ),
        "elapsed_seconds": float(time.time() - started),
    }
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"report={args.output}", flush=True)


if __name__ == "__main__":
    main()
