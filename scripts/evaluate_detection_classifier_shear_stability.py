#!/usr/bin/env python3
"""Evaluate frozen paired detection classifiers on new half-shear cases.

The evaluator applies an existing single-leg classifier independently to the
zero and forward legs.  It reports marginal detection metrics, paired
transition metrics, and centered detection-weighted shape responses without
fitting, calibrating, or selecting a checkpoint.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

from sbsi.detection_classifier import (
    build_detection_feature_frame,
    choose_representative_neighbours,
    detection_selection_response,
    render_catalogue_shape_shear,
)
from sbsi.selection_model import load_selection_model
from scripts.train_detection_classifier_transition_pair import (
    CONDITIONS,
    FEATURES,
    INVARIANT_FEATURES,
    aligned_shapes,
    catalogue_paths,
    changed_morphology,
    classifier_metrics,
    detection_labels,
    file_sha256,
    mean_sem,
    parse_cases,
    transition_metrics,
)


def parse_model(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("model must be NAME=PATH")
    name, raw_path = value.split("=", 1)
    name = name.strip()
    path = Path(raw_path).expanduser()
    if not name:
        raise argparse.ArgumentTypeError("model name must be nonempty")
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"model checkpoint does not exist: {path}")
    return name, path


@torch.no_grad()
def predict_logits(bundle, frame: pd.DataFrame, batch_size: int) -> np.ndarray:
    output = []
    for start in range(0, len(frame), batch_size):
        raw = bundle.preprocessor.raw_tensor_from_frame(
            frame.iloc[start : start + batch_size], bundle.device
        )
        output.append(bundle.logits_from_raw_tensor(raw).cpu().numpy())
    return np.concatenate(output) if output else np.empty(0, dtype=np.float32)


def probability(logits: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))


def summarize_response(records: list[dict]) -> dict:
    output = {}
    for kind in ("fixed_intrinsic", "rendered_per_leg"):
        truth = [record[kind]["truth"] for record in records]
        model = [record[kind]["model"] for record in records]
        residual = [m - t for m, t in zip(model, truth)]
        truth_mean, truth_sem = mean_sem(truth)
        model_mean, model_sem = mean_sem(model)
        residual_mean, residual_sem = mean_sem(residual)
        output[kind] = {
            "truth": truth_mean,
            "truth_case_sem": truth_sem,
            "model": model_mean,
            "model_case_sem": model_sem,
            "model_minus_truth": residual_mean,
            "paired_case_sem": residual_sem,
            "truth_per_case": truth,
            "model_per_case": model,
        }
    return output


def paired_amplitude_comparison(results: dict, model_names: list[str]) -> dict:
    labels = list(results)
    if len(labels) != 2:
        return {}
    low, high = sorted(labels, key=float)
    output = {"low_shear_label": low, "high_shear_label": high, "models": {}}
    for name in model_names:
        output["models"][name] = {}
        for kind in ("fixed_intrinsic", "rendered_per_leg"):
            low_result = results[low]["models"][name]["response"][kind]
            high_result = results[high]["models"][name]["response"][kind]
            item = {}
            for source, key in (("truth", "truth_per_case"), ("model", "model_per_case")):
                delta = np.asarray(low_result[key]) - np.asarray(high_result[key])
                delta_mean, delta_sem = mean_sem(delta.tolist())
                item[f"{source}_low_minus_high"] = delta_mean
                item[f"{source}_low_minus_high_case_sem"] = delta_sem
            low_residual = np.asarray(low_result["model_per_case"]) - np.asarray(
                low_result["truth_per_case"]
            )
            high_residual = np.asarray(high_result["model_per_case"]) - np.asarray(
                high_result["truth_per_case"]
            )
            delta_residual, delta_residual_sem = mean_sem(
                (low_residual - high_residual).tolist()
            )
            item["residual_low_minus_high"] = delta_residual
            item["residual_low_minus_high_case_sem"] = delta_residual_sem
            output["models"][name][kind] = item
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulation-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", type=parse_cases, default=parse_cases("140-159"))
    parser.add_argument("--zero-shear-label", default="0.0")
    parser.add_argument("--shear-labels", nargs="+", default=["0.02", "0.05"])
    parser.add_argument("--model", action="append", type=parse_model, required=True)
    parser.add_argument("--radius-arcsec", type=float, default=3.0)
    parser.add_argument("--impact-exponent", type=float, default=1.0)
    parser.add_argument("--max-primaries-per-case", type=int, default=50_000)
    parser.add_argument("--sampling-seed", type=int, default=20260824)
    parser.add_argument("--prediction-batch-size", type=int, default=65_536)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if len(args.shear_labels) != len(set(args.shear_labels)):
        raise ValueError("shear labels must be unique")
    if len(args.model) != len({name for name, _ in args.model}):
        raise ValueError("model names must be unique")
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
            raise ValueError(
                f"{name}: checkpoint features differ from stability evaluator"
            )
        bundles[name] = bundle
        model_provenance[name] = {
            "path": str(path.resolve()),
            "sha256": file_sha256(path),
            "metadata": bundle.metadata,
        }

    started = time.time()
    results = {}
    for shear_label in args.shear_labels:
        combined_y0, combined_yg, combined_eligible = [], [], []
        combined_logits = {
            name: {"zero": [], "sheared": [], "response": []}
            for name in bundles
        }
        case_reports = []
        for case in args.cases:
            truth0_path, matched0_path = catalogue_paths(
                args.simulation_root, case, args.zero_shear_label
            )
            truthg_path, matchedg_path = catalogue_paths(
                args.simulation_root, case, shear_label
            )
            g0 = pd.read_feather(truth0_path)
            gg = pd.read_feather(truthg_path)
            if len(g0) != len(gg):
                raise ValueError(f"case {case}: truth legs have different row counts")
            for column in ("index", "RA", "DEC", "r", "Re", "sersic_n"):
                if not np.array_equal(g0[column].to_numpy(), gg[column].to_numpy()):
                    raise ValueError(f"case {case}: invariant {column} differs between legs")

            y0, report0 = detection_labels(g0, matched0_path)
            yg, reportg = detection_labels(gg, matchedg_path)
            eligible_rows = np.flatnonzero(
                (gg["r"].to_numpy(dtype=float) > 18.0)
                & (gg["r"].to_numpy(dtype=float) < 28.0)
                & (gg["Re"].to_numpy(dtype=float) > 0.1)
                & (gg["Re"].to_numpy(dtype=float) < 1.5)
            )
            rng = np.random.default_rng(args.sampling_seed + case)
            if len(eligible_rows) > args.max_primaries_per_case:
                rows = np.sort(
                    rng.choice(
                        eligible_rows,
                        size=args.max_primaries_per_case,
                        replace=False,
                    )
                )
            else:
                rows = eligible_rows

            rendered_g = render_catalogue_shape_shear(gg)
            choice = choose_representative_neighbours(
                rendered_g,
                rows,
                radius_arcsec=args.radius_arcsec,
                neighbour_selection="impact",
                impact_exponent=args.impact_exponent,
            )
            frame0 = build_detection_feature_frame(g0, choice, conditions=CONDITIONS)
            frameg = build_detection_feature_frame(
                rendered_g, choice, conditions=CONDITIONS
            )
            if not np.array_equal(
                frame0["input_index"].to_numpy(), frameg["input_index"].to_numpy()
            ):
                raise ValueError(f"case {case}: paired input identifiers are not aligned")
            if not np.allclose(
                frame0[INVARIANT_FEATURES].to_numpy(dtype=float),
                frameg[INVARIANT_FEATURES].to_numpy(dtype=float),
                rtol=0,
                atol=0,
                equal_nan=True,
            ):
                raise ValueError(f"case {case}: non-morphology input changed")

            changed = changed_morphology(frame0, frameg)
            selected_y0, selected_yg = y0[rows], yg[rows]
            e0_parallel, eg_parallel, shear = aligned_shapes(g0, rendered_g, rows)
            nonzero = shear > 0
            nominal = float(shear_label)
            if not np.allclose(shear[nonzero], nominal, rtol=0, atol=1.0e-6):
                raise ValueError(
                    f"case {case}: stored nonzero shear does not match {shear_label}"
                )

            combined_y0.append(selected_y0)
            combined_yg.append(selected_yg)
            combined_eligible.append(changed)
            discordant = changed & (selected_y0 != selected_yg)
            case_reports.append(
                {
                    "case": case,
                    "sampled_rows": int(len(rows)),
                    "primary_sheared_rows": int(nonzero.sum()),
                    "morphology_changed_rows": int(changed.sum()),
                    "eligible_discordant_rows": int(discordant.sum()),
                    "state_counts": {
                        f"{a}{b}": int(
                            (changed & (selected_y0 == bool(a)) & (selected_yg == bool(b))).sum()
                        )
                        for a in (0, 1)
                        for b in (0, 1)
                    },
                    "labels_zero": report0,
                    "labels_sheared": reportg,
                }
            )

            for name, bundle in bundles.items():
                logits0 = predict_logits(bundle, frame0, args.prediction_batch_size)
                logitsg = predict_logits(bundle, frameg, args.prediction_batch_size)
                p0, pg = probability(logits0), probability(logitsg)
                combined_logits[name]["zero"].append(logits0)
                combined_logits[name]["sheared"].append(logitsg)
                response = {}
                for kind, zero_shape, sheared_shape in (
                    ("fixed_intrinsic", e0_parallel, e0_parallel),
                    ("rendered_per_leg", e0_parallel, eg_parallel),
                ):
                    truth_response = detection_selection_response(
                        zero_shape,
                        sheared_shape,
                        selected_y0.astype(float),
                        selected_yg.astype(float),
                        shear,
                    )
                    model_response = detection_selection_response(
                        zero_shape, sheared_shape, p0, pg, shear
                    )
                    response[kind] = {
                        "truth": truth_response,
                        "model": model_response,
                    }
                combined_logits[name]["response"].append(response)
            print(
                f"shear={shear_label} case={case} rows={len(rows)} "
                f"primary_sheared={int(nonzero.sum())} discordant={int(discordant.sum())}",
                flush=True,
            )

        y0_all = np.concatenate(combined_y0)
        yg_all = np.concatenate(combined_yg)
        eligible_all = np.concatenate(combined_eligible)
        models = {}
        for name in bundles:
            logits0 = np.concatenate(combined_logits[name]["zero"])
            logitsg = np.concatenate(combined_logits[name]["sheared"])
            p0, pg = probability(logits0), probability(logitsg)
            models[name] = {
                "classification": {
                    "zero_leg": classifier_metrics(y0_all, p0),
                    "sheared_leg": classifier_metrics(yg_all, pg),
                    "combined": classifier_metrics(
                        np.concatenate([y0_all, yg_all]),
                        np.concatenate([p0, pg]),
                    ),
                },
                "transition": transition_metrics(
                    y0_all, yg_all, logits0, logitsg, eligible_all
                ),
                "response": summarize_response(combined_logits[name]["response"]),
            }
        results[shear_label] = {
            "case_reports": case_reports,
            "sampled_pairs": int(len(y0_all)),
            "models": models,
        }

    report = {
        "format_version": 1,
        "created_unix": time.time(),
        "simulation_root": str(args.simulation_root.resolve()),
        "cases": args.cases,
        "zero_shear_label": args.zero_shear_label,
        "shear_labels": args.shear_labels,
        "features": FEATURES,
        "radius_arcsec": args.radius_arcsec,
        "impact_exponent": args.impact_exponent,
        "max_primaries_per_case": args.max_primaries_per_case,
        "sampling_seed": args.sampling_seed,
        "model_provenance": model_provenance,
        "frozen_evaluation": (
            "no fitting, calibration, early stopping, checkpoint selection, or "
            "hyperparameter choice was performed"
        ),
        "constgold_firewall": (
            f"only half-shear cases {args.cases} were read; the ConstGold catalogue "
            "was not read"
        ),
        "results": results,
        "paired_amplitude_comparison": paired_amplitude_comparison(
            results, list(bundles)
        ),
        "elapsed_seconds": float(time.time() - started),
    }
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"report={args.output}", flush=True)


if __name__ == "__main__":
    main()
