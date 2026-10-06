#!/usr/bin/env python3
"""Evaluate fixed-g0 flow stacks on antithetic ConstGold catalogues.

The population is selected exactly once by the measured g=0 anchor.  Its exact
keys are carried into the +/-h ConstGold legs; MAG_AUTO and FLUX_RADIUS are not
re-evaluated there.  Results distinguish a common both-usable matched sample,
the actual per-leg usable flags, and modeled per-leg usability.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sbsi.fixed_g0_domain import FLOW_FEATURES
from sbsi.flow_paired_shape import physical_context_means
from sbsi.measurement_model import load_measurement_model
from sbsi.selection_model import load_selection_model_ensemble
from sbsi.shear_map import apply_shear_to_ellipticity


BRANCHES = ("matched_usable", "actual_usable_flags", "modeled_usable")


def file_sha256(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8_388_608), b""):
            digest.update(block)
    return digest.hexdigest()


def json_ready(value):
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(json_ready(value), indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    os.replace(temporary, path)


def parse_mapping(values: list[str], option: str) -> dict[str, Path]:
    result = {}
    for value in values:
        label, separator, raw_path = value.partition("=")
        if not separator or not label or not raw_path:
            raise ValueError(f"{option} entries must have form LABEL=PATH")
        if label in result:
            raise ValueError(f"duplicate {option} label {label!r}")
        result[label] = Path(raw_path).resolve()
    return result


def parse_subsets(values: list[str], cases: tuple[int, ...]) -> dict[str, tuple[int, ...]]:
    if not values:
        return {"all": cases}
    allowed = set(cases)
    result = {}
    for value in values:
        label, separator, bounds = value.partition("=")
        first, colon, stop = bounds.partition(":")
        if not separator or not colon or not label:
            raise ValueError("--subset entries must have form LABEL=FIRST:STOP")
        selected = tuple(case for case in cases if int(first) <= case < int(stop))
        if not selected or not set(selected) <= allowed or label in result:
            raise ValueError(f"invalid subset {value!r}")
        result[label] = selected
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--constgold-root", type=Path, required=True)
    parser.add_argument(
        "--anchor-pattern", required=True, help="case NPZ format string containing {case}"
    )
    parser.add_argument("--model", action="append", required=True)
    parser.add_argument("--rblend", action="append", required=True)
    parser.add_argument("--classifier", type=Path, action="append", required=True)
    parser.add_argument("--case", type=int, action="append", required=True)
    parser.add_argument("--subset", action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--h", type=float, default=0.02)
    parser.add_argument("--draws", type=int, default=64)
    parser.add_argument("--sampling-seed", type=int, default=7301)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--n-boot", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260914)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


def raw_root(root: Path, case: int, shear: float) -> Path:
    return root / f"case{case}_{float(shear):.2f}" / "real0" / "catalogues"


def load_anchor(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path) as data:
        ids = np.asarray(data["input_index"], dtype=np.int64)
        context = np.asarray(data["context"], dtype=np.float64)
        gamma = np.asarray(data["gamma"], dtype=np.float64)
    if (
        ids.ndim != 1
        or context.shape != (len(ids), len(FLOW_FEATURES))
        or gamma.shape != (len(ids), 2)
        or len(np.unique(ids)) != len(ids)
        or not np.isfinite(context).all()
        or not np.array_equal(gamma, np.zeros_like(gamma))
    ):
        raise ValueError(f"invalid zero-shear anchor file {path}")
    order = np.argsort(ids, kind="stable")
    return ids[order], context[order]


def shear_frames(context: np.ndarray, h: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    e1 = context[:, 0]
    e2 = context[:, 1]
    plus1, plus2 = apply_shear_to_ellipticity(e1, e2, h, 0.0)
    minus1, minus2 = apply_shear_to_ellipticity(e1, e2, -h, 0.0)
    plus = context.copy()
    minus = context.copy()
    plus[:, :2] = np.column_stack((plus1, plus2))
    minus[:, :2] = np.column_stack((minus1, minus2))
    return (
        pd.DataFrame(plus, columns=FLOW_FEATURES),
        pd.DataFrame(minus, columns=FLOW_FEATURES),
    )


def load_measured_leg(
    root: Path, case: int, shear: float, anchor_ids: np.ndarray
) -> dict:
    catalogue = raw_root(root, case, shear)
    cross_path = catalogue / "CrossMatch" / "tile180.0_-0.5_rot0_matched.feather"
    shape_path = (
        catalogue
        / "Shapes"
        / "shape_catalogue_detect_position_all_tile180.0_-0.5.feather"
    )
    cross = pd.read_feather(cross_path, columns=["id_detec", "id_input"])
    shape = pd.read_feather(shape_path, columns=["NUMBER", "NGMIX_G1", "NGMIX_G2"])
    if (
        cross["id_input"].duplicated().any()
        or cross["id_detec"].duplicated().any()
        or shape["NUMBER"].duplicated().any()
    ):
        raise ValueError(f"non-unique raw identities in ConstGold case {case} shear {shear}")
    raw_detected = np.isin(anchor_ids, cross["id_input"].to_numpy(np.int64))
    joined = cross.merge(
        shape,
        left_on="id_detec",
        right_on="NUMBER",
        how="inner",
        validate="one_to_one",
        sort=False,
    ).set_index("id_input")
    aligned = joined.reindex(anchor_ids)
    values = aligned[["NGMIX_G1", "NGMIX_G2"]].to_numpy(np.float64)
    usable = np.isfinite(values).all(axis=1)
    usable &= np.square(values[:, 0]) + np.square(values[:, 1]) < 1.0
    return {
        "values": values,
        "usable": usable,
        "raw_detected": raw_detected,
        "raw_cross_rows": int(len(cross)),
        "raw_shape_rows": int(len(shape)),
        "cross_without_shape": int(len(cross) - len(joined)),
        "shape_without_cross": int(len(shape) - len(joined)),
        "anchor_raw_detected": int(raw_detected.sum()),
        "anchor_usable": int(usable.sum()),
    }


def load_rblend(
    frame: pd.DataFrame, path: Path, case: int, anchor_ids: np.ndarray
) -> np.ndarray:
    frame = frame.loc[frame["case"] == case]
    if frame["input_index"].duplicated().any():
        raise ValueError(f"duplicate R_blend keys for case {case} in {path}")
    aligned = frame.set_index("input_index")["R_blend"].reindex(anchor_ids)
    if aligned.isna().any():
        raise RuntimeError(
            f"R_blend lookup {path} misses {int(aligned.isna().sum())} anchor keys "
            f"in case {case}"
        )
    values = aligned.to_numpy(np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"non-finite R_blend in {path}")
    return values


def sufficient(values: np.ndarray, weights: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if (
        values.ndim != 2
        or values.shape[1] != 2
        or weights.shape != (len(values),)
        or not np.isfinite(values).all()
        or not np.isfinite(weights).all()
        or np.any(weights < 0)
        or weights.sum() <= 0
    ):
        raise ValueError("invalid response sufficient statistics")
    return {
        "denominator": float(weights.sum()),
        "numerator": np.sum(weights[:, None] * values, axis=0, dtype=np.float64),
    }


@torch.no_grad()
def model_means(
    bundle,
    plus: pd.DataFrame,
    minus: pd.DataFrame,
    *,
    draws: int,
    batch_size: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    contexts = (bundle.context_tensor(plus), bundle.context_tensor(minus))
    outputs = [np.empty((len(plus), 2), dtype=np.float64) for _ in range(2)]
    for block, start in enumerate(range(0, len(plus), batch_size)):
        stop = min(start + batch_size, len(plus))
        means = physical_context_means(
            bundle.model,
            [context[start:stop] for context in contexts],
            draws,
            seed + 1_000_003 * block,
        )
        for direction in range(2):
            outputs[direction][start:stop] = means[direction][:, :2].cpu().numpy()
    return outputs[0], outputs[1]


def response(plus: dict, minus: dict, h: float) -> np.ndarray:
    plus_mean = np.asarray(plus["numerator"], dtype=np.float64) / plus["denominator"]
    minus_mean = np.asarray(minus["numerator"], dtype=np.float64) / minus["denominator"]
    return (plus_mean - minus_mean) / (2.0 * h)


def pooled_sufficient(per_case: dict, cases: tuple[int, ...], branch: str, direction: str):
    return {
        "denominator": float(
            sum(per_case[str(case)][branch][direction]["denominator"] for case in cases)
        ),
        "numerator": sum(
            (
                np.asarray(per_case[str(case)][branch][direction]["numerator"], float)
                for case in cases
            ),
            start=np.zeros(2, dtype=np.float64),
        ),
    }


def bootstrap_draws(
    measured: dict,
    predicted: dict,
    cases: tuple[int, ...],
    branch: str,
    *,
    h: float,
    replicates: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(cases), size=(replicates, len(cases)))
    offsets = len(cases) * np.arange(replicates, dtype=np.int64)[:, None]
    weights = np.bincount(
        (indices + offsets).ravel(), minlength=replicates * len(cases)
    ).reshape(replicates, len(cases))

    def arrays(source, direction):
        denominator = np.asarray(
            [source[str(case)][branch][direction]["denominator"] for case in cases]
        )
        numerator = np.asarray(
            [source[str(case)][branch][direction]["numerator"] for case in cases]
        )
        return denominator, numerator

    responses = []
    for source in (measured, predicted):
        dp, np_ = arrays(source, "plus")
        dm, nm = arrays(source, "minus")
        mean_plus = np.einsum("bc,cj->bj", weights, np_) / np.einsum(
            "bc,c->b", weights, dp
        )[:, None]
        mean_minus = np.einsum("bc,cj->bj", weights, nm) / np.einsum(
            "bc,c->b", weights, dm
        )[:, None]
        responses.append((mean_plus - mean_minus) / (2.0 * h))
    measured_response, predicted_response = responses
    m = 100.0 * (measured_response[:, 0] / predicted_response[:, 0] - 1.0)
    residual = predicted_response - measured_response
    return m, residual


def bootstrap_m(
    measured: dict,
    predicted: dict,
    cases: tuple[int, ...],
    branch: str,
    *,
    h: float,
    replicates: int,
    seed: int,
) -> dict:
    m, residual = bootstrap_draws(
        measured,
        predicted,
        cases,
        branch,
        h=h,
        replicates=replicates,
        seed=seed,
    )
    return {
        "unit": "ConstGold case",
        "replicates": int(replicates),
        "seed": int(seed),
        "m_standard_error_percentage_points": float(m.std(ddof=1)),
        "m_ci95_percent": np.quantile(m, [0.025, 0.975]),
        "predicted_minus_measured_standard_error": residual.std(axis=0, ddof=1),
        "predicted_minus_measured_ci95": np.quantile(
            residual, [0.025, 0.975], axis=0
        ),
    }


def summarize_subset(
    measured: dict,
    predicted_by_model: dict,
    cases: tuple[int, ...],
    *,
    h: float,
    replicates: int,
    seed: int,
) -> dict:
    result = {"cases": list(cases), "n_cases": len(cases), "models": {}}
    measured_responses = {}
    for branch in BRANCHES:
        measured_responses[branch] = response(
            pooled_sufficient(measured, cases, branch, "plus"),
            pooled_sufficient(measured, cases, branch, "minus"),
            h,
        )
    for label, predicted in predicted_by_model.items():
        model_result = {}
        for branch_index, branch in enumerate(BRANCHES):
            predicted_response = response(
                pooled_sufficient(predicted, cases, branch, "plus"),
                pooled_sufficient(predicted, cases, branch, "minus"),
                h,
            )
            measured_response = measured_responses[branch]
            model_result[branch] = {
                "definition": {
                    "matched_usable": (
                        "same fixed-S0 identities usable in both ConstGold legs; "
                        "no sheared-leg magnitude/radius selection"
                    ),
                    "actual_usable_flags": (
                        "fixed-S0 cohort with the actual per-leg usable flags; no matching"
                    ),
                    "modeled_usable": (
                        "fixed-S0 cohort with modeled per-leg p(U|x); measured comparator "
                        "uses actual per-leg usable flags"
                    ),
                }[branch],
                "measured_response": measured_response,
                "predicted_response": predicted_response,
                "m_percent": float(
                    100.0 * (measured_response[0] / predicted_response[0] - 1.0)
                ),
                "bootstrap": bootstrap_m(
                    measured,
                    predicted,
                    cases,
                    branch,
                    h=h,
                    replicates=replicates,
                    # Reuse identical case resamples across model stacks so
                    # subsequent old/new differences remain paired.
                    seed=seed + 97 * branch_index,
                ),
            }
        result["models"][label] = model_result
    result["paired_model_differences"] = {}
    labels = list(predicted_by_model)
    for left_index, left in enumerate(labels):
        for right in labels[:left_index]:
            comparison = {}
            for branch_index, branch in enumerate(BRANCHES):
                left_draws, _ = bootstrap_draws(
                    measured,
                    predicted_by_model[left],
                    cases,
                    branch,
                    h=h,
                    replicates=replicates,
                    seed=seed + 97 * branch_index,
                )
                right_draws, _ = bootstrap_draws(
                    measured,
                    predicted_by_model[right],
                    cases,
                    branch,
                    h=h,
                    replicates=replicates,
                    seed=seed + 97 * branch_index,
                )
                difference = left_draws - right_draws
                point = (
                    result["models"][left][branch]["m_percent"]
                    - result["models"][right][branch]["m_percent"]
                )
                comparison[branch] = {
                    "definition": f"m({left}) - m({right}) on identical cases",
                    "difference_percentage_points": float(point),
                    "paired_case_bootstrap_standard_error": float(
                        difference.std(ddof=1)
                    ),
                    "paired_case_bootstrap_ci95": np.quantile(
                        difference, [0.025, 0.975]
                    ),
                }
            result["paired_model_differences"][f"{left}_minus_{right}"] = comparison
    return result


def main(argv=None):
    args = parse_args(argv)
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("ConstGold flow evaluation must run under Slurm")
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    if args.h <= 0 or args.draws < 2 or args.draws % 2 or args.batch_size <= 0:
        raise ValueError("positive h/batch and even draws >=2 required")
    cases = tuple(dict.fromkeys(int(case) for case in args.case))
    subsets = parse_subsets(args.subset, cases)
    models = parse_mapping(args.model, "--model")
    rblend_paths = parse_mapping(args.rblend, "--rblend")
    if set(models) != set(rblend_paths):
        raise ValueError("--model and --rblend labels must match exactly")
    for path in (*models.values(), *rblend_paths.values(), *args.classifier):
        if not path.is_file():
            raise FileNotFoundError(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    progress = output.with_suffix(".progress.json")

    rblend_tables = {}
    for label, path in rblend_paths.items():
        table = pd.read_feather(
            path, columns=["case", "input_index", "R_blend"]
        )
        if table.duplicated(["case", "input_index"]).any():
            raise ValueError(f"duplicate R_blend keys in {path}")
        if not set(cases) <= set(table["case"].unique()):
            raise ValueError(f"R_blend lookup {path} does not cover all cases")
        rblend_tables[label] = table.loc[table["case"].isin(cases)].copy()

    device = torch.device(args.device)
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
    classifier = load_selection_model_ensemble(args.classifier, device=device)
    if classifier.preprocessor.feature_names != list(FLOW_FEATURES):
        raise ValueError("classifier feature contract differs from fixed-g0 flow")
    bundles = {}
    for label, path in models.items():
        bundle = load_measurement_model(path, device=device)
        if bundle.condition_preprocessor.feature_names != list(FLOW_FEATURES):
            raise ValueError(f"flow feature mismatch for {label}")
        if bundle.target_transform.target_names != [
            "measured_ngmix_g1",
            "measured_ngmix_g2",
            "measured_flux_radius",
            "measured_flux_from_mag_auto",
        ]:
            raise ValueError(f"flow target mismatch for {label}")
        if not callable(getattr(bundle.model, "forward_physical", None)):
            raise ValueError(f"{label} is not a physical flow")
        bundles[label] = bundle

    measured = {}
    predicted = {label: {} for label in models}
    case_reports = {}
    anchor_hashes = {}
    for case in cases:
        print(f"CONSTGOLD_FIXED_G0_CASE_START case={case}", flush=True)
        anchor_path = Path(args.anchor_pattern.format(case=case))
        anchor_hashes[str(case)] = file_sha256(anchor_path)
        anchor_ids, context = load_anchor(anchor_path)
        plus_frame, minus_frame = shear_frames(context, args.h)
        legs = {
            "plus": load_measured_leg(
                args.constgold_root, case, +args.h, anchor_ids
            ),
            "minus": load_measured_leg(
                args.constgold_root, case, -args.h, anchor_ids
            ),
        }
        common = legs["plus"]["usable"] & legs["minus"]["usable"]
        if not common.any():
            raise RuntimeError(f"case {case} has no matched usable fixed-g0 objects")
        probabilities = {
            "plus": np.asarray(classifier.predict_proba(plus_frame), np.float64),
            "minus": np.asarray(classifier.predict_proba(minus_frame), np.float64),
        }
        for direction in ("plus", "minus"):
            probability = probabilities[direction]
            if (
                probability.shape != (len(anchor_ids),)
                or not np.isfinite(probability).all()
                or np.any((probability < 0) | (probability > 1))
            ):
                raise ValueError("invalid classifier probability")

        measured[str(case)] = {}
        for branch in BRANCHES:
            measured[str(case)][branch] = {}
            for direction in ("plus", "minus"):
                mask = common if branch == "matched_usable" else legs[direction]["usable"]
                measured[str(case)][branch][direction] = sufficient(
                    legs[direction]["values"][mask], np.ones(int(mask.sum()))
                )

        base_e1 = context[:, 0]
        base_e2 = context[:, 1]
        plus_e1, plus_e2 = apply_shear_to_ellipticity(
            base_e1, base_e2, args.h, 0.0
        )
        minus_e1, minus_e2 = apply_shear_to_ellipticity(
            base_e1, base_e2, -args.h, 0.0
        )
        truth_delta = {
            "plus": np.column_stack((plus_e1 - base_e1, plus_e2 - base_e2)),
            "minus": np.column_stack((minus_e1 - base_e1, minus_e2 - base_e2)),
        }
        for model_index, (label, bundle) in enumerate(bundles.items()):
            print(f"CONSTGOLD_FIXED_G0_MODEL case={case} model={label}", flush=True)
            rblend = load_rblend(
                rblend_tables[label], rblend_paths[label], case, anchor_ids
            )
            plus_mean, minus_mean = model_means(
                bundle,
                plus_frame,
                minus_frame,
                draws=args.draws,
                batch_size=args.batch_size,
                seed=args.sampling_seed + 10_000_019 * case,
            )
            values = {
                "plus": plus_mean + rblend[:, None] * truth_delta["plus"],
                "minus": minus_mean + rblend[:, None] * truth_delta["minus"],
            }
            predicted[label][str(case)] = {}
            for branch in BRANCHES:
                predicted[label][str(case)][branch] = {}
                for direction in ("plus", "minus"):
                    if branch == "matched_usable":
                        weights = common.astype(np.float64)
                    elif branch == "actual_usable_flags":
                        weights = legs[direction]["usable"].astype(np.float64)
                    else:
                        weights = probabilities[direction]
                    predicted[label][str(case)][branch][direction] = sufficient(
                        values[direction], weights
                    )
            del plus_mean, minus_mean, values
            if device.type == "cuda":
                torch.cuda.empty_cache()

        case_reports[str(case)] = {
            "anchor_rows": int(len(anchor_ids)),
            "matched_usable_rows": int(common.sum()),
            "plus": {
                key: value for key, value in legs["plus"].items() if key not in {"values", "usable", "raw_detected"}
            },
            "minus": {
                key: value for key, value in legs["minus"].items() if key not in {"values", "usable", "raw_detected"}
            },
            "mean_modeled_usable_probability": {
                direction: float(probabilities[direction].mean())
                for direction in ("plus", "minus")
            },
            "unmatched_usable_rows": {
                "plus_only": int(
                    np.sum(legs["plus"]["usable"] & ~legs["minus"]["usable"])
                ),
                "minus_only": int(
                    np.sum(legs["minus"]["usable"] & ~legs["plus"]["usable"])
                ),
            },
        }
        write_json(
            progress,
            {
                "format_version": 1,
                "completed_cases": list(case_reports),
                "case_reports": case_reports,
            },
        )
        print(
            f"CONSTGOLD_FIXED_G0_CASE_DONE case={case} anchor={len(anchor_ids)} "
            f"matched={int(common.sum())}",
            flush=True,
        )

    result = {
        "format_version": 1,
        "domain": {
            "anchor": (
                "S0 evaluated only on measured g=0: detected, MAG_AUTO<25.8, "
                "FLUX_RADIUS>3.0 pixels (0.6 arcsec)"
            ),
            "truth_analysis_cut": None,
            "sheared_leg_magnitude_radius_recut": False,
            "usable_definition": "finite ngmix (e1,e2) with e1^2+e2^2<1; exact (0,0) is valid",
            "response_estimator": "[mean(e_plus)-mean(e_minus)]/(2h)",
            "m_definition": "100*(R11_measured/R11_model-1)",
        },
        "h": float(args.h),
        "draws": int(args.draws),
        "sampling_seed": int(args.sampling_seed),
        "common_antithetic_latents_across_legs_and_models": True,
        "models": {
            label: {
                "flow": str(path),
                "flow_sha256": file_sha256(path),
                "rblend": str(rblend_paths[label]),
                "rblend_sha256": file_sha256(rblend_paths[label]),
            }
            for label, path in models.items()
        },
        "classifiers": [
            {"path": str(path), "sha256": file_sha256(path)}
            for path in args.classifier
        ],
        "case_reports": case_reports,
        "anchor_sha256_by_case": anchor_hashes,
        "per_case_sufficient": {
            "measured": measured,
            "predicted": predicted,
        },
        "subsets": {
            label: summarize_subset(
                measured,
                predicted,
                selected_cases,
                h=args.h,
                replicates=args.n_boot,
                seed=args.bootstrap_seed + 100_003 * subset_index,
            )
            for subset_index, (label, selected_cases) in enumerate(subsets.items())
        },
        "limitations": [
            "ConstGold cases reuse finite truth scenes seen by some training recipes, but use independent all-sheared image measurements.",
            "The modeled-usable branch uses p(U|x) trained on the full parent; it does not reapply the fixed-g0 measured selection.",
            "One flow-training seed and one antithetic integration seed do not quantify training or integration uncertainty.",
            "Historical per-leg selected results have a different population and are not an apples-to-apples baseline for this fixed-cohort test.",
        ],
    }
    write_json(output, result)
    print(f"CONSTGOLD_FIXED_G0_COMPLETE output={output}", flush=True)


if __name__ == "__main__":
    main()
