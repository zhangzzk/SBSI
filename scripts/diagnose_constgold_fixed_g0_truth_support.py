#!/usr/bin/env python3
"""Localize fixed-g0 matched-response bias by the former truth support.

This is a bridge diagnostic, not a new calibration estimator.  It keeps the
fixed measured-g0 anchor and the fixed both-usable ConstGold intersection, then
partitions those exact objects by whether they would have passed the former
truth-domain cuts.  On the overlap it can compare historical and current model
stacks against an identical measured numerator.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sbsi.fixed_g0_domain import FLOW_FEATURES
from sbsi.measurement_model import load_measurement_model
from sbsi.shear_map import apply_shear_to_ellipticity
from scripts.evaluate_constgold_fixed_g0_response import (
    load_anchor,
    load_measured_leg,
    model_means,
    parse_mapping,
    raw_root,
    response,
    shear_frames,
    sufficient,
    write_json,
)


PARTITION = (
    "inside_old_truth_support",
    "outside_magnitude_only",
    "outside_size_only",
    "outside_both",
)
GROUPS = (
    "all_fixed_g0",
    *PARTITION,
    "outside_old_truth_support",
    "truth_faint",
    "truth_bright",
    "truth_small",
    "truth_large",
)


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8_388_608), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--constgold-root", type=Path, required=True)
    parser.add_argument("--anchor-pattern", required=True)
    parser.add_argument("--model", action="append", required=True)
    parser.add_argument("--rblend", action="append", required=True)
    parser.add_argument("--case", type=int, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--h", type=float, default=0.02)
    parser.add_argument("--draws", type=int, default=64)
    parser.add_argument("--sampling-seed", type=int, default=7301)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--n-boot", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260915)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


def truth_groups(
    r_input: np.ndarray, re_input: np.ndarray
) -> dict[str, np.ndarray]:
    """Return the former strict-support partition and useful directional tails."""
    r_input = np.asarray(r_input, dtype=np.float64)
    re_input = np.asarray(re_input, dtype=np.float64)
    if r_input.shape != re_input.shape or r_input.ndim != 1:
        raise ValueError("truth arrays must be aligned one-dimensional arrays")
    if not np.isfinite(r_input).all() or not np.isfinite(re_input).all():
        raise ValueError("truth arrays contain non-finite values")
    mag_inside = (r_input > 18.0) & (r_input < 25.8)
    size_inside = (re_input > 0.37) & (re_input < 1.5)
    groups = {
        "all_fixed_g0": np.ones(len(r_input), dtype=bool),
        "inside_old_truth_support": mag_inside & size_inside,
        "outside_magnitude_only": ~mag_inside & size_inside,
        "outside_size_only": mag_inside & ~size_inside,
        "outside_both": ~mag_inside & ~size_inside,
        "outside_old_truth_support": ~(mag_inside & size_inside),
        "truth_faint": r_input >= 25.8,
        "truth_bright": r_input <= 18.0,
        "truth_small": re_input <= 0.37,
        "truth_large": re_input >= 1.5,
    }
    membership = sum(groups[name].astype(np.int8) for name in PARTITION)
    if not np.array_equal(membership, np.ones(len(r_input), dtype=np.int8)):
        raise RuntimeError("former truth-support groups do not form a partition")
    return groups


def load_truth_groups(
    root: Path, case: int, h: float, anchor_ids: np.ndarray
) -> dict[str, np.ndarray]:
    path = (
        raw_root(root, case, h)
        / "input"
        / "gals_info_tile180.0_-0.5.feather"
    )
    truth = pd.read_feather(
        path, columns=["index_input", "r_input", "Re_input"]
    )
    if truth["index_input"].duplicated().any():
        raise ValueError(f"duplicate truth identities in {path}")
    aligned = truth.set_index("index_input").reindex(anchor_ids)
    if aligned.isna().any().any():
        raise RuntimeError(f"truth catalogue {path} misses fixed-g0 anchor rows")
    return truth_groups(
        aligned["r_input"].to_numpy(float),
        aligned["Re_input"].to_numpy(float),
    )


def load_rblend_allow_missing(
    table: pd.DataFrame, case: int, anchor_ids: np.ndarray
) -> np.ndarray:
    selected = table.loc[table["case"] == case]
    if selected["input_index"].duplicated().any():
        raise ValueError(f"duplicate R_blend keys in case {case}")
    return (
        selected.set_index("input_index")["R_blend"]
        .reindex(anchor_ids)
        .to_numpy(np.float64)
    )


def pooled(per_case: dict, cases: tuple[int, ...], group: str, direction: str):
    return {
        "denominator": float(
            sum(per_case[str(case)][group][direction]["denominator"] for case in cases)
        ),
        "numerator": sum(
            (
                np.asarray(
                    per_case[str(case)][group][direction]["numerator"],
                    dtype=np.float64,
                )
                for case in cases
            ),
            start=np.zeros(2, dtype=np.float64),
        ),
    }


def bootstrap_weights(cases: tuple[int, ...], replicates: int, seed: int):
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(cases), size=(replicates, len(cases)))
    offsets = len(cases) * np.arange(replicates, dtype=np.int64)[:, None]
    return np.bincount(
        (sampled + offsets).ravel(), minlength=replicates * len(cases)
    ).reshape(replicates, len(cases))


def bootstrap_response(
    per_case: dict,
    cases: tuple[int, ...],
    group: str,
    weights: np.ndarray,
    h: float,
) -> np.ndarray:
    means = []
    for direction in ("plus", "minus"):
        denominator = np.asarray(
            [
                per_case[str(case)][group][direction]["denominator"]
                for case in cases
            ],
            dtype=np.float64,
        )
        numerator = np.asarray(
            [
                per_case[str(case)][group][direction]["numerator"]
                for case in cases
            ],
            dtype=np.float64,
        )
        means.append(
            np.einsum("bc,cj->bj", weights, numerator)
            / np.einsum("bc,c->b", weights, denominator)[:, None]
        )
    return (means[0] - means[1]) / (2.0 * h)


def uncertainty(draws: np.ndarray) -> dict:
    return {
        "standard_error": np.std(draws, axis=0, ddof=1),
        "ci95": np.quantile(draws, [0.025, 0.975], axis=0),
    }


def summarize(
    measured: dict,
    predicted: dict[str, dict],
    unavailable: dict[str, dict[str, list[int]]],
    cases: tuple[int, ...],
    *,
    h: float,
    replicates: int,
    seed: int,
) -> dict:
    weights = bootstrap_weights(cases, replicates, seed)
    measured_response = {}
    measured_draws = {}
    result = {"groups": {}, "models": {}, "paired_model_differences": {}}
    for group in GROUPS:
        pooled_plus = pooled(measured, cases, group, "plus")
        pooled_minus = pooled(measured, cases, group, "minus")
        point = response(pooled_plus, pooled_minus, h)
        draws = bootstrap_response(measured, cases, group, weights, h)
        measured_response[group] = point
        measured_draws[group] = draws
        result["groups"][group] = {
            "matched_rows": int(pooled_plus["denominator"]),
            "fraction_of_all_matched": float(
                pooled_plus["denominator"]
                / pooled(measured, cases, "all_fixed_g0", "plus")["denominator"]
            ),
            "measured_response": point,
            "measured_response_bootstrap": uncertainty(draws),
        }

    model_m_draws = {}
    for label, per_case in predicted.items():
        result["models"][label] = {}
        model_m_draws[label] = {}
        for group in GROUPS:
            missing_cases = unavailable[label][group]
            if missing_cases:
                result["models"][label][group] = {
                    "status": "not_evaluable_on_complete_group",
                    "cases_with_missing_rblend_rows": missing_cases,
                }
                continue
            point = response(
                pooled(per_case, cases, group, "plus"),
                pooled(per_case, cases, group, "minus"),
                h,
            )
            draws = bootstrap_response(per_case, cases, group, weights, h)
            m_point = 100.0 * (measured_response[group][0] / point[0] - 1.0)
            m_draws = 100.0 * (measured_draws[group][:, 0] / draws[:, 0] - 1.0)
            model_m_draws[label][group] = m_draws
            result["models"][label][group] = {
                "status": "complete",
                "predicted_response": point,
                "predicted_response_bootstrap": uncertainty(draws),
                "predicted_minus_measured": point - measured_response[group],
                "m_percent": float(m_point),
                "m_bootstrap": uncertainty(m_draws),
            }

        if all(
            result["models"][label].get(group, {}).get("status") == "complete"
            for group in ("all_fixed_g0", *PARTITION)
        ):
            overall_model = np.asarray(
                result["models"][label]["all_fixed_g0"]["predicted_response"]
            )
            denominator = result["groups"]["all_fixed_g0"]["matched_rows"]
            contributions = {}
            for group in PARTITION:
                fraction = result["groups"][group]["matched_rows"] / denominator
                residual = np.asarray(
                    result["models"][label][group]["predicted_minus_measured"]
                )
                contributions[group] = float(
                    -100.0 * fraction * residual[0] / overall_model[0]
                )
            total = sum(contributions.values())
            expected = result["models"][label]["all_fixed_g0"]["m_percent"]
            if not math.isclose(total, expected, rel_tol=0.0, abs_tol=2e-12):
                raise RuntimeError(f"m contribution identity failed for {label}")
            result["models"][label]["matched_m_decomposition"] = {
                "definition": (
                    "exact fixed-count partition: -100*f_group*"
                    "(R_model_group-R_measured_group)/R_model_all"
                ),
                "contribution_percentage_points": contributions,
                "sum_percentage_points": float(total),
            }

    labels = list(predicted)
    bridge_group = "inside_old_truth_support"
    for left_index, left in enumerate(labels):
        for right in labels[:left_index]:
            if (
                bridge_group not in model_m_draws[left]
                or bridge_group not in model_m_draws[right]
            ):
                continue
            draws = (
                model_m_draws[left][bridge_group]
                - model_m_draws[right][bridge_group]
            )
            point = (
                result["models"][left][bridge_group]["m_percent"]
                - result["models"][right][bridge_group]["m_percent"]
            )
            result["paired_model_differences"][f"{left}_minus_{right}"] = {
                "group": bridge_group,
                "difference_percentage_points": float(point),
                "bootstrap": uncertainty(draws),
            }
    return result


def main(argv=None) -> None:
    args = parse_args(argv)
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("ConstGold flow diagnostics must run under Slurm")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if args.h <= 0 or args.draws < 2 or args.draws % 2 or args.batch_size <= 0:
        raise ValueError("positive h/batch and even draws >=2 required")
    cases = tuple(dict.fromkeys(args.case))
    if len(cases) != len(args.case):
        raise ValueError("cases must be unique")
    models = parse_mapping(args.model, "--model")
    rblend_paths = parse_mapping(args.rblend, "--rblend")
    if set(models) != set(rblend_paths):
        raise ValueError("--model and --rblend labels must match")

    rblend_tables = {
        label: pd.read_feather(path, columns=["case", "input_index", "R_blend"])
        for label, path in rblend_paths.items()
    }
    for label, table in rblend_tables.items():
        if table.duplicated(["case", "input_index"]).any():
            raise ValueError(f"duplicate R_blend identities for {label}")

    device = torch.device(args.device)
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
    bundles = {}
    for label, path in models.items():
        bundle = load_measurement_model(path, device=device)
        if bundle.condition_preprocessor.feature_names != list(FLOW_FEATURES):
            raise ValueError(f"flow feature mismatch for {label}")
        expected_targets = [
            "measured_ngmix_g1",
            "measured_ngmix_g2",
            "measured_flux_radius",
            "measured_flux_from_mag_auto",
        ]
        if bundle.target_transform.target_names != expected_targets:
            raise ValueError(f"flow target mismatch for {label}")
        bundles[label] = bundle

    measured: dict[str, dict] = {}
    predicted: dict[str, dict] = {label: {} for label in models}
    unavailable = {
        label: {group: [] for group in GROUPS} for label in models
    }
    case_reports = {}
    for case in cases:
        print(f"CONSTGOLD_TRUTH_SUPPORT_START case={case}", flush=True)
        anchor_ids, context = load_anchor(Path(args.anchor_pattern.format(case=case)))
        groups = load_truth_groups(args.constgold_root, case, args.h, anchor_ids)
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
        measured[str(case)] = {}
        for group in GROUPS:
            mask = common & groups[group]
            measured[str(case)][group] = {
                direction: sufficient(
                    legs[direction]["values"][mask], np.ones(int(mask.sum()))
                )
                for direction in ("plus", "minus")
            }

        base_e1, base_e2 = context[:, 0], context[:, 1]
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
        for label, bundle in bundles.items():
            rblend = load_rblend_allow_missing(
                rblend_tables[label], case, anchor_ids
            )
            available = np.isfinite(rblend)
            if not available.any():
                raise RuntimeError(f"{label} has no R_blend coverage in case {case}")
            plus_mean, minus_mean = model_means(
                bundle,
                plus_frame.iloc[available].reset_index(drop=True),
                minus_frame.iloc[available].reset_index(drop=True),
                draws=args.draws,
                batch_size=args.batch_size,
                seed=args.sampling_seed + 10_000_019 * case,
            )
            values = {
                "plus": plus_mean + rblend[available, None] * truth_delta["plus"][available],
                "minus": minus_mean
                + rblend[available, None] * truth_delta["minus"][available],
            }
            available_index = np.flatnonzero(available)
            predicted[label][str(case)] = {}
            for group in GROUPS:
                selected = common & groups[group]
                if np.any(selected & ~available):
                    unavailable[label][group].append(int(case))
                    continue
                positions = np.searchsorted(available_index, np.flatnonzero(selected))
                predicted[label][str(case)][group] = {
                    direction: sufficient(
                        values[direction][positions], np.ones(len(positions))
                    )
                    for direction in ("plus", "minus")
                }
            del plus_mean, minus_mean, values
            if device.type == "cuda":
                torch.cuda.empty_cache()

        case_reports[str(case)] = {
            "anchor_rows": int(len(anchor_ids)),
            "matched_rows": int(common.sum()),
            "anchor_group_rows": {
                group: int(groups[group].sum()) for group in GROUPS
            },
            "matched_group_rows": {
                group: int(np.sum(common & groups[group])) for group in GROUPS
            },
            "model_rblend_coverage_rows": {
                label: int(
                    np.isfinite(
                        load_rblend_allow_missing(
                            rblend_tables[label], case, anchor_ids
                        )
                    ).sum()
                )
                for label in models
            },
        }
        print(
            f"CONSTGOLD_TRUTH_SUPPORT_DONE case={case} anchor={len(anchor_ids)} "
            f"matched={int(common.sum())} old_support="
            f"{int(np.sum(common & groups['inside_old_truth_support']))}",
            flush=True,
        )

    result = {
        "format_version": 1,
        "purpose": (
            "diagnostic bridge only: fixed measured-g0 anchor and fixed both-usable "
            "intersection partitioned by former truth support"
        ),
        "domain": {
            "anchor": (
                "measured g=0 detected, MAG_AUTO<25.8, strict "
                "FLUX_RADIUS>3.0 pixels (0.6 arcsec)"
            ),
            "truth_analysis_cut": None,
            "sheared_leg_magnitude_radius_recut": False,
            "matched_definition": (
                "same fixed-S0 identities with valid shape in both ConstGold legs"
            ),
            "former_truth_support_used_only_as_diagnostic_partition": {
                "r_input": "18 < r_input < 25.8",
                "Re_input_semimajor_arcsec": "0.37 < Re_input < 1.5",
            },
            "m_definition": "100*(R11_measured/R11_model-1)",
        },
        "cases": list(cases),
        "h": float(args.h),
        "draws": int(args.draws),
        "sampling_seed": int(args.sampling_seed),
        "common_antithetic_latents_across_legs_and_models": True,
        "models": {
            label: {
                "flow": str(path.resolve()),
                "flow_sha256": file_sha256(path),
                "rblend": str(rblend_paths[label].resolve()),
                "rblend_sha256": file_sha256(rblend_paths[label]),
            }
            for label, path in models.items()
        },
        "case_reports": case_reports,
        "per_case_sufficient": {
            "measured": measured,
            "predicted": predicted,
        },
        "summary": summarize(
            measured,
            predicted,
            unavailable,
            cases,
            h=args.h,
            replicates=args.n_boot,
            seed=args.bootstrap_seed,
        ),
        "limitations": [
            "The former truth cuts define diagnostic subgroups only; they are not reapplied to the fixed-g0 catalogue definition.",
            "The historical-stack bridge is interpretable only on the exact old-truth-support overlap where its R_blend lookup has coverage.",
            "One flow-training seed and one common latent-integration seed do not measure training or integration uncertainty.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, result)
    print(f"CONSTGOLD_TRUTH_SUPPORT_COMPLETE output={args.output}", flush=True)


if __name__ == "__main__":
    main()
