"""Transfer the frozen coherent-anchor bias emulator to V2.2 ConstGold.

The emulator predicts the anchor target ``R_blend_truth - R_blend_v22``.
This script adds that prediction to the frozen ConstGold V2.2 model and asks
whether it closes the already-defined total response gap.  ConstGold is used
only for evaluation: neither the emulator nor any threshold is refit here.

The evaluated population is exactly the one in
``v22_constgold_gap_features_c40-139.feather``: V2.2-supported pairs in the
rectangular true-property domain (r < 25.8, Re > 0.5 arcsec).  Rendered case is
the uncertainty unit.  Exact one-arcsec response shells are required so the
50-feature model is applied without imputation or feature substitution.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import stats

from scripts.build_anchor_bias_features import (
    FULL_FEATURES,
    SHELLS,
    TARGET,
    derive_features,
)


KEY = ["case", "input_index"]
PAIR_COLUMNS = [
    *KEY,
    "dominant_response",
    "dominant_distance",
    "dominant_abs_response",
    "primary_mag",
    "primary_size",
    "primary_sersic_n",
    "dominant_secondary_mag",
    "dominant_secondary_size",
    "dominant_secondary_sersic_n",
    "dominant_flux_ratio_primary",
    "dominant_flux_rank",
    "dominant_distance_rank",
    "R_blend",
    "R_abs_sum",
    "neighbour_flux_sum",
    "n_pairs",
    "closest_pair_distance",
    "mean_pair_distance",
    "std_pair_distance",
    "runner_up_abs_response",
    "n_response_pairs_ge_10pct_max",
    "top_abs_fraction",
    "dominant_to_runner_up_abs_response",
    "dominant_flux_share_neighbours",
    "dominant_size_ratio_primary",
    *[f"n_pair_{name}" for _, _, name in SHELLS],
    *[f"R_pair_{name}" for _, _, name in SHELLS],
    *[f"R_abs_pair_{name}" for _, _, name in SHELLS],
]
OUTCOME_COLUMNS = [
    *KEY,
    "primary_mag",
    "primary_size",
    "R_sim",
    "R_flow",
    "R_blend",
    "R_model",
    "gap",
]
MODEL_ORDER = ["primary_only", "physical_scene", "response_structure", "full"]
SEEDS = [501, 502, 503, 505, 506, 507, 508, 509,
         510, 511, 512, 513, 514, 515, 516, 517]


def finite_stat(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=float)
    if values.size < 2 or not np.isfinite(values).all():
        raise ValueError("need at least two finite case values")
    sd = float(values.std(ddof=1))
    mean = float(values.mean())
    if sd <= 10.0 * np.finfo(float).eps * max(1.0, abs(mean)):
        # A frozen constant correction legitimately has zero case scatter.
        # Avoid emitting scipy's +/-inf t statistic, which is not strict JSON.
        t_value = None
        p_value = 0.0 if mean != 0.0 else 1.0
    else:
        test = stats.ttest_1samp(values, popmean=0.0)
        t_value = float(test.statistic)
        p_value = float(test.pvalue)
    return {
        "mean": mean,
        "case_sd": sd,
        "case_sem": sd / np.sqrt(values.size),
        "n_cases": int(values.size),
        "t": t_value,
        "p": p_value,
    }


def jackknife_ratio(numerator: np.ndarray, denominator: np.ndarray) -> dict:
    numerator = np.asarray(numerator, dtype=float)
    denominator = np.asarray(denominator, dtype=float)
    if numerator.shape != denominator.shape or numerator.size < 3:
        raise ValueError("jackknife ratio needs matched vectors with >=3 cases")
    n = numerator.size
    total_num = float(numerator.sum())
    total_den = float(denominator.sum())
    if abs(total_den) < 1.0e-12:
        raise ZeroDivisionError("ratio denominator is numerically zero")
    value = total_num / total_den
    leave_one_out = (total_num - numerator) / (total_den - denominator)
    centre = float(leave_one_out.mean())
    sem = float(np.sqrt((n - 1.0) / n * np.sum((leave_one_out - centre) ** 2)))
    return {"value": float(value), "jackknife_case_sem": sem, "n_cases": int(n)}


def quantiles(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("prediction distribution is not finite")
    probabilities = [0.0, 0.001, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999, 1.0]
    result = np.quantile(values, probabilities)
    return {
        f"q{int(round(probability * 1000)):04d}": float(value)
        for probability, value in zip(probabilities, result)
    }


def seed_from_path(path: str) -> int:
    match = re.search(r"_s(\d+)\.feather$", os.path.basename(path))
    if match is None:
        raise RuntimeError(f"cannot parse flow seed from {path}")
    return int(match.group(1))


def packed_key(case: np.ndarray, input_index: np.ndarray) -> np.ndarray:
    case = np.asarray(case, dtype=np.int64)
    input_index = np.asarray(input_index, dtype=np.int64)
    if len(input_index) and (
        int(input_index.min()) < 0 or int(input_index.max()) >= (1 << 40)
    ):
        raise RuntimeError("input_index exceeds packed-key allocation")
    return (case << 40) | input_index


def seed_stat(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=float)
    if len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("need at least two finite seed values")
    sd = float(values.std(ddof=1))
    return {
        "mean": float(values.mean()),
        "seed_sd": sd,
        "seed_sem": sd / np.sqrt(len(values)),
        "n_seeds": int(len(values)),
    }


def score_m_by_flow_seed(dump_glob: str, frame: pd.DataFrame,
                         correction: np.ndarray) -> dict:
    """Compute absolute m with the required 16-seed flow ensemble."""
    paths = sorted(glob.glob(dump_glob), key=seed_from_path)
    seeds = [seed_from_path(path) for path in paths]
    if seeds != SEEDS:
        raise RuntimeError(f"expected flow seeds {SEEDS}, found {seeds}")
    correction = np.asarray(correction, dtype=float)
    if len(correction) != len(frame) or not np.isfinite(correction).all():
        raise ValueError("invalid per-object correction for seed scoring")
    target_case = frame.case.to_numpy(np.int64)
    target_index = frame.input_index.to_numpy(np.int64)
    target_key = packed_key(target_case, target_index)
    if len(np.unique(target_key)) != len(target_key):
        raise RuntimeError("duplicate target key in seed scoring")
    target_order = np.argsort(target_key, kind="mergesort")
    target_key_sorted = target_key[target_order]
    if np.any(np.diff(target_key_sorted) <= 0):
        raise RuntimeError("target key is not strictly ordered after sorting")
    reference_r_sim = frame.R_sim.to_numpy(float)[target_order]
    reference_r_blend = frame.R_blend.to_numpy(float)[target_order]
    correction_sorted = correction[target_order]
    correction_mean = float(correction_sorted.mean())

    positions = None
    raw_m = []
    corrected_m = []
    components = []
    for seed, path in zip(seeds, paths):
        table = pd.read_feather(
            path, columns=["case", "input_index", "r_sim", "R_flow", "R_blend"]
        )
        case = table.case.to_numpy(np.int64)
        index = table.input_index.to_numpy(np.int64)
        if positions is None:
            source_key = packed_key(case, index)
            if np.any(np.diff(source_key) <= 0):
                raise RuntimeError(f"dump key is not strictly ordered: {path}")
            positions = np.searchsorted(source_key, target_key_sorted)
            if positions[-1] >= len(source_key) or not np.array_equal(
                source_key[positions], target_key_sorted
            ):
                raise RuntimeError("flow dump does not cover exact ConstGold gap population")
        elif not np.array_equal(case[positions], target_case[target_order]) \
                or not np.array_equal(index[positions], target_index[target_order]):
            raise RuntimeError(f"flow dump target-key order differs: {path}")
        r_sim = table.r_sim.to_numpy(float)[positions]
        r_flow = table.R_flow.to_numpy(float)[positions]
        r_blend = table.R_blend.to_numpy(float)[positions]
        del table
        if not np.array_equal(r_sim, reference_r_sim):
            raise RuntimeError(f"R_sim differs from frozen gap table: {path}")
        if not np.array_equal(r_blend, reference_r_blend):
            raise RuntimeError(f"R_blend differs from frozen gap table: {path}")
        if not np.isfinite(r_flow).all():
            raise RuntimeError(f"non-finite R_flow on target population: {path}")
        means = [float(r_sim.mean()), float(r_flow.mean()), float(r_blend.mean())]
        model = means[1] + means[2]
        m_raw = 100.0 * (means[0] / model - 1.0)
        m_corrected = 100.0 * (
            means[0] / (model + correction_mean) - 1.0
        )
        raw_m.append(m_raw)
        corrected_m.append(m_corrected)
        components.append(means)
        print(
            f"seed {seed}: m raw={m_raw:+.4f}% "
            f"corrected={m_corrected:+.4f}%", flush=True,
        )
    raw_m = np.asarray(raw_m)
    corrected_m = np.asarray(corrected_m)
    paired = corrected_m - raw_m
    component_mean = np.asarray(components).mean(axis=0)
    return {
        "definition": "m = mean(R_sim) / mean(R_flow + R_blend [+ correction]) - 1, formed per flow seed",
        "n_rows": int(len(frame)),
        "seeds": seeds,
        "raw_m_percent_by_seed": raw_m.tolist(),
        "corrected_m_percent_by_seed": corrected_m.tolist(),
        "paired_change_percent_by_seed": paired.tolist(),
        "raw_m_percent": seed_stat(raw_m),
        "corrected_m_percent": seed_stat(corrected_m),
        "paired_change_percent": seed_stat(paired),
        "components": {
            "R_sim": float(component_mean[0]),
            "R_flow": float(component_mean[1]),
            "R_blend": float(component_mean[2]),
            "predicted_bias": correction_mean,
            "R_model_raw": float(component_mean[1] + component_mean[2]),
            "R_model_corrected": float(
                component_mean[1] + component_mean[2] + correction_mean
            ),
        },
    }


def prepare_emulator_features(pair_frame: pd.DataFrame) -> pd.DataFrame:
    """Map the exact ConstGold pair replay onto the anchor feature schema."""
    if missing := set(PAIR_COLUMNS) - set(pair_frame):
        raise KeyError(f"ConstGold pair table lacks {sorted(missing)}")
    if pair_frame.duplicated(KEY).any():
        raise RuntimeError("ConstGold pair table has duplicate keys")
    frame = pair_frame.copy()
    frame["primary_flux"] = np.power(
        10.0, -0.4 * frame.primary_mag.to_numpy(float)
    )
    frame["has_deployed_pair"] = np.int8(1)
    frame["min_pair_distance"] = frame.closest_pair_distance
    frame["response"] = frame.dominant_response
    frame["R_blend_lsst_r_extnbr_v22"] = frame.R_blend
    # derive_features forms the target as well; setting truth equal to the
    # frozen prediction makes that unused bookkeeping column exactly zero.
    frame["R_blend_truth"] = frame.R_blend
    frame["other_abs_response_sum"] = (
        frame.R_abs_sum - frame.dominant_abs_response
    )
    derived = derive_features(frame)
    output = derived[[*KEY, *FULL_FEATURES]].copy()
    if output[FULL_FEATURES].isna().any().any():
        raise RuntimeError("paired ConstGold row has a missing emulator feature")
    if not np.isfinite(output[FULL_FEATURES].to_numpy(float)).all():
        raise RuntimeError("paired ConstGold row has a non-finite emulator feature")
    return output


def block_summary(case_table: pd.DataFrame, selected: np.ndarray,
                  correction_columns: dict[str, str]) -> dict:
    local = case_table.loc[np.asarray(selected, dtype=bool)].copy()
    if len(local) < 2:
        raise RuntimeError("summary block has fewer than two cases")
    raw = local.raw_gap.to_numpy(float)
    raw_m = 100.0 * (
        local.R_sim.to_numpy(float) / local.R_model.to_numpy(float) - 1.0
    )
    result = {
        "case_window": [int(local.case.min()), int(local.case.max())],
        "n_cases": int(len(local)),
        "n_rows": int(local.n_rows.sum()),
        "raw_gap": finite_stat(raw),
        "raw_m_percent_case_distribution": finite_stat(raw_m),
        "components": {
            name: finite_stat(local[name].to_numpy(float))
            for name in ("R_sim", "R_flow", "R_blend", "R_model")
        },
        "corrections": {},
    }
    for name, column in correction_columns.items():
        correction = local[column].to_numpy(float)
        remaining = raw - correction
        corrected_m = 100.0 * (
            local.R_sim.to_numpy(float)
            / (local.R_model.to_numpy(float) + correction)
            - 1.0
        )
        closure = raw - correction - remaining
        if float(np.max(np.abs(closure))) > 1.0e-14:
            raise RuntimeError(f"{name} additive correction does not close")
        result["corrections"][name] = {
            "predicted_bias": finite_stat(correction),
            "remaining_gap": finite_stat(remaining),
            "corrected_m_percent_case_distribution": finite_stat(corrected_m),
            "paired_m_change_percent": finite_stat(corrected_m - raw_m),
            "recovered_fraction": jackknife_ratio(correction, raw),
            "remaining_fraction": jackknife_ratio(remaining, raw),
            "closes_within_2_case_sem": bool(
                abs(float(remaining.mean()))
                <= 2.0 * float(remaining.std(ddof=1) / np.sqrt(len(remaining)))
            ),
        }
    return result


def prediction_deciles(frame: pd.DataFrame, prediction: np.ndarray,
                       edges: np.ndarray) -> tuple[pd.DataFrame, dict]:
    prediction = np.asarray(prediction, dtype=float)
    if len(prediction) != len(frame):
        raise ValueError("prediction length mismatch")
    if len(edges) != 11 or not np.all(np.diff(edges) > 0):
        raise ValueError("expected ten strictly ordered frozen prediction bins")
    code = np.searchsorted(edges, prediction, side="right") - 1
    code = np.clip(code, 0, 9)
    work = frame[["case", "gap"]].copy()
    work["bin"] = code
    work["prediction"] = prediction
    work["corrected_gap"] = work.gap - work.prediction
    grouped = work.groupby(["case", "bin"], sort=True).agg(
        n_rows=("gap", "size"),
        raw_gap=("gap", "mean"),
        prediction=("prediction", "mean"),
        corrected_gap=("corrected_gap", "mean"),
    ).reset_index()
    rows = []
    for index in range(10):
        local = grouped.loc[grouped.bin == index]
        if local.case.nunique() != frame.case.nunique():
            raise RuntimeError(f"prediction bin {index} is not occupied in every case")
        raw = finite_stat(local.raw_gap.to_numpy(float))
        predicted = finite_stat(local.prediction.to_numpy(float))
        corrected = finite_stat(local.corrected_gap.to_numpy(float))
        rows.append({
            "bin": int(index),
            "lo": None if not np.isfinite(edges[index]) else float(edges[index]),
            "hi": None if not np.isfinite(edges[index + 1]) else float(edges[index + 1]),
            "n_rows": int(local.n_rows.sum()),
            "n_cases": int(local.case.nunique()),
            "raw_gap_mean": raw["mean"],
            "raw_gap_case_sem": raw["case_sem"],
            "predicted_bias_mean": predicted["mean"],
            "predicted_bias_case_sem": predicted["case_sem"],
            "corrected_gap_mean": corrected["mean"],
            "corrected_gap_case_sem": corrected["case_sem"],
        })
    table = pd.DataFrame(rows)
    low = grouped.loc[grouped.bin == 0, ["case", "raw_gap", "prediction"]]
    high = grouped.loc[grouped.bin == 9, ["case", "raw_gap", "prediction"]]
    paired = low.merge(high, on="case", suffixes=("_low", "_high"), validate="one_to_one")
    observed_span = paired.raw_gap_high.to_numpy(float) - paired.raw_gap_low.to_numpy(float)
    predicted_span = (
        paired.prediction_high.to_numpy(float)
        - paired.prediction_low.to_numpy(float)
    )
    raw_means = table.raw_gap_mean.to_numpy(float)
    prediction_means = table.predicted_bias_mean.to_numpy(float)
    slope, intercept = np.polyfit(prediction_means, raw_means, 1)
    equal_weight = np.full(len(table), 1.0 / len(table))
    row_weight = table.n_rows.to_numpy(float)
    row_weight /= row_weight.sum()

    def curve_error(weights: np.ndarray) -> dict:
        raw = table.raw_gap_mean.to_numpy(float)
        corrected = table.corrected_gap_mean.to_numpy(float)
        raw_rmse = float(np.sqrt(np.sum(weights * raw ** 2)))
        corrected_rmse = float(np.sqrt(np.sum(weights * corrected ** 2)))
        return {
            "raw_rmse": raw_rmse,
            "corrected_rmse": corrected_rmse,
            "corrected_over_raw_rmse": corrected_rmse / raw_rmse,
            "raw_mae": float(np.sum(weights * np.abs(raw))),
            "corrected_mae": float(np.sum(weights * np.abs(corrected))),
        }
    calibration = {
        "edges_source": "anchor development cases 400-699",
        "observed_high_minus_low": finite_stat(observed_span),
        "predicted_high_minus_low": finite_stat(predicted_span),
        "bin_mean_spearman": float(
            stats.spearmanr(prediction_means, raw_means).statistic
        ),
        "bin_mean_slope_total_gap_on_predicted_blend_bias": float(slope),
        "bin_mean_intercept": float(intercept),
        "observed_over_predicted_high_minus_low": float(
            observed_span.mean() / predicted_span.mean()
        ),
        "equal_decile_error": curve_error(equal_weight),
        "row_weighted_decile_error": curve_error(row_weight),
    }
    return table, calibration


def block_shift_summary(case_table: pd.DataFrame,
                        correction_column: str) -> dict:
    development = case_table.loc[case_table.case <= 89]
    validation = case_table.loc[case_table.case >= 90]

    def contrast(dev_values: np.ndarray, val_values: np.ndarray) -> dict:
        dev_values = np.asarray(dev_values, dtype=float)
        val_values = np.asarray(val_values, dtype=float)
        test = stats.ttest_ind(val_values, dev_values, equal_var=False)
        difference = float(val_values.mean() - dev_values.mean())
        sem = float(np.sqrt(
            dev_values.var(ddof=1) / len(dev_values)
            + val_values.var(ddof=1) / len(val_values)
        ))
        return {
            "validation_minus_development": difference,
            "independent_case_sem": sem,
            "welch_t": float(test.statistic),
            "welch_p": float(test.pvalue),
        }

    dev_raw = development.raw_gap.to_numpy(float)
    val_raw = validation.raw_gap.to_numpy(float)
    dev_prediction = development[correction_column].to_numpy(float)
    val_prediction = validation[correction_column].to_numpy(float)
    return {
        "raw_gap": contrast(dev_raw, val_raw),
        "predicted_bias": contrast(dev_prediction, val_prediction),
        "remaining_gap": contrast(
            dev_raw - dev_prediction, val_raw - val_prediction
        ),
    }


def support_table(anchor_development: pd.DataFrame,
                  constgold: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    any_outside = np.zeros(len(constgold), dtype=bool)
    rows = []
    for feature in FULL_FEATURES:
        anchor_values = anchor_development[feature].to_numpy(float)
        const_values = constgold[feature].to_numpy(float)
        anchor_finite = np.isfinite(anchor_values)
        const_finite = np.isfinite(const_values)
        if not anchor_finite.any() or not const_finite.any():
            raise RuntimeError(f"feature {feature} has no finite transfer support")
        lower = float(anchor_values[anchor_finite].min())
        upper = float(anchor_values[anchor_finite].max())
        below = const_finite & (const_values < lower)
        above = const_finite & (const_values > upper)
        outside = below | above
        any_outside |= outside
        rows.append({
            "feature": feature,
            "anchor_development_min": lower,
            "anchor_development_max": upper,
            "anchor_missing_fraction": float(np.mean(~anchor_finite)),
            "constgold_missing_fraction": float(np.mean(~const_finite)),
            "constgold_below_fraction": float(np.mean(below)),
            "constgold_above_fraction": float(np.mean(above)),
            "constgold_outside_fraction": float(np.mean(outside)),
        })
    table = pd.DataFrame(rows).sort_values(
        "constgold_outside_fraction", ascending=False, kind="mergesort"
    ).reset_index(drop=True)
    summary = {
        "definition": "outside the finite min/max observed in anchor development cases 400-699",
        "fraction_rows_outside_at_least_one_feature": float(any_outside.mean()),
        "maximum_single_feature_outside_fraction": float(
            table.constgold_outside_fraction.max()
        ),
        "top_features": table.head(10).to_dict(orient="records"),
    }
    return table, summary


def markdown_report(payload: dict) -> str:
    overall = payload["blocks"]["all_c40_139"]
    full = overall["corrections"]["full"]
    raw = overall["raw_gap"]
    correction = full["predicted_bias"]
    remaining = full["remaining_gap"]
    recovery = full["recovered_fraction"]
    global_verdict = "yes" if full["closes_within_2_case_sem"] else "no"
    conditional = payload["conditional_transfer"]
    row_rmse = conditional["row_weighted_decile_error"]
    raw_m = payload["seed_m"]["raw_m_percent"]
    corrected_m = payload["seed_m"]["corrected_m_percent"]
    lines = [
        "# Frozen anchor bias-emulator transfer to V2.2 ConstGold",
        "",
        "## Direct answer",
        "",
        f"**Pooled mean: {global_verdict}. Conditional recovery: no.** The full "
        "emulator numerically closes the global mean, but its anchor-learned conditional "
        "bias pattern does not transfer to ConstGold. This is not a validated correction.",
        "",
        f"Raw `R_sim - R_model` is `{raw['mean']:+.6f} ± {raw['case_sem']:.6f}`. "
        f"The frozen emulator predicts a correction of `{correction['mean']:+.6f} ± "
        f"{correction['case_sem']:.6f}`, leaving `{remaining['mean']:+.6f} ± "
        f"{remaining['case_sem']:.6f}`. It recovers `{100.0 * recovery['value']:.1f}% ± "
        f"{100.0 * recovery['jackknife_case_sem']:.1f}%` of the mean gap.",
        "",
        f"Using the required 16 flow seeds, mean `m` moves from "
        f"`{raw_m['mean']:+.3f}% ± {raw_m['seed_sem']:.3f}%` to "
        f"`{corrected_m['mean']:+.3f}% ± {corrected_m['seed_sem']:.3f}%`.",
        "",
        "The correction is added in response units: "
        "`R_model_corrected = R_flow + R_blend_v22 + predicted_bias`.",
        "",
        "## Stability across the two ConstGold case blocks",
        "",
        "| Cases | Rows | Raw gap | Predicted correction | Remaining gap | Recovered | Closed at 2 SEM? |",
        "|---|---:|---:|---:|---:|---:|:---:|",
    ]
    for block_name in ("development_c40_89", "validation_c90_139", "all_c40_139"):
        block = payload["blocks"][block_name]
        item = block["corrections"]["full"]
        lines.append(
            f"| {block['case_window'][0]}–{block['case_window'][1]} | "
            f"{block['n_rows']:,} | {block['raw_gap']['mean']:+.6f} ± "
            f"{block['raw_gap']['case_sem']:.6f} | "
            f"{item['predicted_bias']['mean']:+.6f} ± "
            f"{item['predicted_bias']['case_sem']:.6f} | "
            f"{item['remaining_gap']['mean']:+.6f} ± "
            f"{item['remaining_gap']['case_sem']:.6f} | "
            f"{100.0 * item['recovered_fraction']['value']:.1f}% | "
            f"{'yes' if item['closes_within_2_case_sem'] else 'no'} |"
        )
    lines.extend([
        "",
        "## Is recovery feature-driven?",
        "",
        "| Frozen correction | Predicted correction | Remaining gap | Recovered |",
        "|---|---:|---:|---:|",
    ])
    for name in ["anchor_development_constant", *MODEL_ORDER]:
        item = overall["corrections"][name]
        lines.append(
            f"| `{name}` | {item['predicted_bias']['mean']:+.6f} | "
            f"{item['remaining_gap']['mean']:+.6f} | "
            f"{100.0 * item['recovered_fraction']['value']:.1f}% |"
        )
    observed_span = conditional["observed_high_minus_low"]
    predicted_span = conditional["predicted_high_minus_low"]
    lines.extend([
        "",
        "## Conditional transfer",
        "",
        f"Across the ten prediction bins frozen from anchor development cases, the "
        f"ConstGold top-minus-bottom total residual is `{observed_span['mean']:+.6f} ± "
        f"{observed_span['case_sem']:.6f}`, while the emulator predicts "
        f"`{predicted_span['mean']:+.6f}`—about "
        f"`{1.0 / conditional['observed_over_predicted_high_minus_low']:.1f}x` too large. "
        f"Bin-mean Spearman rho is "
        f"`{conditional['bin_mean_spearman']:+.3f}` and the slope of total residual "
        f"on predicted blend bias is "
        f"`{conditional['bin_mean_slope_total_gap_on_predicted_blend_bias']:+.3f}`. "
        f"Across these bins, applying the correction worsens the row-weighted "
        f"conditional RMSE from `{row_rmse['raw_rmse']:.5f}` to "
        f"`{row_rmse['corrected_rmse']:.5f}` "
        f"(`{row_rmse['corrected_over_raw_rmse']:.2f}x`).",
        "",
        "A frozen constant equal to the anchor-development mean already recovers "
        f"`{100.0 * overall['corrections']['anchor_development_constant']['recovered_fraction']['value']:.1f}%` "
        "of the pooled gap. The full emulator's extra apparent improvement therefore "
        "does not demonstrate that its learned per-anchor structure is correct.",
        "",
        "## Interpretation guard",
        "",
        "- This is the same paired rectangular ConstGold population used by the earlier "
        "gap panels, not the full 11.67-million fiducial population.",
        "- The emulator predicts a blend-only anchor residual, whereas the observed "
        "ConstGold residual is total (`R_sim - R_flow - R_blend`). Global closure can "
        "therefore include cancellation with flow error or estimand differences.",
        "- The frozen emulator failed its anchor row-MSE gate. This transfer test is a "
        "diagnostic, not a deployed calibration, and nothing was refit on ConstGold.",
        f"- `{100.0 * payload['support']['fraction_rows_outside_at_least_one_feature']:.2f}%` "
        "of ConstGold rows lie outside the anchor-development min/max in at least one "
        "of the 50 coordinates; see the support CSV for the per-feature audit.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pair-features", required=True)
    ap.add_argument("--constgold-gap", required=True)
    ap.add_argument("--anchor-features", required=True)
    ap.add_argument("--bias-emulator", required=True)
    ap.add_argument("--bias-emulator-json", required=True)
    ap.add_argument("--dump-glob", required=True)
    ap.add_argument("--output-prefix", required=True)
    args = ap.parse_args()
    outputs = {
        "json": args.output_prefix + ".json",
        "markdown": args.output_prefix + ".md",
        "cases_csv": args.output_prefix + "_cases.csv",
        "deciles_csv": args.output_prefix + "_deciles.csv",
        "support_csv": args.output_prefix + "_support.csv",
    }
    for path in [args.pair_features, args.constgold_gap, args.anchor_features,
                 args.bias_emulator, args.bias_emulator_json]:
        if not os.path.isfile(path) or os.path.getsize(path) == 0:
            raise FileNotFoundError(path)
    for path in outputs.values():
        if os.path.exists(path):
            raise FileExistsError(f"refusing existing output {path}")
    Path(args.output_prefix).parent.mkdir(parents=True, exist_ok=True)

    artifact = joblib.load(args.bias_emulator)
    with open(args.bias_emulator_json, encoding="utf-8") as handle:
        emulator_report = json.load(handle)
    if artifact.get("target") != TARGET:
        raise RuntimeError("unexpected bias-emulator target")
    if artifact.get("constgold_opened") is not False:
        raise RuntimeError("bias-emulator provenance does not certify a frozen transfer")
    if set(MODEL_ORDER) - set(artifact.get("models", {})):
        raise RuntimeError("bias-emulator artifact lacks a predeclared model variant")
    for name in MODEL_ORDER:
        if artifact["feature_sets"][name] != emulator_report["feature_sets"][name]:
            raise RuntimeError(f"feature schema mismatch for {name}")

    print("read exact ConstGold pair features", flush=True)
    pair = pd.read_feather(args.pair_features, columns=PAIR_COLUMNS)
    const_features = prepare_emulator_features(pair)
    del pair
    print(f"prepared {len(const_features):,} pair-supported feature rows", flush=True)

    outcome = pd.read_feather(args.constgold_gap, columns=OUTCOME_COLUMNS)
    if outcome.duplicated(KEY).any():
        raise RuntimeError("ConstGold outcome table has duplicate keys")
    outcome = outcome.rename(columns={
        "primary_mag": "outcome_primary_mag",
        "primary_size": "outcome_primary_size",
    })
    frame = outcome.merge(const_features, on=KEY, how="left", validate="one_to_one")
    del outcome, const_features
    if frame[FULL_FEATURES].isna().any().any():
        raise RuntimeError("ConstGold outcome population is not covered by pair features")
    primary_mag_delta = (
        frame.outcome_primary_mag.to_numpy(float)
        - frame.primary_mag.to_numpy(float)
    )
    if float(np.max(np.abs(primary_mag_delta))) > 1.0e-6:
        raise RuntimeError("ConstGold primary magnitude mismatch")
    primary_size_delta = (
        frame.outcome_primary_size.to_numpy(float)
        - np.power(10.0, frame.log10_primary_size.to_numpy(float))
    )
    if float(np.max(np.abs(primary_size_delta))) > 1.0e-6:
        raise RuntimeError("ConstGold primary size mismatch")
    if not np.allclose(
        frame.R_blend.to_numpy(float), frame.scene_prediction.to_numpy(float),
        rtol=0.0, atol=1.0e-6,
    ):
        raise RuntimeError("ConstGold pair replay differs from frozen R_blend")
    gap_closure = (
        frame.R_sim - frame.R_flow - frame.R_blend - frame.gap
    ).to_numpy(float)
    if float(np.max(np.abs(gap_closure))) > 2.0e-14:
        raise RuntimeError("ConstGold raw gap does not close")
    if sorted(frame.case.unique().tolist()) != list(range(40, 140)):
        raise RuntimeError("expected ConstGold cases 40-139")
    signed_shell_closure = (
        sum(frame[f"R_pair_{name}"] for _, _, name in SHELLS)
        - frame.scene_prediction
    ).to_numpy(float)
    absolute_shell_closure = (
        sum(frame[f"R_abs_pair_{name}"] for _, _, name in SHELLS)
        - frame.R_abs_sum
    ).to_numpy(float)
    if float(np.max(np.abs(signed_shell_closure))) > 1.0e-6:
        raise RuntimeError("ConstGold signed shell features do not close")
    if float(np.max(np.abs(absolute_shell_closure))) > 1.0e-6:
        raise RuntimeError("ConstGold absolute shell features do not close")
    print(f"joined exact prior gap population: {len(frame):,} rows", flush=True)

    anchor = pd.read_feather(
        args.anchor_features, columns=["case", TARGET, *FULL_FEATURES]
    )
    anchor_development = anchor.loc[anchor.case <= 699].copy()
    anchor_test = anchor.loc[anchor.case >= 700].copy()
    if sorted(anchor_development.case.unique().tolist()) != list(range(400, 700)):
        raise RuntimeError("anchor development cases differ from frozen artifact")
    if sorted(anchor_test.case.unique().tolist()) != list(range(700, 900)):
        raise RuntimeError("anchor test cases differ from frozen artifact")

    prediction_columns = {}
    anchor_full_development_prediction = None
    anchor_full_test_prediction = None
    for name in MODEL_ORDER:
        features = artifact["feature_sets"][name]
        model = artifact["models"][name]
        print(f"score frozen {name} model ({len(features)} features)", flush=True)
        const_prediction = model.predict(
            frame[features].to_numpy(np.float32)
        ).astype(float)
        column = f"predicted_bias_{name}"
        frame[column] = const_prediction
        prediction_columns[name] = column
        if name == "full":
            anchor_full_development_prediction = model.predict(
                anchor_development[features].to_numpy(np.float32)
            ).astype(float)
            anchor_full_test_prediction = model.predict(
                anchor_test[features].to_numpy(np.float32)
            ).astype(float)
    assert anchor_full_development_prediction is not None
    assert anchor_full_test_prediction is not None

    anchor_dev_prediction_by_case = pd.DataFrame({
        "case": anchor_development.case.to_numpy(np.int64),
        "prediction": anchor_full_development_prediction,
        "target": anchor_development[TARGET].to_numpy(float),
    }).groupby("case", sort=True).mean()
    anchor_test_prediction_by_case = pd.DataFrame({
        "case": anchor_test.case.to_numpy(np.int64),
        "prediction": anchor_full_test_prediction,
        "target": anchor_test[TARGET].to_numpy(float),
    }).groupby("case", sort=True).mean()
    anchor_constant = float(anchor_dev_prediction_by_case.target.mean())
    frame["predicted_bias_anchor_development_constant"] = anchor_constant
    correction_columns = {
        "anchor_development_constant": "predicted_bias_anchor_development_constant",
        **prediction_columns,
    }

    aggregation = {
        "n_rows": ("gap", "size"),
        "R_sim": ("R_sim", "mean"),
        "R_flow": ("R_flow", "mean"),
        "R_blend": ("R_blend", "mean"),
        "R_model": ("R_model", "mean"),
        "raw_gap": ("gap", "mean"),
    }
    for name, column in correction_columns.items():
        aggregation[column] = (column, "mean")
    cases = frame.groupby("case", sort=True).agg(**aggregation).reset_index()
    blocks = {
        "development_c40_89": block_summary(
            cases, cases.case <= 89, correction_columns,
        ),
        "validation_c90_139": block_summary(
            cases, cases.case >= 90, correction_columns,
        ),
        "all_c40_139": block_summary(
            cases, np.ones(len(cases), dtype=bool), correction_columns,
        ),
    }

    anchor_edges = np.r_[
        -np.inf,
        np.quantile(anchor_full_development_prediction, np.arange(1, 10) / 10.0),
        np.inf,
    ]
    if len(np.unique(anchor_edges)) != 11:
        raise RuntimeError("anchor prediction decile edges are not unique")
    deciles, conditional = prediction_deciles(
        frame, frame[prediction_columns["full"]].to_numpy(float), anchor_edges,
    )
    support, support_summary = support_table(anchor_development, frame)
    seed_m = score_m_by_flow_seed(
        args.dump_glob, frame,
        frame[prediction_columns["full"]].to_numpy(float),
    )

    expected_anchor_test_prediction = emulator_report["test_metrics"]["full"][
        "global_prediction"
    ]["mean"]
    reproduced_anchor_test_prediction = float(
        anchor_test_prediction_by_case.prediction.mean()
    )
    if not np.isclose(
        reproduced_anchor_test_prediction, expected_anchor_test_prediction,
        rtol=0.0, atol=1.0e-12,
    ):
        raise RuntimeError("loaded model does not reproduce its anchor test prediction")

    cases.to_csv(outputs["cases_csv"], index=False)
    deciles.to_csv(outputs["deciles_csv"], index=False)
    support.to_csv(outputs["support_csv"], index=False)
    payload = {
        "title": "Frozen coherent-anchor bias-emulator transfer to V2.2 ConstGold",
        "design": (
            "models and all feature transformations frozen on coherent-anchor c400-699; "
            "ConstGold c40-139 evaluation only; rendered case is uncertainty unit"
        ),
        "population": {
            "label": "same paired rectangular population as prior V2.2 ConstGold gap panels",
            "true_primary_mag_max": 25.8,
            "true_primary_size_min_arcsec": 0.5,
            "requires_deployed_v22_pair": True,
            "n_rows": int(len(frame)),
            "n_cases": int(frame.case.nunique()),
        },
        "correction_definition": (
            "R_model_corrected = R_flow + R_blend_v22 + anchor_bias_emulator(features)"
        ),
        "emulator_target": artifact["target"],
        "emulator_target_sign": artifact["target_sign"],
        "emulator_model_good_on_anchor_gate": bool(emulator_report["model_good"]),
        "models_refit_on_constgold": False,
        "constgold_used_for_feature_or_capacity_selection": False,
        "blocks": blocks,
        "anchor_reference": {
            "development_cases": [400, 699],
            "test_cases": [700, 899],
            "development_target_case_mean": finite_stat(
                anchor_dev_prediction_by_case.target.to_numpy(float)
            ),
            "development_full_prediction_case_mean": finite_stat(
                anchor_dev_prediction_by_case.prediction.to_numpy(float)
            ),
            "test_target_case_mean": finite_stat(
                anchor_test_prediction_by_case.target.to_numpy(float)
            ),
            "test_full_prediction_case_mean": finite_stat(
                anchor_test_prediction_by_case.prediction.to_numpy(float)
            ),
            "test_prediction_reproduction_abs_difference": abs(
                reproduced_anchor_test_prediction - expected_anchor_test_prediction
            ),
        },
        "prediction_distributions": {
            "anchor_development_full": quantiles(
                anchor_full_development_prediction
            ),
            "constgold_full": quantiles(
                frame[prediction_columns["full"]].to_numpy(float)
            ),
        },
        "conditional_transfer": conditional,
        "seed_m": seed_m,
        "full_model_block_shift": block_shift_summary(
            cases, prediction_columns["full"]
        ),
        "support": support_summary,
        "quality": {
            "duplicate_constgold_keys": int(frame.duplicated(KEY).sum()),
            "raw_gap_closure_max_abs": float(np.max(np.abs(gap_closure))),
            "primary_mag_join_max_abs": float(
                np.max(np.abs(primary_mag_delta))
            ),
            "primary_size_join_max_abs": float(
                np.max(np.abs(primary_size_delta))
            ),
            "pair_replay_rblend_max_abs": float(np.max(np.abs(
                frame.R_blend.to_numpy(float)
                - frame.scene_prediction.to_numpy(float)
            ))),
            "signed_shell_closure_max_abs": float(
                np.max(np.abs(signed_shell_closure))
            ),
            "absolute_shell_closure_max_abs": float(
                np.max(np.abs(absolute_shell_closure))
            ),
            "full_feature_missing_cells": int(frame[FULL_FEATURES].isna().sum().sum()),
            "full_feature_infinite_cells": int(np.isinf(
                frame[FULL_FEATURES].to_numpy(float)
            ).sum()),
        },
        "artifacts": {
            "source_pair_features": os.path.abspath(args.pair_features),
            "source_constgold_gap": os.path.abspath(args.constgold_gap),
            "source_anchor_features": os.path.abspath(args.anchor_features),
            "source_bias_emulator": os.path.abspath(args.bias_emulator),
            **{name: os.path.abspath(path) for name, path in outputs.items()},
        },
        "interpretation": (
            "diagnostic transfer only: emulator target is blend-only anchor residual, "
            "whereas ConstGold outcome is total response residual including flow"
        ),
    }
    report = markdown_report(payload)
    with open(outputs["json"], "x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    with open(outputs["markdown"], "x", encoding="utf-8") as handle:
        handle.write(report)
        handle.write("\n")
    print(report, flush=True)
    print("ANCHOR_BIAS_EMULATOR_CONSTGOLD_TRANSFER_DONE", flush=True)


if __name__ == "__main__":
    main()
