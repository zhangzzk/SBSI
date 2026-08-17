"""Test physical explanations inside the frozen V2.2 dominant-pair carrier.

The response-dominance carrier is a model-output description, not a physical
cause.  This script asks whether four latent pair coordinates explain residual
variation *inside* that carrier:

* secondary/primary flux ratio;
* secondary/primary effective-radius ratio;
* geometric overlap scale, (R_p + R_s) / distance;
* secondary/primary mean-surface-brightness ratio, proportional to
  (F_s/F_p) / (R_s/R_p)^2.

The primary carrier is the previously frozen
``dominant_to_runner_up_abs_response > 5 and dominant_response >= 0`` rule.
``top_abs_fraction > 0.7 and dominant_response >= 0`` is a fixed sensitivity
rule.  No outcome is used to choose a physical cut: development blocks supply
only covariate quantiles and scales.  Those definitions are applied unchanged
to validation cases.

Within each validation case, physical slopes and upper-minus-lower-quartile
contrasts are formed after blocking on a quantile grid of predicted amplitude
and dominance strength.  The case is the uncertainty unit.  Positive
deficit and positive contrast both mean that V2.2 underpredicts more at larger
values of the physical coordinate.

Anchor truth is direct coherent-shear truth.  The confirmatory constgold target
is the same-object neighbour proxy ``R_sim - r_sim_self - R_blend``.  The
constgold full residual and self/flow term are included as diagnostic targets,
not as BlendEMU labels.
"""
from __future__ import annotations

import argparse
import json
import os
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


KEY = ["case", "input_index"]
CONTROL_COLUMNS = [
    "log10_dominant_abs_response",
    "log10_dominance_strength",
]
AMPLITUDE_CONTROLS = {
    "dominant_pair": "log10_dominant_abs_response",
    "scene": "abs_scene_prediction",
}
PHYSICAL_FEATURES = [
    "log10_flux_ratio",
    "log10_size_ratio",
    "log10_overlap_scale",
    "log10_surface_brightness_ratio",
]
JOINT_BASIS = [
    "log10_flux_ratio",
    "log10_size_ratio",
    "log10_overlap_scale",
]
FEATURE_LABELS = {
    "log10_flux_ratio": "log10(F_s/F_p)",
    "log10_size_ratio": "log10(R_s/R_p)",
    "log10_overlap_scale": "log10[(R_p+R_s)/d]",
    "log10_surface_brightness_ratio": "log10(SB_s/SB_p)",
}
CARRIERS = {
    "pilot_frozen_ratio_gt5_positive": {
        "description": (
            "dominant_to_runner_up_abs_response > 5 AND "
            "dominant_response >= 0"
        ),
        "kind": "ratio",
        "threshold": 5.0,
    },
    "shared_top_fraction_gt0p7_positive": {
        "description": "top_abs_fraction > 0.7 AND dominant_response >= 0",
        "kind": "top_fraction",
        "threshold": 0.7,
    },
}


def finite_stat(values: Iterable[float]) -> dict:
    values = np.asarray(list(values), dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 2:
        raise ValueError("need at least two finite case values")
    sd = float(values.std(ddof=1))
    test = stats.ttest_1samp(values, 0.0)
    return {
        "mean": float(values.mean()),
        "case_sd": sd,
        "case_sem": sd / np.sqrt(values.size),
        "n_cases": int(values.size),
        "t": float(test.statistic),
        "p_raw": float(test.pvalue),
    }


def holm_adjust(pvalues: dict[str, float]) -> dict[str, float]:
    """Holm-adjust a named family of finite p-values."""
    if not pvalues:
        return {}
    if any(not np.isfinite(value) or value < 0 or value > 1
           for value in pvalues.values()):
        raise ValueError("Holm inputs must be finite p-values in [0, 1]")
    ordered = sorted(pvalues, key=pvalues.get)
    adjusted: dict[str, float] = {}
    running = 0.0
    size = len(ordered)
    for rank, name in enumerate(ordered):
        candidate = min(1.0, (size - rank) * pvalues[name])
        running = max(running, candidate)
        adjusted[name] = running
    return {name: float(adjusted[name]) for name in pvalues}


def derive_physical_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add positive-domain log physical coordinates and control variables."""
    frame = frame.copy()
    positive = {
        "dominant_flux_ratio_primary": frame.dominant_flux_ratio_primary,
        "primary_size": frame.primary_size,
        "dominant_secondary_size": frame.dominant_secondary_size,
        "dominant_distance": frame.dominant_distance,
        "dominant_abs_response": frame.dominant_abs_response,
        "dominant_to_runner_up_abs_response": (
            frame.dominant_to_runner_up_abs_response
        ),
    }
    bad = {
        name: int((~np.isfinite(values.to_numpy(float)) | (values <= 0)).sum())
        for name, values in positive.items()
    }
    if any(bad.values()):
        raise RuntimeError(f"non-positive/non-finite physical inputs: {bad}")

    frame["log10_flux_ratio"] = np.log10(
        frame.dominant_flux_ratio_primary.to_numpy(float)
    )
    frame["log10_size_ratio"] = np.log10(
        frame.dominant_secondary_size.to_numpy(float)
        / frame.primary_size.to_numpy(float)
    )
    frame["log10_overlap_scale"] = np.log10(
        (
            frame.primary_size.to_numpy(float)
            + frame.dominant_secondary_size.to_numpy(float)
        ) / frame.dominant_distance.to_numpy(float)
    )
    frame["log10_surface_brightness_ratio"] = (
        frame.log10_flux_ratio.to_numpy(float)
        - 2.0 * frame.log10_size_ratio.to_numpy(float)
    )
    frame["log10_dominant_abs_response"] = np.log10(
        frame.dominant_abs_response.to_numpy(float)
    )
    frame["log10_dominance_strength"] = np.log10(
        frame.dominant_to_runner_up_abs_response.to_numpy(float)
    )
    if "prediction" in frame:
        frame["abs_scene_prediction"] = np.abs(
            frame.prediction.to_numpy(float)
        )
    required = [*PHYSICAL_FEATURES, *CONTROL_COLUMNS]
    if not np.isfinite(frame[required].to_numpy(float)).all():
        raise RuntimeError("derived non-finite physical coordinate")
    return frame


def carrier_mask(frame: pd.DataFrame, carrier_name: str) -> np.ndarray:
    spec = CARRIERS[carrier_name]
    positive = frame.dominant_response.to_numpy(float) >= 0.0
    if spec["kind"] == "ratio":
        return positive & (
            frame.dominant_to_runner_up_abs_response.to_numpy(float)
            > float(spec["threshold"])
        )
    if spec["kind"] == "top_fraction":
        return positive & (
            frame.top_abs_fraction.to_numpy(float) > float(spec["threshold"])
        )
    raise ValueError(f"unknown carrier kind {spec['kind']}")


def quantile_edges(values: np.ndarray, n_bins: int) -> list[float]:
    values = np.asarray(values, float)
    edges = np.quantile(values, np.arange(1, n_bins) / n_bins)
    if len(np.unique(edges)) != len(edges):
        raise RuntimeError("development quantile edges are not unique")
    return [float(value) for value in edges]


def assign_control_cells(frame: pd.DataFrame, design: dict) -> np.ndarray:
    control_columns = design.get("control_columns", CONTROL_COLUMNS)
    first = np.digitize(
        frame[control_columns[0]].to_numpy(float),
        np.asarray(design["control_edges"][control_columns[0]], float),
    )
    second = np.digitize(
        frame[control_columns[1]].to_numpy(float),
        np.asarray(design["control_edges"][control_columns[1]], float),
    )
    n_bins = int(design["n_control_bins"])
    return first * n_bins + second


def learn_covariate_design(frame: pd.DataFrame, n_control_bins: int,
                           amplitude_control: str = "dominant_pair") -> dict:
    """Freeze scales/cuts using development covariates and no outcome."""
    if len(frame) < 1000 or frame.case.nunique() < 10:
        raise ValueError("development carrier population is unexpectedly small")
    amplitude_column = AMPLITUDE_CONTROLS[amplitude_control]
    control_columns = [amplitude_column, "log10_dominance_strength"]
    if not np.isfinite(frame[control_columns].to_numpy(float)).all():
        raise RuntimeError("non-finite control coordinate")
    design = {
        "n_control_bins": int(n_control_bins),
        "amplitude_control": amplitude_control,
        "control_columns": control_columns,
        "control_edges": {
            column: quantile_edges(frame[column].to_numpy(float), n_control_bins)
            for column in control_columns
        },
        "feature_moments": {},
        "feature_quartile_edges": {},
        "cell_feature_tails": {},
        "cell_weights": {},
        "n_development_rows": int(len(frame)),
        "n_development_cases": int(frame.case.nunique()),
        "outcome_used_to_define_design": False,
    }
    for feature in PHYSICAL_FEATURES:
        values = frame[feature].to_numpy(float)
        sd = float(values.std(ddof=1))
        if not np.isfinite(sd) or sd <= 0:
            raise RuntimeError(f"invalid development scale for {feature}")
        design["feature_moments"][feature] = {
            "mean": float(values.mean()), "sd": sd,
        }
        design["feature_quartile_edges"][feature] = quantile_edges(values, 4)

    cells = assign_control_cells(frame, design)
    counts = pd.Series(cells).value_counts().sort_index()
    total = float(counts.sum())
    for cell, count in counts.items():
        local = frame.loc[cells == int(cell)]
        if len(local) < 100:
            continue
        label = str(int(cell))
        design["cell_weights"][label] = float(count / total)
        design["cell_feature_tails"][label] = {
            feature: {
                "q25": float(local[feature].quantile(0.25)),
                "q75": float(local[feature].quantile(0.75)),
            }
            for feature in PHYSICAL_FEATURES
        }
    weight_sum = float(sum(design["cell_weights"].values()))
    if len(design["cell_weights"]) < n_control_bins ** 2 // 2:
        raise RuntimeError("too few populated development control cells")
    if weight_sum < 0.99:
        raise RuntimeError(f"development control cells retain only {weight_sum:.3%}")
    design["retained_cell_weight"] = weight_sum
    return design


def case_profile(frame: pd.DataFrame, target: str, feature: str,
                 design: dict) -> dict:
    edges = np.asarray(design["feature_quartile_edges"][feature], float)
    bins = np.digitize(frame[feature].to_numpy(float), edges)
    work = frame[["case", target]].copy()
    work["quartile"] = bins
    entries = []
    for quartile in range(4):
        local = work.loc[work.quartile == quartile]
        by_case = local.groupby("case", sort=True)[target].mean()
        entries.append({
            "quartile": quartile + 1,
            "n_rows": int(len(local)),
            "fraction": float(len(local) / len(work)),
            "conditional_deficit": finite_stat(by_case.to_numpy(float)),
        })
    low = work.loc[work.quartile == 0].groupby("case", sort=True)[target].mean()
    high = work.loc[work.quartile == 3].groupby("case", sort=True)[target].mean()
    paired = low.to_frame("low").join(high.to_frame("high"), how="inner")
    return {
        "development_edges": [float(value) for value in edges],
        "quartiles": entries,
        "upper_minus_lower": finite_stat(
            (paired.high - paired.low).to_numpy(float)
        ),
    }


def controlled_tail_contrast(frame: pd.DataFrame, target: str, feature: str,
                             design: dict, min_group_rows: int = 3,
                             min_weight_coverage: float = 0.8) -> dict:
    """Fixed-cell, fixed-weight upper-minus-lower physical contrast by case."""
    work = frame[["case", target, feature]].copy()
    work["cell"] = assign_control_cells(frame, design)
    weights = {int(key): float(value)
               for key, value in design["cell_weights"].items()}
    thresholds = {int(key): value
                  for key, value in design["cell_feature_tails"].items()}
    records = []
    for case, local_case in work.groupby("case", sort=True):
        high_sum = 0.0
        low_sum = 0.0
        used_weight = 0.0
        high_rows = 0
        low_rows = 0
        for cell, weight in weights.items():
            local = local_case.loc[local_case.cell == cell]
            if local.empty:
                continue
            q = thresholds[cell][feature]
            low = local.loc[local[feature] <= q["q25"], target]
            high = local.loc[local[feature] >= q["q75"], target]
            if len(low) < min_group_rows or len(high) < min_group_rows:
                continue
            low_sum += weight * float(low.mean())
            high_sum += weight * float(high.mean())
            used_weight += weight
            high_rows += len(high)
            low_rows += len(low)
        if used_weight >= min_weight_coverage:
            high_mean = high_sum / used_weight
            low_mean = low_sum / used_weight
            records.append({
                "case": int(case), "high": high_mean, "low": low_mean,
                "contrast": high_mean - low_mean,
                "weight_coverage": used_weight,
                "n_high": high_rows, "n_low": low_rows,
            })
    result = pd.DataFrame(records)
    if len(result) < 10:
        raise RuntimeError(
            f"only {len(result)} cases support controlled contrast for {feature}"
        )
    return {
        "high_conditional_deficit": finite_stat(result.high.to_numpy(float)),
        "low_conditional_deficit": finite_stat(result.low.to_numpy(float)),
        "high_minus_low": finite_stat(result.contrast.to_numpy(float)),
        "weight_coverage": {
            "mean": float(result.weight_coverage.mean()),
            "minimum": float(result.weight_coverage.min()),
        },
        "mean_rows_per_case": {
            "high": float(result.n_high.mean()),
            "low": float(result.n_low.mean()),
        },
        "n_validation_cases_total": int(frame.case.nunique()),
    }


def _within_cell_demean(values: np.ndarray, cells: np.ndarray) -> np.ndarray:
    values = np.asarray(values, float)
    cells = np.asarray(cells, np.int64)
    out = values.copy()
    for cell in np.unique(cells):
        local = cells == cell
        out[local] -= out[local].mean(axis=0)
    return out


def casewise_slopes(frame: pd.DataFrame, target: str, features: list[str],
                    design: dict) -> dict:
    """Average one within-control-cell OLS slope vector per validation case."""
    cells = assign_control_cells(frame, design)
    means = np.asarray([
        design["feature_moments"][feature]["mean"] for feature in features
    ], float)
    scales = np.asarray([
        design["feature_moments"][feature]["sd"] for feature in features
    ], float)
    x_all = (frame[features].to_numpy(float) - means) / scales
    y_all = frame[target].to_numpy(float)
    cases = frame.case.to_numpy(np.int64)
    rows = []
    for case in np.unique(cases):
        local = cases == case
        x = _within_cell_demean(x_all[local], cells[local])
        y = _within_cell_demean(y_all[local, None], cells[local])[:, 0]
        active = np.linalg.norm(x, axis=1) > 1.0e-12
        x = x[active]
        y = y[active]
        if len(y) <= len(features) + 2:
            continue
        beta, _, rank, singular = np.linalg.lstsq(x, y, rcond=None)
        if rank != len(features) or not np.isfinite(beta).all():
            continue
        rows.append({
            "case": int(case),
            **{feature: float(value) for feature, value in zip(features, beta)},
            "condition_number": float(singular[0] / singular[-1]),
        })
    slopes = pd.DataFrame(rows)
    if len(slopes) < 10:
        raise RuntimeError(
            f"only {len(slopes)} cases have full-rank slopes for {features}"
        )
    result = {
        "features": {
            feature: finite_stat(slopes[feature].to_numpy(float))
            for feature in features
        },
        "n_validation_cases_total": int(frame.case.nunique()),
        "n_cases_fit": int(len(slopes)),
        "case_condition_number": {
            "median": float(slopes.condition_number.median()),
            "q95": float(slopes.condition_number.quantile(0.95)),
            "maximum": float(slopes.condition_number.max()),
        },
        "coefficient_unit": "deficit per one development-population SD",
    }
    adjusted = holm_adjust({
        feature: item["p_raw"] for feature, item in result["features"].items()
    })
    for feature, value in adjusted.items():
        result["features"][feature]["p_holm_within_model"] = value
    return result


def summarize_carrier(frame: pd.DataFrame, target: str,
                      selected: np.ndarray) -> dict:
    work = frame[["case", target]].copy()
    work["selected"] = np.asarray(selected, bool)
    selected_case = work.loc[work.selected].groupby("case", sort=True)[target].mean()
    outside_case = work.loc[~work.selected].groupby("case", sort=True)[target].mean()
    fractions = work.groupby("case", sort=True).selected.mean()
    paired = selected_case.to_frame("selected").join(
        outside_case.to_frame("outside"), how="inner"
    )
    return {
        "selected_fraction": finite_stat(fractions.to_numpy(float)),
        "selected_conditional_deficit": finite_stat(selected_case.to_numpy(float)),
        "outside_conditional_deficit": finite_stat(outside_case.to_numpy(float)),
        "selected_minus_outside": finite_stat(
            (paired.selected - paired.outside).to_numpy(float)
        ),
        "n_selected_rows": int(np.asarray(selected, bool).sum()),
        "n_outside_rows": int((~np.asarray(selected, bool)).sum()),
    }


def analyze_validation(frame: pd.DataFrame, target: str, design: dict) -> dict:
    feature_results = {}
    for feature in PHYSICAL_FEATURES:
        feature_results[feature] = {
            "label": FEATURE_LABELS[feature],
            "marginal_profile": case_profile(frame, target, feature, design),
            "controlled_upper_minus_lower": controlled_tail_contrast(
                frame, target, feature, design
            ),
            "controlled_univariate_slope": casewise_slopes(
                frame, target, [feature], design
            )["features"][feature],
        }

    contrast_p = {
        feature: item["controlled_upper_minus_lower"]["high_minus_low"]["p_raw"]
        for feature, item in feature_results.items()
    }
    slope_p = {
        feature: item["controlled_univariate_slope"]["p_raw"]
        for feature, item in feature_results.items()
    }
    contrast_holm = holm_adjust(contrast_p)
    slope_holm = holm_adjust(slope_p)
    for feature in PHYSICAL_FEATURES:
        feature_results[feature]["controlled_upper_minus_lower"][
            "high_minus_low"
        ]["p_holm_four_features"] = contrast_holm[feature]
        feature_results[feature]["controlled_univariate_slope"][
            "p_holm_four_features"
        ] = slope_holm[feature]

    return {
        "target": target,
        "n_rows": int(len(frame)),
        "n_cases": int(frame.case.nunique()),
        "features": feature_results,
        "joint_physical_basis": casewise_slopes(
            frame, target, JOINT_BASIS, design
        ),
        "surface_brightness_independence_note": (
            "log10_surface_brightness_ratio = log10_flux_ratio - "
            "2*log10_size_ratio, so it is excluded from the joint basis to "
            "avoid exact collinearity"
        ),
    }


def load_anchor(response_paths: list[str], dominance_paths: list[str]) -> pd.DataFrame:
    response_columns = [
        *KEY, "R_blend_truth", "R_blend_lsst_r_extnbr_v22",
    ]
    dominance_columns = [
        "case", "anchor_index", "response", "dominant_abs_response",
        "dominant_distance", "r_input_p", "Re_input_p",
        "dominant_secondary_size", "dominant_flux_ratio_primary",
        "top_abs_fraction", "dominant_to_runner_up_abs_response",
    ]
    responses = pd.concat([
        pd.read_feather(path, columns=response_columns)
        for path in response_paths
    ], ignore_index=True)
    dominance = pd.concat([
        pd.read_feather(path, columns=dominance_columns)
        for path in dominance_paths
    ], ignore_index=True).rename(columns={
        "anchor_index": "input_index",
        "response": "dominant_response",
        "r_input_p": "primary_mag",
        "Re_input_p": "primary_size",
    })
    if responses.duplicated(KEY).any() or dominance.duplicated(KEY).any():
        raise RuntimeError("duplicate anchor key")
    frame = responses.merge(dominance, on=KEY, how="inner", validate="one_to_one")
    if len(frame) / len(responses) < 0.90:
        raise RuntimeError("anchor dominance join coverage below 90%")
    frame["deficit"] = (
        frame.R_blend_truth - frame.R_blend_lsst_r_extnbr_v22
    )
    frame["prediction"] = frame.R_blend_lsst_r_extnbr_v22
    return derive_physical_features(frame)


def load_constgold(gap_path: str, pair_path: str,
                   half_selfresp: str) -> pd.DataFrame:
    gap = pd.read_feather(
        gap_path, columns=[*KEY, "R_sim", "R_blend", "gap"]
    )
    pairs = pd.read_feather(pair_path, columns=[
        *KEY, "dominant_response", "dominant_distance",
        "dominant_abs_response", "primary_mag", "primary_size",
        "dominant_secondary_size", "dominant_flux_ratio_primary",
        "top_abs_fraction", "dominant_to_runner_up_abs_response",
    ])
    half = pd.read_feather(half_selfresp, columns=[*KEY, "r_sim_self"])
    half = half.loc[np.isfinite(half.r_sim_self)].copy()
    for name, local in (("gap", gap), ("pairs", pairs), ("half", half)):
        if local.duplicated(KEY).any():
            raise RuntimeError(f"duplicate constgold {name} key")
    frame = gap.merge(pairs, on=KEY, how="inner", validate="one_to_one")
    if len(frame) != len(gap):
        raise RuntimeError("constgold pair table does not cover the gap table")
    frame = frame.merge(half, on=KEY, how="inner", validate="one_to_one")
    frame["deficit_proxy"] = frame.R_sim - frame.r_sim_self - frame.R_blend
    frame["deficit_total"] = frame.gap
    frame["self_flow"] = frame.deficit_total - frame.deficit_proxy
    frame["prediction"] = frame.R_blend
    closure = (
        frame.deficit_proxy + frame.self_flow - frame.deficit_total
    ).to_numpy(float)
    if float(np.max(np.abs(closure))) > 1.0e-12:
        raise RuntimeError("constgold decomposition does not close per row")
    return derive_physical_features(frame)


def _clean_json(value):
    if isinstance(value, dict):
        return {key: _clean_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clean_json(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def markdown(payload: dict) -> str:
    lines = [
        "# V2.2 dominant-pair physical-root test",
        "",
        "Positive means V2.2 underpredicts. Development outcomes were not used: "
        "development cases supplied covariate scales and quantiles only, and all "
        "reported effects are on validation cases with case-level uncertainty.",
        "",
    ]
    for carrier_name, carrier in payload["carriers"].items():
        lines.extend([
            f"## {carrier_name}",
            "",
            f"Frozen rule: `{carrier['description']}`.",
            "",
            "| Physical coordinate | Anchor controlled high-low | Holm p | "
            "Constgold neighbour controlled high-low | Holm p | Same sign |",
            "|---|---:|---:|---:|---:|:---:|",
        ])
        anchor = carrier["anchor_validation"]
        const = carrier["constgold_validation"]["deficit_proxy"]
        for feature in PHYSICAL_FEATURES:
            a = anchor["features"][feature]["controlled_upper_minus_lower"][
                "high_minus_low"
            ]
            c = const["features"][feature]["controlled_upper_minus_lower"][
                "high_minus_low"
            ]
            same = (np.sign(a["mean"]) == np.sign(c["mean"]))
            lines.append(
                f"| `{FEATURE_LABELS[feature]}` | {a['mean']:+.5f} +- "
                f"{a['case_sem']:.5f} | {a['p_holm_four_features']:.3g} | "
                f"{c['mean']:+.5f} +- {c['case_sem']:.5f} | "
                f"{c['p_holm_four_features']:.3g} | {'yes' if same else 'no'} |"
            )
        lines.extend([
            "",
            "Joint within-cell slopes (deficit per development-population SD):",
            "",
            "| Independent coordinate | Anchor slope | Holm p | "
            "Constgold neighbour slope | Holm p |",
            "|---|---:|---:|---:|---:|",
        ])
        aj = anchor["joint_physical_basis"]["features"]
        cj = const["joint_physical_basis"]["features"]
        for feature in JOINT_BASIS:
            lines.append(
                f"| `{FEATURE_LABELS[feature]}` | {aj[feature]['mean']:+.5f} +- "
                f"{aj[feature]['case_sem']:.5f} | "
                f"{aj[feature]['p_holm_within_model']:.3g} | "
                f"{cj[feature]['mean']:+.5f} +- {cj[feature]['case_sem']:.5f} | "
                f"{cj[feature]['p_holm_within_model']:.3g} |"
            )
        lines.append("")
    lines.extend([
        "## Interpretation guard",
        "",
        "The constgold neighbour target is a same-object localization proxy, not "
        "a literal per-object BlendEMU error label. Surface brightness is derived "
        "exactly from flux and size ratios, so it is not an independent fourth "
        "coordinate in the joint model. These are diagnostic associations; none "
        "is fitted as a correction.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--anchor-response", nargs="+", required=True)
    ap.add_argument("--anchor-dominance", nargs="+", required=True)
    ap.add_argument("--constgold-gap", required=True)
    ap.add_argument("--constgold-pairs", required=True)
    ap.add_argument("--half-selfresp", required=True)
    ap.add_argument("--anchor-development-max", type=int, default=599)
    ap.add_argument("--constgold-development-max", type=int, default=89)
    ap.add_argument("--n-control-bins", type=int, default=4)
    ap.add_argument(
        "--amplitude-control", choices=sorted(AMPLITUDE_CONTROLS),
        default="dominant_pair",
    )
    ap.add_argument("--output-json", required=True)
    ap.add_argument("--output-md", required=True)
    args = ap.parse_args()
    for output in (args.output_json, args.output_md):
        if os.path.exists(output):
            raise FileExistsError(f"refusing existing output {output}")

    anchor = load_anchor(args.anchor_response, args.anchor_dominance)
    constgold = load_constgold(
        args.constgold_gap, args.constgold_pairs, args.half_selfresp
    )
    anchor_dev = anchor.loc[anchor.case <= args.anchor_development_max].copy()
    anchor_val = anchor.loc[anchor.case > args.anchor_development_max].copy()
    const_dev = constgold.loc[
        constgold.case <= args.constgold_development_max
    ].copy()
    const_val = constgold.loc[
        constgold.case > args.constgold_development_max
    ].copy()
    for name, local in (
        ("anchor development", anchor_dev), ("anchor validation", anchor_val),
        ("constgold development", const_dev), ("constgold validation", const_val),
    ):
        if local.empty or local.case.nunique() < 10:
            raise RuntimeError(f"empty or too-small {name} block")

    payload = {
        "design": (
            "fixed response-dominance carrier; development covariates only define "
            f"scales and quantiles; validation effects block on case and on a "
            f"{args.n_control_bins}x{args.n_control_bins} grid of "
            f"{args.amplitude_control} predicted amplitude x dominance strength; "
            "positive deficit means V2.2 underpredicts"
        ),
        "feature_definitions": {
            "log10_flux_ratio": "log10(F_secondary/F_primary)",
            "log10_size_ratio": "log10(Re_secondary/Re_primary)",
            "log10_overlap_scale": "log10((Re_primary+Re_secondary)/distance)",
            "log10_surface_brightness_ratio": (
                "log10_flux_ratio - 2*log10_size_ratio"
            ),
        },
        "control_definitions": {
            "log10_dominant_abs_response": (
                "log10(abs predicted response of the dominant pair)"
            ),
            "log10_dominance_strength": (
                "log10(dominant abs response / runner-up abs response)"
            ),
            "abs_scene_prediction": "abs(sum of deployed pair predictions)",
        },
        "amplitude_control": args.amplitude_control,
        "n_control_bins": int(args.n_control_bins),
        "windows": {
            "anchor_development": [
                int(anchor_dev.case.min()), int(anchor_dev.case.max())
            ],
            "anchor_validation": [
                int(anchor_val.case.min()), int(anchor_val.case.max())
            ],
            "constgold_development": [
                int(const_dev.case.min()), int(const_dev.case.max())
            ],
            "constgold_validation": [
                int(const_val.case.min()), int(const_val.case.max())
            ],
        },
        "targets": {
            "anchor_deficit": "R_blend_truth - R_blend_V2.2",
            "constgold_deficit_proxy": "R_sim - r_sim_self - R_blend_V2.2",
            "constgold_self_flow": "r_sim_self - R_flow",
            "constgold_deficit_total": "R_sim - R_flow - R_blend_V2.2",
        },
        "multiplicity": (
            "Holm correction across the four physical features separately for "
            "controlled contrasts and univariate slopes; joint three-coordinate "
            "basis corrected across its three coefficients"
        ),
        "carriers": {},
    }

    for carrier_name, spec in CARRIERS.items():
        a_dev_mask = carrier_mask(anchor_dev, carrier_name)
        a_val_mask = carrier_mask(anchor_val, carrier_name)
        c_dev_mask = carrier_mask(const_dev, carrier_name)
        c_val_mask = carrier_mask(const_val, carrier_name)
        a_dev_carrier = anchor_dev.loc[a_dev_mask].copy()
        a_val_carrier = anchor_val.loc[a_val_mask].copy()
        c_dev_carrier = const_dev.loc[c_dev_mask].copy()
        c_val_carrier = const_val.loc[c_val_mask].copy()
        print(
            f"{carrier_name}: anchor dev/val={len(a_dev_carrier):,}/"
            f"{len(a_val_carrier):,}; constgold dev/val={len(c_dev_carrier):,}/"
            f"{len(c_val_carrier):,}", flush=True,
        )
        anchor_design = learn_covariate_design(
            a_dev_carrier, args.n_control_bins, args.amplitude_control
        )
        const_design = learn_covariate_design(
            c_dev_carrier, args.n_control_bins, args.amplitude_control
        )
        entry = {
            "description": spec["description"],
            "anchor_covariate_design": anchor_design,
            "constgold_covariate_design": const_design,
            "anchor_carrier_summary_validation": summarize_carrier(
                anchor_val, "deficit", a_val_mask
            ),
            "constgold_carrier_summary_validation": {
                target: summarize_carrier(const_val, target, c_val_mask)
                for target in ("deficit_proxy", "self_flow", "deficit_total")
            },
            "anchor_validation": analyze_validation(
                a_val_carrier, "deficit", anchor_design
            ),
            "constgold_validation": {
                target: analyze_validation(c_val_carrier, target, const_design)
                for target in ("deficit_proxy", "self_flow", "deficit_total")
            },
        }
        payload["carriers"][carrier_name] = entry

    payload = _clean_json(payload)
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    with open(args.output_md, "x", encoding="utf-8") as handle:
        handle.write(markdown(payload))
    print(markdown(payload), flush=True)
    print("V22_DOMINANT_PHYSICS_DONE", flush=True)


if __name__ == "__main__":
    main()
