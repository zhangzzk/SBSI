"""Find a frozen, interpretable V2.2 gap carrier shared by anchors and constgold.

The search uses amplitude-matched (|g|=0.02) coherent anchors and the
same-object constgold neighbour-term proxy

    (constgold total truth - half-shear self truth) - V2.2 blend prediction.

Signs are aligned so positive always means V2.2 underpredicts.  Cases 400--599
and 40--89 are development blocks.  The selected rule is then frozen and
evaluated on anchor cases 600--899 and constgold cases 90--139.  Candidate
rules are a predeclared grid of simple response-structure and dominant-pair
physics cuts; no measured response is used as a coordinate.

The neighbour-term proxy includes selection-estimand differences and possible
non-additivity.  It is not uniquely an emulator-error label.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from dataclasses import dataclass
from functools import reduce
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


KEY = ["case", "input_index"]
FEATURES = [
    "primary_mag", "primary_size", "primary_sersic_n",
    "dominant_response", "dominant_abs_response", "dominant_distance",
    "dominant_secondary_mag", "dominant_secondary_size",
    "dominant_secondary_sersic_n", "dominant_flux_ratio_primary",
    "dominant_flux_share_neighbours", "dominant_flux_rank",
    "dominant_distance_rank", "runner_up_abs_response", "R_abs_sum",
    "n_pairs", "n_response_pairs_ge_10pct_max", "top_abs_fraction",
    "dominant_to_runner_up_abs_response", "signed_to_abs_response",
]


@dataclass(frozen=True)
class Condition:
    name: str
    feature: str
    op: str
    lo: float | None = None
    hi: float | None = None
    value: float | None = None
    family: str = "other"

    def apply(self, frame: pd.DataFrame) -> np.ndarray:
        x = frame[self.feature].to_numpy(float)
        if self.op == "between":
            assert self.lo is not None and self.hi is not None
            return (x >= self.lo) & (x < self.hi)
        if self.op == "gt":
            assert self.value is not None
            return x > self.value
        if self.op == "ge":
            assert self.value is not None
            return x >= self.value
        if self.op == "lt":
            assert self.value is not None
            return x < self.value
        if self.op == "eq":
            assert self.value is not None
            return x == self.value
        raise ValueError(f"unknown condition op {self.op}")

    def payload(self) -> dict:
        def finite_or_none(value: float | None) -> float | None:
            if value is None or not np.isfinite(value):
                return None
            return float(value)

        return {
            "name": self.name, "feature": self.feature, "op": self.op,
            "lo": finite_or_none(self.lo), "hi": finite_or_none(self.hi),
            "value": finite_or_none(self.value),
            "family": self.family,
        }


def interval_conditions(feature: str, edges: list[float], family: str,
                        prefix: str | None = None) -> list[Condition]:
    prefix = prefix or feature
    return [
        Condition(
            name=f"{prefix}:[{lo:g},{hi:g})", feature=feature,
            op="between", lo=lo, hi=hi, family=family,
        )
        for lo, hi in zip(edges[:-1], edges[1:])
    ]


def candidate_rules() -> tuple[dict[str, Condition], list[tuple[str, ...]]]:
    """Return the fixed condition grid and unique rule intersections."""
    distance = interval_conditions(
        "dominant_distance", [0, 1, 2, 3, 5, 10.000001], "distance", "distance"
    )
    sign = [
        Condition("dominant_response:negative", "dominant_response", "lt",
                  value=0.0, family="sign"),
        Condition("dominant_response:positive", "dominant_response", "ge",
                  value=0.0, family="sign"),
    ]
    ratio = [
        Condition(f"ratio:>{value:g}", "dominant_to_runner_up_abs_response",
                  "gt", value=value, family="structure")
        for value in (5.0, 10.0, 20.0, 40.0, 100.0)
    ]
    top = [
        Condition(f"top_fraction:>{value:g}", "top_abs_fraction", "gt",
                  value=value, family="structure")
        for value in (0.5, 0.6, 0.7, 0.7752772106835227, 0.85, 0.9)
    ]
    dominant_abs = interval_conditions(
        "dominant_abs_response",
        [0, 0.005, 0.015, 0.035, 0.1, 0.2, np.inf],
        "structure", "dominant_abs",
    )
    runner_abs = interval_conditions(
        "runner_up_abs_response",
        [0, 0.002, 0.004, 0.008, 0.02, np.inf],
        "structure", "runner_abs",
    )
    abs_sum = interval_conditions(
        "R_abs_sum", [0, 0.015, 0.03, 0.065, 0.18, np.inf],
        "structure", "abs_sum",
    )
    strong_count = [
        Condition("n_strong:eq1", "n_response_pairs_ge_10pct_max", "eq",
                  value=1, family="structure"),
        Condition("n_strong:eq2", "n_response_pairs_ge_10pct_max", "eq",
                  value=2, family="structure"),
        Condition("n_strong:ge3", "n_response_pairs_ge_10pct_max", "ge",
                  value=3, family="structure"),
    ]
    signed = interval_conditions(
        "signed_to_abs_response", [-1.000001, -0.5, 0, 0.5, 1.000001],
        "structure", "signed_to_abs",
    )
    flux = interval_conditions(
        "dominant_flux_ratio_primary", [0, 0.5, 1, 2, 5, np.inf],
        "physics", "flux_ratio",
    )
    ranks = [
        Condition("dominant_is_brightest", "dominant_flux_rank", "eq",
                  value=1, family="physics"),
        Condition("dominant_not_brightest", "dominant_flux_rank", "gt",
                  value=1, family="physics"),
        Condition("dominant_is_closest", "dominant_distance_rank", "eq",
                  value=1, family="physics"),
        Condition("dominant_not_closest", "dominant_distance_rank", "gt",
                  value=1, family="physics"),
    ]
    morphology = [
        *interval_conditions(
            "primary_mag", [-np.inf, 22, 23, 24, 25, 25.800001],
            "physics", "primary_mag",
        ),
        *interval_conditions(
            "dominant_secondary_mag", [-np.inf, 22, 23, 24, 25, np.inf],
            "physics", "secondary_mag",
        ),
        *interval_conditions(
            "primary_size", [0.5, 0.7, 1.0, 1.5, np.inf],
            "physics", "primary_size",
        ),
        *interval_conditions(
            "dominant_secondary_size", [0, 0.5, 0.8, 1.2, np.inf],
            "physics", "secondary_size",
        ),
    ]
    all_conditions = [
        *distance, *sign, *ratio, *top, *dominant_abs, *runner_abs,
        *abs_sum, *strong_count, *signed, *flux, *ranks, *morphology,
    ]
    conditions = {item.name: item for item in all_conditions}
    if len(conditions) != len(all_conditions):
        raise RuntimeError("duplicate condition name")

    rules: set[tuple[str, ...]] = {(name,) for name in conditions}
    structure = ratio + top + dominant_abs + runner_abs + abs_sum + strong_count + signed
    for item in structure:
        for d in distance:
            rules.add((item.name, d.name))
            for s in sign:
                rules.add((item.name, d.name, s.name))
        for s in sign:
            rules.add((item.name, s.name))
    for d in distance:
        for s in sign:
            rules.add((d.name, s.name))

    # Detailed physical intersections are restricted to the two anchor-derived
    # dominance definitions so the grid stays interpretable and bounded.
    anchor_carriers = [
        conditions["ratio:>20"],
        conditions["top_fraction:>0.775277"],
    ]
    for carrier in anchor_carriers:
        for physical in [*flux, *ranks, *morphology]:
            rules.add((carrier.name, physical.name))
    return conditions, sorted(rules)


def normalize_anchor(dom: pd.DataFrame, response: pd.DataFrame) -> pd.DataFrame:
    rename = {
        "anchor_index": "input_index",
        "r_input_p": "primary_mag",
        "Re_input_p": "primary_size",
        "sersic_n_input_p": "primary_sersic_n",
        "response": "dominant_response",
    }
    dom = dom.rename(columns=rename)
    frame = response.merge(dom, on=KEY, how="inner", validate="one_to_one")
    frame["deficit"] = (
        frame.R_blend_truth - frame.R_blend_lsst_r_extnbr_v22
    )
    frame["signed_to_abs_response"] = np.divide(
        frame.R_blend_lsst_r_extnbr_v22.to_numpy(float),
        frame.R_abs_sum.to_numpy(float),
        out=np.zeros(len(frame), dtype=float),
        where=frame.R_abs_sum.to_numpy(float) > 0,
    )
    return frame[[*KEY, *FEATURES, "deficit"]].copy()


def load_anchor(response_paths: list[str], dominance_paths: list[str]) -> pd.DataFrame:
    response_columns = [
        *KEY, "R_blend_truth", "R_blend_lsst_r_extnbr_v22",
    ]
    dominance_columns = [
        "case", "anchor_index", "r_input_p", "Re_input_p",
        "sersic_n_input_p", "response", "dominant_abs_response",
        "dominant_distance", "dominant_secondary_mag",
        "dominant_secondary_size", "sersic_n_input_s",
        "dominant_flux_ratio_primary", "dominant_flux_share_neighbours",
        "dominant_flux_rank", "dominant_distance_rank",
        "runner_up_abs_response", "R_abs_sum", "n_pairs",
        "n_response_pairs_ge_10pct_max", "top_abs_fraction",
        "dominant_to_runner_up_abs_response",
    ]
    responses = pd.concat(
        [pd.read_feather(path, columns=response_columns) for path in response_paths],
        ignore_index=True,
    )
    dominance = pd.concat(
        [pd.read_feather(path, columns=dominance_columns) for path in dominance_paths],
        ignore_index=True,
    ).rename(columns={"sersic_n_input_s": "dominant_secondary_sersic_n"})
    if responses.duplicated(KEY).any() or dominance.duplicated(
        ["case", "anchor_index"]
    ).any():
        raise RuntimeError("duplicate anchor keys")
    return normalize_anchor(dominance, responses)


def load_constgold(gap_path: str, pair_path: str, half_selfresp: str,
                   case_min: int, case_max: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    gap = pd.read_feather(
        gap_path, columns=[*KEY, "R_sim", "R_blend", "gap"]
    )
    gap = gap.loc[gap.case.between(case_min, case_max)].copy()
    pairs = pd.read_feather(pair_path, columns=[*KEY, *FEATURES[:-1]])
    pairs = pairs.loc[pairs.case.between(case_min, case_max)].copy()
    if gap.duplicated(KEY).any() or pairs.duplicated(KEY).any():
        raise RuntimeError("duplicate constgold gap/pair keys")
    full = gap.merge(pairs, on=KEY, how="inner", validate="one_to_one")
    if len(full) != len(gap):
        raise RuntimeError(
            f"constgold pair features cover {len(full):,}/{len(gap):,} gap rows"
        )
    full["signed_to_abs_response"] = np.divide(
        full.R_blend.to_numpy(float), full.R_abs_sum.to_numpy(float),
        out=np.zeros(len(full), dtype=float),
        where=full.R_abs_sum.to_numpy(float) > 0,
    )
    full["deficit_total"] = full.gap
    full_required = [*FEATURES, "deficit_total"]
    if not np.isfinite(full[full_required].to_numpy(float)).all():
        raise RuntimeError("non-finite full constgold shared-localization row")
    half = pd.read_feather(half_selfresp, columns=[*KEY, "r_sim_self"])
    half = half.loc[half.case.between(case_min, case_max)].copy()
    half = half.loc[np.isfinite(half.r_sim_self)].copy()
    if half.duplicated(KEY).any():
        raise RuntimeError("duplicate half-shear self-response keys")
    matched = full.merge(half, on=KEY, how="inner", validate="one_to_one")
    matched["deficit_proxy"] = (
        matched.R_sim - matched.r_sim_self - matched.R_blend
    )
    required = [*FEATURES, "deficit_proxy", "deficit_total"]
    if not np.isfinite(matched[required].to_numpy(float)).all():
        raise RuntimeError("non-finite constgold shared-localization row")
    return (
        full[[*KEY, *full_required]].copy(),
        matched[[*KEY, *required]].copy(),
    )


def basic_stat(values: np.ndarray) -> dict:
    values = np.asarray(values, float)
    sd = float(values.std(ddof=1)) if len(values) > 1 else float("nan")
    return {
        "mean": float(values.mean()), "case_sd": sd,
        "case_sem": sd / np.sqrt(len(values)) if len(values) > 1 else float("nan"),
        "n_cases": int(len(values)),
    }


class CaseAggregator:
    def __init__(self, frame: pd.DataFrame, deficit_column: str):
        self.frame = frame
        self.deficit = frame[deficit_column].to_numpy(float)
        self.cases, self.inverse = np.unique(
            frame.case.to_numpy(np.int64), return_inverse=True
        )
        self.count = np.bincount(self.inverse).astype(float)
        self.total = np.bincount(self.inverse, weights=self.deficit)
        self.global_case = self.total / self.count

    def summarize(self, mask: np.ndarray) -> dict | None:
        mask = np.asarray(mask, bool)
        selected_count = np.bincount(self.inverse, weights=mask.astype(float))
        if np.any(selected_count == 0) or np.any(selected_count == self.count):
            return None
        selected_sum = np.bincount(
            self.inverse, weights=np.where(mask, self.deficit, 0.0)
        )
        complement_count = self.count - selected_count
        complement_sum = self.total - selected_sum
        selected = selected_sum / selected_count
        complement = complement_sum / complement_count
        contribution = selected_sum / self.count
        fraction = selected_count / self.count
        contrast = selected - complement
        test = stats.ttest_1samp(contrast, 0.0)
        global_mean = float(self.global_case.mean())
        return {
            "n_rows": int(mask.sum()),
            "global_deficit": basic_stat(self.global_case),
            "selected_fraction": basic_stat(fraction),
            "selected_conditional_deficit": basic_stat(selected),
            "complement_conditional_deficit": basic_stat(complement),
            "selected_minus_complement": basic_stat(contrast),
            "selected_contribution": basic_stat(contribution),
            "carrier_share": float(contribution.mean() / global_mean),
            "contrast_t": (
                float(test.statistic) if np.isfinite(test.statistic) else None
            ),
            "contrast_p": float(test.pvalue) if np.isfinite(test.pvalue) else None,
        }


def rule_mask(rule: tuple[str, ...], masks: dict[str, np.ndarray]) -> np.ndarray:
    return reduce(np.logical_and, (masks[name] for name in rule))


def search_score(anchor: dict | None, const: dict | None) -> float | None:
    if anchor is None or const is None:
        return None
    fractions = [
        anchor["selected_fraction"]["mean"], const["selected_fraction"]["mean"]
    ]
    if min(fractions) < 0.02 or max(fractions) > 0.60:
        return None
    selected = [
        anchor["selected_conditional_deficit"],
        const["selected_conditional_deficit"],
    ]
    contrast = [anchor["selected_minus_complement"], const["selected_minus_complement"]]
    if min(item["mean"] for item in selected) <= 0:
        return None
    if min(item["mean"] for item in contrast) <= 0:
        return None
    min_share = min(anchor["carrier_share"], const["carrier_share"])
    min_contrast_z = min(
        item["mean"] / item["case_sem"] for item in contrast
    )
    min_selected_z = min(item["mean"] / item["case_sem"] for item in selected)
    return float(
        min_share
        + 0.05 * np.clip(min_contrast_z, -5.0, 5.0)
        + 0.02 * np.clip(min_selected_z, -5.0, 5.0)
    )


def evaluate_rule(rule: tuple[str, ...], conditions: dict[str, Condition],
                  frame: pd.DataFrame, aggregator: CaseAggregator) -> dict | None:
    mask = np.ones(len(frame), dtype=bool)
    for name in rule:
        mask &= conditions[name].apply(frame)
    return aggregator.summarize(mask)


def validation_gates(anchor: dict, const_proxy: dict) -> dict:
    def z(item: dict) -> float:
        return float(item["mean"] / item["case_sem"])

    return {
        "selected_deficit_positive_gt2sem_both": bool(
            min(z(anchor["selected_conditional_deficit"]),
                z(const_proxy["selected_conditional_deficit"])) > 2.0
        ),
        "selected_exceeds_complement_gt2sem_both": bool(
            min(z(anchor["selected_minus_complement"]),
                z(const_proxy["selected_minus_complement"])) > 2.0
        ),
        "carrier_share_at_least_half_both": bool(
            min(anchor["carrier_share"], const_proxy["carrier_share"]) >= 0.5
        ),
        "complement_within_2sem_of_zero_both": bool(
            max(
                abs(z(anchor["complement_conditional_deficit"])),
                abs(z(const_proxy["complement_conditional_deficit"])),
            ) <= 2.0
        ),
    }


def markdown(payload: dict) -> str:
    chosen = payload["chosen_rule"]
    val = payload["validation"]
    windows = payload["windows"]
    lines = [
        "# Shared V2.2 anchor/constgold gap localization",
        "",
        "Positive residual means V2.2 underpredicts. The rule was selected on "
        f"anchor cases {windows['anchor_development'][0]}–{windows['anchor_development'][1]} "
        f"and constgold cases {windows['constgold_development'][0]}–"
        f"{windows['constgold_development'][1]}, then frozen for anchor cases "
        f"{windows['anchor_validation'][0]}–{windows['anchor_validation'][1]} "
        f"and constgold cases {windows['constgold_validation'][0]}–"
        f"{windows['constgold_validation'][1]}.",
        "",
        f"Chosen rule: `{chosen['description']}`.",
        "",
        "| Validation target | Fraction | Conditional deficit | Rest deficit | Contribution | Carrier share |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, item in (
        (f"Anchors, |g|={payload['anchor_shear']:g}", val["anchor"]),
        ("Constgold neighbour proxy", val["constgold_proxy"]),
        ("Constgold total", val["constgold_total"]),
    ):
        lines.append(
            f"| {name} | {item['selected_fraction']['mean']:.1%} | "
            f"{item['selected_conditional_deficit']['mean']:+.5f} ± "
            f"{item['selected_conditional_deficit']['case_sem']:.5f} | "
            f"{item['complement_conditional_deficit']['mean']:+.5f} ± "
            f"{item['complement_conditional_deficit']['case_sem']:.5f} | "
            f"{item['selected_contribution']['mean']:+.5f} | "
            f"{item['carrier_share']:.1%} |"
        )
    lines.extend([
        "", "Validation gates: " + ", ".join(
            f"`{name}={value}`" for name, value in payload["validation_gates"].items()
        ) + ".", "",
        "The constgold neighbour proxy is total constgold truth minus matched "
        "half-shear self truth. It also contains selection-estimand differences "
        "and non-additivity, so this is localization rather than a literal emulator label.",
        "",
    ])
    provenance = payload["pilot_frozen_rule_provenance"]
    frozen_name = provenance["rule"]
    if provenance["untouched_anchor_validation_in_this_run"] is not None:
        frozen = payload["fixed_anchor_carrier_controls"][frozen_name]
        lines.extend([
            "## Pilot-frozen rule on untouched anchors", "",
            f"Rule: `{frozen_name}`. It was selected on "
            f"{provenance['selected_on']} and first replicated on "
            f"{provenance['first_replication']}. The anchor values below use "
            f"the previously untouched {provenance['untouched_anchor_validation_in_this_run']} block.",
            "",
            "| Target | Fraction | Conditional deficit | Rest deficit | Carrier share |",
            "|---|---:|---:|---:|---:|",
        ])
        for name, item in (
            ("Anchors", frozen["anchor"]),
            ("Constgold neighbour proxy", frozen["constgold_proxy"]),
            ("Constgold total", frozen["constgold_total"]),
        ):
            lines.append(
                f"| {name} | {item['selected_fraction']['mean']:.1%} | "
                f"{item['selected_conditional_deficit']['mean']:+.5f} ± "
                f"{item['selected_conditional_deficit']['case_sem']:.5f} | "
                f"{item['complement_conditional_deficit']['mean']:+.5f} ± "
                f"{item['complement_conditional_deficit']['case_sem']:.5f} | "
                f"{item['carrier_share']:.1%} |"
            )
        lines.append("")
    total = payload["total_gap_search"]
    total_val = total["validation"]
    lines.extend([
        "## Exploratory total-gap search", "",
        f"Chosen total-gap rule: `{total['chosen_rule']['description']}`.",
        "",
        f"On validation it carries {total_val['anchor']['carrier_share']:.1%} "
        f"of the anchor deficit and {total_val['constgold_total']['carrier_share']:.1%} "
        "of the full constgold deficit.",
        "Total-gap validation gates: " + ", ".join(
            f"`{name}={value}`"
            for name, value in total["validation_gates"].items()
        ) + ". A failed gate keeps this result exploratory.", "",
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
    ap.add_argument("--anchor-shear", type=float, default=0.02)
    ap.add_argument("--constgold-development-max", type=int, default=89)
    ap.add_argument("--top-k", type=int, default=25)
    ap.add_argument("--output-json", required=True)
    ap.add_argument("--output-md", required=True)
    args = ap.parse_args()
    for output in (args.output_json, args.output_md):
        if os.path.exists(output):
            raise FileExistsError(f"refusing existing output {output}")

    anchor = load_anchor(args.anchor_response, args.anchor_dominance)
    const_full, const = load_constgold(
        args.constgold_gap, args.constgold_pairs, args.half_selfresp, 40, 139
    )
    anchor_dev = anchor.loc[anchor.case <= args.anchor_development_max].copy()
    anchor_val = anchor.loc[anchor.case > args.anchor_development_max].copy()
    const_dev = const.loc[const.case <= args.constgold_development_max].copy()
    const_val = const.loc[const.case > args.constgold_development_max].copy()
    const_full_dev = const_full.loc[
        const_full.case <= args.constgold_development_max
    ].copy()
    const_full_val = const_full.loc[
        const_full.case > args.constgold_development_max
    ].copy()
    observed_anchor = np.sort(anchor.case.unique())
    expected_anchor = np.arange(observed_anchor.min(), observed_anchor.max() + 1)
    if not np.array_equal(observed_anchor, expected_anchor):
        raise RuntimeError("anchor case range is not contiguous")
    if anchor_dev.case.nunique() < 20 or anchor_val.case.nunique() < 20:
        raise RuntimeError("need at least 20 anchor cases in each block")
    if const_dev.case.nunique() != 50 or const_val.case.nunique() != 50:
        raise RuntimeError("expected two 50-case constgold blocks")

    conditions, rules = candidate_rules()
    masks_anchor = {name: item.apply(anchor_dev) for name, item in conditions.items()}
    masks_const = {name: item.apply(const_dev) for name, item in conditions.items()}
    masks_const_total = {
        name: item.apply(const_full_dev) for name, item in conditions.items()
    }
    agg_anchor = CaseAggregator(anchor_dev, "deficit")
    agg_const = CaseAggregator(const_dev, "deficit_proxy")
    agg_const_total = CaseAggregator(const_full_dev, "deficit_total")

    scanned = []
    scanned_total = []
    for index, rule in enumerate(rules, start=1):
        anchor_summary = agg_anchor.summarize(rule_mask(rule, masks_anchor))
        const_summary = agg_const.summarize(rule_mask(rule, masks_const))
        score = search_score(anchor_summary, const_summary)
        if score is not None:
            scanned.append({
                "rule": list(rule), "description": " AND ".join(rule),
                "score": score, "anchor": anchor_summary,
                "constgold_proxy": const_summary,
            })
        const_total_summary = agg_const_total.summarize(
            rule_mask(rule, masks_const_total)
        )
        total_score = search_score(anchor_summary, const_total_summary)
        if total_score is not None:
            scanned_total.append({
                "rule": list(rule), "description": " AND ".join(rule),
                "score": total_score, "anchor": anchor_summary,
                "constgold_total": const_total_summary,
            })
        if index % 100 == 0:
            print(f"searched {index}/{len(rules)} candidate rules", flush=True)
    if not scanned:
        raise RuntimeError("no candidate rule passed development eligibility")
    if not scanned_total:
        raise RuntimeError("no total-gap rule passed development eligibility")
    scanned.sort(key=lambda item: item["score"], reverse=True)
    scanned_total.sort(key=lambda item: item["score"], reverse=True)
    chosen = scanned[0]
    chosen_rule = tuple(chosen["rule"])

    val_anchor_agg = CaseAggregator(anchor_val, "deficit")
    val_proxy_agg = CaseAggregator(const_val, "deficit_proxy")
    val_total_agg = CaseAggregator(const_full_val, "deficit_total")
    validation = {
        "anchor": evaluate_rule(chosen_rule, conditions, anchor_val, val_anchor_agg),
        "constgold_proxy": evaluate_rule(
            chosen_rule, conditions, const_val, val_proxy_agg
        ),
        "constgold_total": evaluate_rule(
            chosen_rule, conditions, const_full_val, val_total_agg
        ),
    }
    if any(item is None for item in validation.values()):
        raise RuntimeError("chosen rule is empty/full in a validation block")

    total_chosen = scanned_total[0]
    total_chosen_rule = tuple(total_chosen["rule"])
    total_validation = {
        "anchor": evaluate_rule(
            total_chosen_rule, conditions, anchor_val, val_anchor_agg
        ),
        "constgold_total": evaluate_rule(
            total_chosen_rule, conditions, const_full_val, val_total_agg
        ),
        "constgold_proxy": evaluate_rule(
            total_chosen_rule, conditions, const_val, val_proxy_agg
        ),
    }
    if any(item is None for item in total_validation.values()):
        raise RuntimeError("total-gap chosen rule is empty/full in validation")

    controls = {}
    for names in (
        ("ratio:>5", "dominant_response:positive"),
        ("ratio:>10", "dominant_response:positive"),
        ("ratio:>20",),
        ("top_fraction:>0.775277",),
    ):
        label = " AND ".join(names)
        controls[label] = {
            "anchor": evaluate_rule(names, conditions, anchor_val, val_anchor_agg),
            "constgold_proxy": evaluate_rule(
                names, conditions, const_val, val_proxy_agg
            ),
            "constgold_total": evaluate_rule(
                names, conditions, const_full_val, val_total_agg
            ),
        }

    gates = validation_gates(validation["anchor"], validation["constgold_proxy"])
    payload = {
        "design": (
            "fixed interpretable rule grid; development and validation windows "
            "listed explicitly; case is uncertainty unit; positive means model underprediction"
        ),
        "interpretation_guard": (
            "constgold proxy = total truth - matched half-shear self truth - V2.2 blend; "
            "includes selection-estimand differences and possible non-additivity"
        ),
        "population": {
            "anchor_rows": int(len(anchor)), "anchor_cases": int(anchor.case.nunique()),
            "constgold_full_rows": int(len(const_full)),
            "constgold_shared_rows": int(len(const)),
            "constgold_shared_fraction": float(len(const) / len(const_full)),
            "constgold_cases": int(const.case.nunique()),
        },
        "anchor_shear": float(args.anchor_shear),
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
        "candidate_grid": {
            "n_conditions": len(conditions), "n_rules": len(rules),
            "conditions": {name: item.payload() for name, item in conditions.items()},
            "eligibility": "2%-60% in both development datasets; positive selected and selected-minus-rest deficits in both",
            "ranking": "maximise minimum carrier share with small capped bonuses for case-level selected and contrast z-scores",
        },
        "chosen_rule": {
            "conditions": list(chosen_rule),
            "description": chosen["description"],
            "development_score": chosen["score"],
            "development": {
                "anchor": chosen["anchor"],
                "constgold_proxy": chosen["constgold_proxy"],
            },
        },
        "validation": validation,
        "validation_gates": gates,
        "total_gap_search": {
            "role": (
                "exploratory search against the full constgold residual; this mixes "
                "the neighbour and self/flow branches"
            ),
            "chosen_rule": {
                "conditions": list(total_chosen_rule),
                "description": total_chosen["description"],
                "development_score": total_chosen["score"],
                "development": {
                    "anchor": total_chosen["anchor"],
                    "constgold_total": total_chosen["constgold_total"],
                },
            },
            "validation": total_validation,
            "validation_gates": validation_gates(
                total_validation["anchor"], total_validation["constgold_total"]
            ),
            "top_development_rules": scanned_total[:args.top_k],
        },
        "fixed_anchor_carrier_controls": controls,
        "pilot_frozen_rule_provenance": {
            "rule": "ratio:>5 AND dominant_response:positive",
            "selected_on": "anchor c400--449 plus constgold-proxy c40--89",
            "first_replication": "anchor c450--499 plus constgold-proxy c90--139",
            "untouched_anchor_validation_in_this_run": (
                f"c{int(anchor_val.case.min())}--{int(anchor_val.case.max())}"
                if int(anchor_val.case.min()) >= 600 else None
            ),
        },
        "top_development_rules": scanned[:args.top_k],
    }
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    with open(args.output_md, "x", encoding="utf-8") as handle:
        handle.write(markdown(payload))
    print(markdown(payload), flush=True)
    print("V22_SHARED_GAP_LOCALIZATION_DONE", flush=True)


if __name__ == "__main__":
    main()
