#!/usr/bin/env python3
"""Validate and tabulate the fixed-g0 ConstGold model comparison."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from scripts.evaluate_constgold_fixed_g0_response import (
    BRANCHES,
    file_sha256,
    pooled_sufficient,
    response,
    write_json,
)


MEASURED_BRANCH = {
    "matched_usable": "matched_usable",
    "actual_usable_flags": "actual_usable_flags",
    "modeled_usable": "actual_usable_flags",
}
EXPECTED_SUBSETS = {
    "cases40_49": tuple(range(40, 50)),
    "cases40_59": tuple(range(40, 60)),
    "cases60_79": tuple(range(60, 80)),
    "cases80_89": tuple(range(80, 90)),
    "cases40_89": tuple(range(40, 90)),
}


def load_json(path: Path) -> dict:
    def reject_constant(value: str):
        raise ValueError(f"non-finite JSON constant {value!r} in {path}")

    value = json.loads(path.read_text(), parse_constant=reject_constant)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path}")
    return value


def require_close(actual, expected, label: str, *, atol: float = 1e-13) -> None:
    try:
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=atol)
    except AssertionError as error:
        raise ValueError(f"{label} mismatch: {error}") from error


def require_finite(value, label: str) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if isinstance(item, (dict, list, tuple, int, float)):
                require_finite(item, f"{label}.{key}")
        return
    array = np.asarray(value, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError(f"{label} is not finite")


def validate_case_reports(result: dict, audit: dict, cases: tuple[int, ...]) -> None:
    result_reports = result["case_reports"]
    audit_reports = audit["case_reports"]
    if set(result_reports) != {str(case) for case in cases}:
        raise ValueError("result case reports do not cover the audited cases exactly")
    if result.get("anchor_sha256_by_case") != audit["anchor_sha256_by_case"]:
        raise ValueError("result and measured audit use different g=0 anchors")
    for case in cases:
        key = str(case)
        got = result_reports[key]
        expected = audit_reports[key]
        comparisons = {
            "anchor_rows": got["anchor_rows"],
            "matched_usable_rows": got["matched_usable_rows"],
            "plus_raw_detected": got["plus"]["anchor_raw_detected"],
            "minus_raw_detected": got["minus"]["anchor_raw_detected"],
            "plus_usable": got["plus"]["anchor_usable"],
            "minus_usable": got["minus"]["anchor_usable"],
            "plus_only_usable": got["unmatched_usable_rows"]["plus_only"],
            "minus_only_usable": got["unmatched_usable_rows"]["minus_only"],
            "plus_cross_without_shape": got["plus"]["cross_without_shape"],
            "minus_cross_without_shape": got["minus"]["cross_without_shape"],
        }
        for field, value in comparisons.items():
            if value != expected[field]:
                raise ValueError(f"case {case} {field} differs from measured audit")


def validate_sufficient(result: dict, audit: dict, cases: tuple[int, ...]) -> None:
    measured = result["per_case_sufficient"]["measured"]
    expected = audit["per_case_sufficient"]
    if set(measured) != {str(case) for case in cases}:
        raise ValueError("measured sufficient statistics have the wrong case set")
    for case in cases:
        key = str(case)
        for branch in ("matched_usable", "actual_usable_flags"):
            for direction in ("plus", "minus"):
                got = measured[key][branch][direction]
                want = expected[key][branch][direction]
                if got["denominator"] != want["denominator"]:
                    raise ValueError(
                        f"case {case} {branch} {direction} denominator mismatch"
                    )
                require_close(
                    got["numerator"],
                    want["numerator"],
                    f"case {case} {branch} {direction} numerator",
                    atol=0.0,
                )
        for direction in ("plus", "minus"):
            got = measured[key]["modeled_usable"][direction]
            want = measured[key]["actual_usable_flags"][direction]
            if got["denominator"] != want["denominator"]:
                raise ValueError("modeled comparator denominator is not actual usability")
            require_close(
                got["numerator"],
                want["numerator"],
                "modeled comparator numerator",
                atol=0.0,
            )


def validate_domain(result: dict, audit: dict) -> None:
    domain = result["domain"]
    if domain.get("truth_analysis_cut") is not None:
        raise ValueError("truth analysis cut must be absent")
    if domain.get("sheared_leg_magnitude_radius_recut") is not False:
        raise ValueError("sheared-leg magnitude/radius recut must be false")
    if "MAG_AUTO<25.8" not in domain.get("anchor", ""):
        raise ValueError("g=0 magnitude anchor is missing from result domain")
    if "FLUX_RADIUS>3.0" not in domain.get("anchor", ""):
        raise ValueError("g=0 radius anchor is missing from result domain")
    if result.get("common_antithetic_latents_across_legs_and_models") is not True:
        raise ValueError("common antithetic latent declaration is absent")
    if result.get("h") != audit.get("h"):
        raise ValueError("result and audit use different shear steps")
    if not isinstance(result.get("draws"), int) or result["draws"] < 2:
        raise ValueError("invalid flow draw count")


def validate_truth_pairing(result: dict, audit: dict, truth: dict) -> None:
    cases = set(result["case_reports"])
    if set(truth.get("case_reports", {})) != cases:
        raise ValueError("truth-pairing audit has the wrong case set")
    if truth.get("h") != result.get("h") or truth.get("truth_analysis_cut") is not None:
        raise ValueError("truth-pairing audit has the wrong shear/domain")
    for case in cases:
        truth_report = truth["case_reports"][case]
        if truth_report.get("missing_anchor_rows") != 0:
            raise ValueError(f"truth-pairing audit misses anchors in case {case}")
        if truth_report.get("anchor_rows") != audit["case_reports"][case]["anchor_rows"]:
            raise ValueError(f"truth-pairing anchor count differs in case {case}")
        if truth_report.get("truth_rows", 0) < truth_report["anchor_rows"]:
            raise ValueError(f"truth-pairing row count is invalid in case {case}")
        digest = truth_report.get("invariant_truth_sha256", "")
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ValueError(f"truth-pairing digest is invalid in case {case}")


def validate_provenance(result: dict) -> None:
    cached_hashes = {}

    def validate_file(record: dict, key: str, hash_key: str) -> None:
        path = Path(record[key])
        if not path.is_file():
            raise FileNotFoundError(path)
        if path not in cached_hashes:
            cached_hashes[path] = file_sha256(path)
        digest = cached_hashes[path]
        if digest != record[hash_key]:
            raise ValueError(f"provenance hash mismatch for {path}")

    for record in result["models"].values():
        validate_file(record, "flow", "flow_sha256")
        validate_file(record, "rblend", "rblend_sha256")
    if len(result.get("classifiers", [])) != 3:
        raise ValueError("expected the fixed three-seed classifier ensemble")
    for record in result["classifiers"]:
        validate_file(record, "path", "sha256")


def review(
    result: dict,
    audit: dict,
    truth: dict,
    expected_models: tuple[str, ...],
    *,
    expected_subsets: dict[str, tuple[int, ...]] | None = None,
    expected_draws: int | None = None,
    expected_sampling_seed: int | None = None,
    expected_bootstrap_replicates: int | None = None,
) -> dict:
    validate_domain(result, audit)
    validate_truth_pairing(result, audit, truth)
    validate_provenance(result)
    cases = tuple(sorted(int(case) for case in audit["case_reports"]))
    if expected_subsets is not None:
        expected_cases = set().union(*map(set, expected_subsets.values()))
        if set(cases) != expected_cases:
            raise ValueError("audited cases do not match the required case windows")
        if set(result["subsets"]) != set(expected_subsets):
            raise ValueError("result does not contain every required case window")
        if set(audit["summaries"]) != set(expected_subsets):
            raise ValueError("measured audit does not contain every required case window")
    if expected_draws is not None and result["draws"] != expected_draws:
        raise ValueError("result uses the wrong number of flow draws")
    if (
        expected_sampling_seed is not None
        and result["sampling_seed"] != expected_sampling_seed
    ):
        raise ValueError("result uses the wrong flow sampling seed")
    validate_case_reports(result, audit, cases)
    validate_sufficient(result, audit, cases)
    model_labels = tuple(result["models"])
    if expected_models and set(model_labels) != set(expected_models):
        raise ValueError(
            f"model set {set(model_labels)} differs from expected {set(expected_models)}"
        )
    predicted = result["per_case_sufficient"]["predicted"]
    if set(predicted) != set(model_labels):
        raise ValueError("predicted sufficient statistics do not match model metadata")

    rows = []
    paired_rows = []
    for subset, summary in result["subsets"].items():
        selected = tuple(int(case) for case in summary["cases"])
        if expected_subsets is not None and selected != expected_subsets[subset]:
            raise ValueError(f"subset {subset} has the wrong cases")
        if summary["n_cases"] != len(selected) or not set(selected) <= set(cases):
            raise ValueError(f"invalid case list for subset {subset}")
        audit_summary = audit["summaries"].get(subset)
        if audit_summary is None or audit_summary["cases"] != list(selected):
            raise ValueError(f"subset {subset} differs from measured audit")
        if set(summary["models"]) != set(model_labels):
            raise ValueError(f"subset {subset} has the wrong model set")
        for model in model_labels:
            if set(summary["models"][model]) != set(BRANCHES):
                raise ValueError(f"subset {subset}, model {model} has wrong branches")
            for branch in BRANCHES:
                entry = summary["models"][model][branch]
                audit_branch = MEASURED_BRANCH[branch]
                measured_response = audit_summary["branches"][audit_branch][
                    "measured_response"
                ]
                require_close(
                    entry["measured_response"],
                    measured_response,
                    f"{subset} {model} {branch} measured response",
                )
                recomputed_prediction = response(
                    pooled_sufficient(predicted[model], selected, branch, "plus"),
                    pooled_sufficient(predicted[model], selected, branch, "minus"),
                    result["h"],
                )
                require_close(
                    entry["predicted_response"],
                    recomputed_prediction,
                    f"{subset} {model} {branch} predicted response",
                )
                recomputed_m = 100.0 * (
                    measured_response[0] / recomputed_prediction[0] - 1.0
                )
                require_close(
                    entry["m_percent"],
                    recomputed_m,
                    f"{subset} {model} {branch} m",
                )
                bootstrap = entry["bootstrap"]
                require_finite(bootstrap, f"{subset} {model} {branch} bootstrap")
                if (
                    expected_bootstrap_replicates is not None
                    and bootstrap["replicates"] != expected_bootstrap_replicates
                ):
                    raise ValueError(
                        f"{subset} {model} {branch} has the wrong bootstrap size"
                    )
                rows.append(
                    {
                        "subset": subset,
                        "n_cases": len(selected),
                        "model": model,
                        "branch": branch,
                        "measured_R11": float(measured_response[0]),
                        "predicted_R11": float(recomputed_prediction[0]),
                        "m_percent": float(recomputed_m),
                        "m_se_percentage_points": float(
                            bootstrap["m_standard_error_percentage_points"]
                        ),
                        "m_ci95_percent": bootstrap["m_ci95_percent"],
                    }
                )

        differences = summary["paired_model_differences"]
        expected_pairs = {
            frozenset((left, right))
            for left_index, left in enumerate(model_labels)
            for right in model_labels[:left_index]
        }
        observed_pairs = set()
        for comparison, branches in differences.items():
            left, separator, right = comparison.partition("_minus_")
            pair = frozenset((left, right))
            if (
                not separator
                or left not in model_labels
                or right not in model_labels
                or left == right
                or pair in observed_pairs
            ):
                raise ValueError(f"invalid paired comparison {comparison}")
            observed_pairs.add(pair)
            if set(branches) != set(BRANCHES):
                raise ValueError(f"paired comparison {comparison} has wrong branches")
            for branch, entry in branches.items():
                point = (
                    summary["models"][left][branch]["m_percent"]
                    - summary["models"][right][branch]["m_percent"]
                )
                require_close(
                    entry["difference_percentage_points"],
                    point,
                    f"{subset} {comparison} {branch} paired point difference",
                )
                require_finite(entry, f"{subset} {comparison} {branch}")
                paired_rows.append(
                    {
                        "subset": subset,
                        "n_cases": len(selected),
                        "left_model": left,
                        "right_model": right,
                        "branch": branch,
                        "difference_percentage_points": float(point),
                        "paired_case_bootstrap_standard_error": float(
                            entry["paired_case_bootstrap_standard_error"]
                        ),
                        "paired_case_bootstrap_ci95": entry[
                            "paired_case_bootstrap_ci95"
                        ],
                    }
                )
        if observed_pairs != expected_pairs:
            raise ValueError(f"subset {subset} has incomplete paired comparisons")

    return {
        "format_version": 1,
        "validation": "passed",
        "source_format_version": result["format_version"],
        "models": list(model_labels),
        "case_count": len(cases),
        "case_range": [min(cases), max(cases)],
        "draws": result["draws"],
        "sampling_seed": result["sampling_seed"],
        "rows": rows,
        "paired_rows": paired_rows,
    }


def markdown_table(reviewed: dict) -> str:
    lines = [
        "# Fixed-g0 ConstGold model comparison",
        "",
        (
            f"Validation: **{reviewed['validation']}**; "
            f"{reviewed['case_count']} cases; {reviewed['draws']} antithetic flow "
            f"draws; seed {reviewed['sampling_seed']}.")
    ]
    by_subset = {}
    for row in reviewed["rows"]:
        by_subset.setdefault(row["subset"], []).append(row)
    for subset, rows in by_subset.items():
        lines.extend(
            [
                "",
                f"## {subset} ({rows[0]['n_cases']} cases)",
                "",
                "| Model | Branch | measured R11 | model R11 | m [%] | SE [pp] | 95% CI [%] |",
                "|---|---|---:|---:|---:|---:|---:|",
            ]
        )
        for row in rows:
            low, high = row["m_ci95_percent"]
            lines.append(
                f"| {row['model']} | {row['branch']} | "
                f"{row['measured_R11']:.7f} | {row['predicted_R11']:.7f} | "
                f"{row['m_percent']:+.4f} | {row['m_se_percentage_points']:.4f} | "
                f"[{low:+.4f}, {high:+.4f}] |"
            )
    paired_by_subset = {}
    for row in reviewed["paired_rows"]:
        paired_by_subset.setdefault(row["subset"], []).append(row)
    for subset, rows in paired_by_subset.items():
        lines.extend(
            [
                "",
                f"## {subset}: paired model differences",
                "",
                "| Left - right | Branch | delta m [pp] | paired SE [pp] | paired 95% CI [pp] |",
                "|---|---|---:|---:|---:|",
            ]
        )
        for row in rows:
            low, high = row["paired_case_bootstrap_ci95"]
            lines.append(
                f"| {row['left_model']} - {row['right_model']} | "
                f"{row['branch']} | {row['difference_percentage_points']:+.4f} | "
                f"{row['paired_case_bootstrap_standard_error']:.4f} | "
                f"[{low:+.4f}, {high:+.4f}] |"
            )
    return "\n".join(lines) + "\n"


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--measured-audit", type=Path, required=True)
    parser.add_argument("--truth-pairing-audit", type=Path, required=True)
    parser.add_argument("--expected-model", action="append", default=[])
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    args = parser.parse_args(argv)
    for output in (args.output_json, args.output_markdown):
        if output.exists():
            raise FileExistsError(f"refusing to overwrite {output}")
    reviewed = review(
        load_json(args.result),
        load_json(args.measured_audit),
        load_json(args.truth_pairing_audit),
        tuple(args.expected_model),
        expected_subsets=EXPECTED_SUBSETS,
        expected_draws=64,
        expected_sampling_seed=7301,
        expected_bootstrap_replicates=10000,
    )
    reviewed["inputs"] = {
        "result": {
            "path": str(args.result.resolve()),
            "sha256": file_sha256(args.result),
        },
        "measured_audit": {
            "path": str(args.measured_audit.resolve()),
            "sha256": file_sha256(args.measured_audit),
        },
        "truth_pairing_audit": {
            "path": str(args.truth_pairing_audit.resolve()),
            "sha256": file_sha256(args.truth_pairing_audit),
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output_json, reviewed)
    temporary = args.output_markdown.with_suffix(args.output_markdown.suffix + ".tmp")
    temporary.write_text(markdown_table(reviewed))
    os.replace(temporary, args.output_markdown)
    print(
        f"CONSTGOLD_FIXED_G0_REVIEW_PASSED rows={len(reviewed['rows'])} "
        f"output={args.output_json}",
        flush=True,
    )


if __name__ == "__main__":
    main()
