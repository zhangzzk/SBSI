#!/usr/bin/env python3
"""Paired case-bootstrap change between two identically defined joint audits."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from plots.constgold_property_bias import bootstrap_response, make_case_weights
from scripts.train_fixed_g0_flow import sha256, write_json


def response_axis(result):
    axis = result.get("shear_axis", 1)
    if axis not in (1, 2) or result.get("response_vector_order", ["e1", "e2"]) != ["e1", "e2"]:
        raise ValueError("axis 1 or 2 with e1,e2 response-vector order required")
    return axis


def require_comparable(baseline, candidate):
    if response_axis(baseline) != response_axis(candidate):
        raise ValueError("non-comparable joint audit: shear_axis")
    for key in ("cases", "h", "draws", "seed", "population", "cohorts", "cuts", "usability", "case_reports"):
        if baseline[key] != candidate[key]:
            raise ValueError(f"non-comparable joint audit: {key}")
    for key in ("parent_manifest_sha256", "classifiers", "rblend"):
        if baseline["inputs"][key] != candidate["inputs"][key]:
            raise ValueError(f"non-comparable joint inputs: {key}")
    if baseline["per_case_sufficient"].keys() != candidate["per_case_sufficient"].keys():
        raise ValueError("different response branches")


def require_latent_repeat(baseline, candidate):
    """Allow only the latent seed to change in a frozen-model repeat."""
    if baseline["seed"] == candidate["seed"]:
        raise ValueError("latent repeat requires distinct seeds")
    require_comparable(baseline, dict(candidate, seed=baseline["seed"]))
    if baseline["inputs"] != candidate["inputs"]:
        raise ValueError("latent repeat requires identical model, data and evaluator provenance")
    left, right = baseline.get("experimental_response"), candidate.get("experimental_response")
    if (left is None) != (right is None):
        raise ValueError("latent repeat requires identical response composition")
    if left is not None:
        for key in set(left) | set(right):
            if key != "case_reports" and left.get(key) != right.get(key):
                raise ValueError(f"latent repeat response differs: {key}")
        if left["case_reports"].keys() != right["case_reports"].keys():
            raise ValueError("latent repeat response cases differ")
        for case in left["case_reports"]:
            if left["case_reports"][case]["pairs"] != right["case_reports"][case]["pairs"]:
                raise ValueError("latent repeat pair provenance differs")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--independent-latent-repeat", action="store_true",
                        help="Require identical frozen models/data and different latent seeds")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    base = json.loads(args.baseline.read_text())
    candidate = json.loads(args.candidate.read_text())
    if args.independent_latent_repeat:
        require_latent_repeat(base, candidate)
    else:
        require_comparable(base, candidate)
    if len(base["cases"]) < 2 or base.get("operational_single_case", False) or candidate.get("operational_single_case", False):
        raise ValueError("paired case-bootstrap comparison requires at least two nonoperational cases")
    axis = response_axis(base)
    weights = make_case_weights(10000, len(base["cases"]), 20260914)
    result = {}
    for key in base["per_case_sufficient"]:
        draws, measured_reference = {}, None
        for arm, data in (("baseline", base), ("candidate", candidate)):
            raw = data["per_case_sufficient"][key]
            stats = {source: {direction: {field: np.asarray(value) for field, value in values.items()}
                              for direction, values in directions.items()} for source, directions in raw.items()}
            if measured_reference is not None:
                for direction in ("plus", "minus"):
                    for field in ("numerator", "denominator"):
                        np.testing.assert_array_equal(stats["measured"][direction][field], measured_reference[direction][field])
            measured_reference = stats["measured"]
            measured = bootstrap_response(stats["measured"]["plus"], stats["measured"]["minus"], weights, base["h"])[:, 0, axis-1]
            predicted = bootstrap_response(stats["model"]["plus"], stats["model"]["minus"], weights, base["h"])[:, 0, axis-1]
            draws[arm] = 100 * (measured / predicted - 1)
            if not np.isfinite(draws[arm]).all():
                raise ValueError("nonfinite bias replicas")
        delta = draws["candidate"] - draws["baseline"]
        result[key] = {"baseline": base["summary"][key], "candidate": candidate["summary"][key],
                       "delta_m_percentage_points": candidate["summary"][key]["m_percent"] - base["summary"][key]["m_percent"],
                       "paired_delta_se_pp": float(delta.std(ddof=1)), "paired_delta_ci95_pp": np.quantile(delta, [.025, .975])}
        if key.startswith("truth_parent/") and key.endswith("actual_usable_flags"):
            print(f"PAIRED_JOINT_CHANGE {key} delta={result[key]['delta_m_percentage_points']:+.6f} "
                  f"se={result[key]['paired_delta_se_pp']:.6f}pp", flush=True)
    write_json(args.output, {"comparisons": result, "shear_axis": axis,
                            "diagonal_response_component": f"R{axis}{axis}", "script_sha256": sha256(__file__), "baseline_sha256": sha256(args.baseline),
                            "candidate_sha256": sha256(args.candidate),
                            "comparison_kind": "independent_latent_repeat" if args.independent_latent_repeat else "paired_model_change",
                            "limitations": (["same frozen model and measurements; only latent integration stream differs",
                                             "paired case bootstrap describes repeat differences; two streams do not bound finite-draw bias or training uncertainty",
                                             "no favorable stream selection or calibration claim"] if args.independent_latent_repeat else
                                            ["paired case sampling uncertainty only; excludes training and latent MC uncertainty",
                                             "reused development scenes; no checkpoint selection or acceptance on these biases"])})


if __name__ == "__main__":
    main()
