#!/usr/bin/env python3
"""Freeze and reduce the reused response screen for the selected-shape flow."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np


CASES = tuple(range(120, 140))
SEEDS = (7301, 7302, 7303, 7304)
FLOW_FEATURES = (
    "e1_input_p", "e2_input_p", "sersic_n_input_p", "r_input_p",
    "circularized_Re_input_p", "nbr_flux_near", "nbr_flux_far",
    "nbr_flux_max", "R_blend",
)


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8_388_608), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: str | Path, value: dict) -> None:
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def load_protocol(root: str | Path, verify=False) -> dict:
    root = Path(root).resolve()
    protocol = json.loads((root / "protocol.json").read_text())
    if verify:
        for name, value in protocol["inputs"].items():
            if sha256(value) != protocol["input_hashes"][name]:
                raise RuntimeError(f"response-screen input changed: {value}")
    return protocol


def init(args):
    root = Path(args.root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    (root / "logs").mkdir(exist_ok=True)
    (root / "response").mkdir(exist_ok=True)
    if (root / "protocol.json").exists():
        raise FileExistsError("preserving existing response protocol")
    repository = Path(__file__).resolve().parents[1]
    lookup_root = Path(args.lookup_root).resolve()
    inputs = {
        "baseline_flow": Path(args.baseline_flow).resolve(),
        "candidate_flow": Path(args.candidate_flow).resolve(),
        "candidate_density_decision": Path(args.candidate_density_decision).resolve(),
        "crowd_lookup": Path(args.crowd_lookup).resolve(),
        "response_adapter": repository / "scripts/evaluate_constgold_selected_shape_response.py",
        "base_evaluator": repository / "scripts/evaluate_constgold_complete_forward_response.py",
        "reviewer": Path(__file__).resolve(),
        "job_script": Path(args.job_script).resolve(),
    }
    for index, detector in enumerate(args.detector):
        inputs[f"detector{index}"] = Path(detector).resolve()
    for case in CASES:
        inputs[f"rblend{case}"] = lookup_root / f"case{case}/rblend.feather"
        inputs[f"receipt{case}"] = lookup_root / f"case{case}/receipt.json"
    inputs.update({f"sbsi/{path.name}": path for path in sorted((repository / "sbsi").glob("*.py"))})
    if any(not path.is_file() for path in inputs.values()):
        raise FileNotFoundError("response-screen input is missing")
    density_decision = json.loads(inputs["candidate_density_decision"].read_text())
    if not density_decision.get("passes_response_blind_density_readiness"):
        raise RuntimeError("candidate density did not pass its response-blind gate")
    if density_decision["selected_sha256"] != sha256(inputs["candidate_flow"]):
        raise RuntimeError("candidate density decision and flow disagree")
    protocol = {
        "purpose": "reused cases120-139 mechanism screen of p(e1,e2|S,x), not promotion",
        "inputs": {name: str(path) for name, path in inputs.items()},
        "input_hashes": {name: sha256(path) for name, path in inputs.items()},
        "arms": {
            "baseline": str(inputs["baseline_flow"]),
            "selected_shape": str(inputs["candidate_flow"]),
        },
        "catalogues": {
            "root": str(Path(args.catalogue_root).resolve()),
            "lookup_root": str(lookup_root),
            "cases": list(CASES),
            "existing_only": True,
            "new_image_simulation": False,
            "new_measurement": False,
        },
        "sample": {
            "mag_auto": "<25.8",
            "flux_radius": ">=0.75arcsec convolved SExtractor circular half-light radius",
            "ellipticity_norm_cut": None,
        },
        "sampling": {
            "h": 0.02,
            "draws_per_row_per_seed": 256,
            "qmc_seeds": list(SEEDS),
            "common_random_numbers_between_shear_legs": True,
        },
        "factorization": "direct p(S|x) times conditional selected-shape mean",
        "gate": {
            "required": [
                "candidate absolute central m is smaller than baseline",
                "candidate absolute residual contribution is smaller than baseline",
                "candidate central m and both component contributions are each inside +/-0.3pp",
                "candidate QMC SE <=0.005pp and selected-shape support bound is zero",
            ],
            "interpretation": "survival is a reused mechanism result only; failure stops this minimal branch",
        },
    }
    write_json(root / "protocol.json", protocol)
    load_protocol(root, verify=True)
    print("MINIMAL_SELECTED_SHAPE_RESPONSE_PROTOCOL_READY", sha256(root / "protocol.json"), flush=True)


def _summary(result: dict) -> dict:
    if (
        result.get("analysis") != "ConstGold complete forward likelihood response"
        or result.get("cases") != list(CASES)
        or result.get("shear_component") != "g1"
        or float(result.get("h", np.nan)) != 0.02
        or result.get("probability_factorization") != "direct_selected"
    ):
        raise RuntimeError("response identity drifted")
    cut = result["measured_output_cut"]
    if (
        cut.get("ellipticity_norm_max") is not None
        or float(cut["measured_mag_auto_max"]) != 25.8
        or float(cut["measured_radius_arcsec_min"]) != 0.75
    ):
        raise RuntimeError("response sample drifted")
    measured = np.asarray(result["pooled"]["measured"]["complete"]["response_first_column"], dtype=float)
    predicted_record = result["pooled"]["predicted"]["complete"]
    predicted = np.asarray(predicted_record["response_first_column"], dtype=float)
    truth = result["pooled"]["truth_shape_population"]
    truth_actual = np.asarray(truth["actual_detection_and_selection"]["response_first_column"], dtype=float)
    truth_model = np.asarray(truth["predicted_detection_and_selection"]["response_first_column"], dtype=float)
    by_seed = np.asarray(predicted_record["response_by_sampling_seed"], dtype=float)
    if by_seed.shape != (len(SEEDS), 2) or predicted[0] <= 0 or not np.isfinite(by_seed).all():
        raise RuntimeError("invalid response arrays")
    total_error = predicted - measured
    truth_error = truth_model - truth_actual
    residual_error = total_error - truth_error
    m = 100.0 * (measured[0] / predicted[0] - 1.0)
    contributions = {
        "truth_population": float(-100.0 * truth_error[0] / predicted[0]),
        "measurement_residual": float(-100.0 * residual_error[0] / predicted[0]),
        "total": float(-100.0 * total_error[0] / predicted[0]),
    }
    if not np.isclose(contributions["total"], m, rtol=1e-12, atol=1e-12):
        raise RuntimeError("response decomposition does not close")
    m_by_seed = 100.0 * (measured[0] / by_seed[:, 0] - 1.0)
    support = result["direct_selected_conditional_support"]
    return {
        "measured_R11": float(measured[0]),
        "model_R11": float(predicted[0]),
        "m_percent": float(m),
        "case_standard_error_percent": float(result["response_acceptance"]["case_bootstrap_standard_error_percent"]),
        "case_ci95_percent": result["response_acceptance"]["case_bootstrap_ci95_percent"],
        "qmc_standard_error_percent": float(m_by_seed.std(ddof=1) / np.sqrt(len(m_by_seed))),
        "m_by_qmc_seed_percent": m_by_seed.tolist(),
        "contributions_percent": contributions,
        "model_minus_image_response": total_error.tolist(),
        "truth_population_response_error": truth_error.tolist(),
        "measurement_residual_response_error": residual_error.tolist(),
        "support_response_abs_bound": support["pooled_ensemble_response_abs_bound"],
    }


def review(args):
    root = Path(args.root).resolve()
    protocol = load_protocol(root, verify=True)
    summaries = {}
    raw = {}
    for arm in ("baseline", "selected_shape"):
        path = root / f"response/{arm}.json"
        raw[arm] = json.loads(path.read_text())
        summaries[arm] = _summary(raw[arm])
        models = raw[arm]["models"]
        if len(models) != len(SEEDS) or any(
            Path(item["path"]) != Path(protocol["arms"][arm])
            or item["sha256"] != protocol["input_hashes"][f"{arm if arm == 'baseline' else 'candidate'}_flow"]
            for item in models
        ):
            raise RuntimeError(f"{arm} flow identity drifted")
    np.testing.assert_allclose(
        summaries["baseline"]["measured_R11"], summaries["selected_shape"]["measured_R11"], rtol=0, atol=0
    )
    baseline = summaries["baseline"]
    candidate = summaries["selected_shape"]
    gates = {
        "absolute_m_improves": abs(candidate["m_percent"]) < abs(baseline["m_percent"]),
        "absolute_residual_contribution_improves": abs(candidate["contributions_percent"]["measurement_residual"]) < abs(baseline["contributions_percent"]["measurement_residual"]),
        "central_abs_m_below_0p3": abs(candidate["m_percent"]) < 0.3,
        "truth_component_abs_below_0p3": abs(candidate["contributions_percent"]["truth_population"]) < 0.3,
        "residual_component_abs_below_0p3": abs(candidate["contributions_percent"]["measurement_residual"]) < 0.3,
        "qmc_standard_error_at_most_0p005": candidate["qmc_standard_error_percent"] <= 0.005,
        "selected_shape_support_bound_is_zero": float(candidate["support_response_abs_bound"][0]) == 0.0,
    }
    passed = all(gates.values())
    decision = {
        "screen_only_not_promotion": True,
        "cases": list(CASES),
        "sample": protocol["sample"],
        "baseline": baseline,
        "selected_shape": candidate,
        "gates": gates,
        "passes_reused_mechanism_screen": passed,
        "decision": "retain_minimal_candidate_for_new_independent_evidence" if passed else "stop_minimal_selected_shape_branch",
        "new_image_simulation": False,
        "limitations": [
            "cases120-139 are reused development response cases",
            "the cases140-239 endpoint is already consumed",
            "this result cannot establish fresh or multi-axis 0.3% acceptance",
        ],
    }
    write_json(root / "decision.json", decision)
    print("MINIMAL_SELECTED_SHAPE_RESPONSE_REVIEW", json.dumps(decision), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    initial = subs.add_parser("init")
    initial.add_argument("--root", required=True)
    initial.add_argument("--catalogue-root", required=True)
    initial.add_argument("--lookup-root", required=True)
    initial.add_argument("--crowd-lookup", required=True)
    initial.add_argument("--baseline-flow", required=True)
    initial.add_argument("--candidate-flow", required=True)
    initial.add_argument("--candidate-density-decision", required=True)
    initial.add_argument("--detector", action="append", required=True)
    initial.add_argument("--job-script", required=True)
    reviewer = subs.add_parser("review")
    reviewer.add_argument("--root", required=True)
    args = parser.parse_args()
    {"init": init, "review": review}[args.command](args)


if __name__ == "__main__":
    main()
