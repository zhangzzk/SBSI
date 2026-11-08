#!/usr/bin/env python
"""Verify the frozen candidate's existing cases140--239 g1 response gate."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np


CASES = tuple(range(140, 240))
QMC_SEEDS = tuple(range(7301, 7317))
FLOW_PATH = Path(
    "/project/ls-gruen/users/zekang.zhang/sbsi_caches/"
    "minimal_rblend_flow_condition_v1/training/flow8_rblend/epoch016.pt"
)
FLOW_SHA256 = "52caed1c0d955693d0b0825688f4ef344d10f731b5b6813d146e153564b02db0"
DETECTOR_ROOT = Path(
    "/project/ls-gruen/users/zekang.zhang/sbsi_caches/"
    "minimal_coherent_u_flow_rblend_ablation_v2/models/flow8_rblend"
)
DETECTOR_PATHS = tuple(
    DETECTOR_ROOT / f"seed{seed}/selected.pt"
    for seed in (20260913, 20260914, 20260915)
)
DETECTOR_HASHES = (
    "bbbd27a86c18bffb3e8b98e46c455ebcd2a27844ead363913bb5289ae19e6685",
    "031498aa4140be4d2dfc7605f5211a1d2379e14c16f3fbc23b79f2f7f9e028cd",
    "dab1e4cddf3b61da4cc6dee810091209b29df4dbf5dcc86cd20d3e23eebbd702",
)
FLOW_FEATURES = (
    "e1_input_p",
    "e2_input_p",
    "sersic_n_input_p",
    "r_input_p",
    "circularized_Re_input_p",
    "nbr_flux_near",
    "nbr_flux_far",
    "nbr_flux_max",
    "R_blend",
)
CATALOGUE_ROOT = Path(
    "/project/ls-gruen/users/zekang.zhang/"
    "lsst_sims_fs2_25876_constant_g005_c140-239"
)
EXPERIMENT_ROOT = Path(
    "/project/ls-gruen/users/zekang.zhang/sbsi_caches/"
    "minimal_rblend_flow_condition_v1/existing_g005_c140_239"
)
LOOKUP_PATHS = tuple(
    EXPERIMENT_ROOT / "lookup" / f"rblend_trial9_c{first}-{first + 9}.feather"
    for first in range(140, 240, 10)
)
CROWD_LOOKUP = Path(
    "/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/"
    "v22_g005_c140-239/crowd_flux_conc_c140-239.feather"
)


def _sha256(path: Path) -> str:
    value = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def _load(path: Path) -> dict:
    def reject(value: str):
        raise ValueError(f"non-finite JSON constant {value} in {path}")

    return json.loads(path.read_text(), parse_constant=reject)


def _require_identity(result: dict, prior: dict) -> None:
    if result.get("analysis") != "ConstGold complete forward likelihood response":
        raise RuntimeError("wrong response analysis")
    if (
        result.get("shear_component") != "g1"
        or result.get("cases") != list(CASES)
        or float(result.get("h", np.nan)) != 0.05
        or result.get("analysis_primary_re_min") is not None
    ):
        raise RuntimeError("response cases, shear, or truth support drifted")
    cut = result.get("measured_output_cut", {})
    if (
        cut.get("ellipticity_norm_max") is not None
        or float(cut.get("measured_mag_auto_max", np.nan)) != 25.8
        or float(cut.get("measured_radius_arcsec_min", np.nan)) != 0.75
    ):
        raise RuntimeError("response is not the frozen no-|e| sample")
    sampling = result.get("sampling", {})
    if (
        sampling.get("qmc") is not True
        or sampling.get("common_random_numbers_between_legs") is not True
        or sampling.get("draws_per_atom_per_flow_seed") != 64
        or sampling.get("seeds") != list(QMC_SEEDS)
    ):
        raise RuntimeError("response QMC protocol drifted")

    models = result.get("models", [])
    if len(models) != len(QMC_SEEDS):
        raise RuntimeError("response does not contain all QMC runs")
    for model, seed in zip(models, QMC_SEEDS):
        if (
            Path(model.get("path", "")) != FLOW_PATH
            or model.get("sha256") != FLOW_SHA256
            or model.get("sampling_seed") != seed
        ):
            raise RuntimeError("measurement-flow identity drifted")
    detector = result.get("detection_model", {})
    if (
        detector.get("aggregation") != "arithmetic_mean_probability"
        or tuple(map(Path, detector.get("classifiers", []))) != DETECTOR_PATHS
        or tuple(detector.get("classifier_sha256", [])) != DETECTOR_HASHES
        or tuple(detector.get("features", [])) != FLOW_FEATURES
        or detector.get("neighbour_selection") != "nearest"
        or float(detector.get("impact_exponent", np.nan)) != 2.0
    ):
        raise RuntimeError("detector ensemble identity drifted")
    inputs = result.get("inputs", {})
    if (
        Path(inputs.get("constgold_root", "")) != CATALOGUE_ROOT
        or Path(inputs.get("crowd_lookup", "")) != CROWD_LOOKUP
        or tuple(map(Path, inputs.get("rblend_lookup", []))) != LOOKUP_PATHS
    ):
        raise RuntimeError("existing catalogue inputs drifted")
    if (
        prior.get("analysis") != "minimal shared-R_blend reused response decision"
        or prior.get("passes_reused_structural_screen") is not True
        or prior.get("decision") != "survives_reused_screen_only"
    ):
        raise RuntimeError("prior g1=+/-0.02 structural gate did not pass")


def summarize(result: dict, prior: dict) -> dict:
    _require_identity(result, prior)
    measured = np.asarray(
        result["pooled"]["measured"]["complete"]["response_first_column"],
        dtype=np.float64,
    )
    model_record = result["pooled"]["predicted"]["complete"]
    model = np.asarray(model_record["response_first_column"], dtype=np.float64)
    truth_record = result["pooled"]["truth_shape_population"]
    truth_actual = np.asarray(
        truth_record["actual_detection_and_selection"]["response_first_column"],
        dtype=np.float64,
    )
    truth_model = np.asarray(
        truth_record["predicted_detection_and_selection"]["response_first_column"],
        dtype=np.float64,
    )
    by_qmc = np.asarray(model_record["response_by_sampling_seed"], dtype=np.float64)
    values = np.concatenate(
        (measured, model, truth_actual, truth_model, by_qmc.ravel())
    )
    if (
        measured.shape != (2,)
        or model.shape != (2,)
        or truth_actual.shape != (2,)
        or truth_model.shape != (2,)
        or by_qmc.shape != (len(QMC_SEEDS), 2)
        or not np.isfinite(values).all()
        or model[0] <= 0.0
        or (by_qmc[:, 0] <= 0.0).any()
    ):
        raise RuntimeError("malformed or non-finite response arrays")

    total_error = model - measured
    truth_error = truth_model - truth_actual
    residual_error = total_error - truth_error
    m_percent = 100.0 * (measured[0] / model[0] - 1.0)
    contributions = {
        "truth_population": float(-100.0 * truth_error[0] / model[0]),
        "measurement_residual": float(-100.0 * residual_error[0] / model[0]),
        "total": float(-100.0 * total_error[0] / model[0]),
    }
    if not np.isclose(contributions["total"], m_percent, rtol=1e-12, atol=1e-12):
        raise RuntimeError("response decomposition does not close")

    acceptance = result.get("response_acceptance", {})
    case_ci = np.asarray(acceptance.get("case_bootstrap_ci95_percent"), dtype=float)
    case_se = float(acceptance.get("case_bootstrap_standard_error_percent", np.nan))
    if (
        case_ci.shape != (2,)
        or not np.isfinite(case_ci).all()
        or case_ci[0] >= case_ci[1]
        or not np.isfinite(case_se)
        or case_se < 0.0
        or not np.isclose(
            float(acceptance.get("m_percent", np.nan)),
            m_percent,
            rtol=1e-12,
            atol=1e-12,
        )
    ):
        raise RuntimeError("malformed or inconsistent response acceptance")
    ci_pass = bool(case_ci[0] > -0.3 and case_ci[1] < 0.3)
    if acceptance.get("passes_marginal_ci95_equivalence") is not ci_pass:
        raise RuntimeError("stored and reconstructed CI gate disagree")

    m_by_qmc = 100.0 * (measured[0] / by_qmc[:, 0] - 1.0)
    qmc_se = float(m_by_qmc.std(ddof=1) / np.sqrt(len(m_by_qmc)))
    gates = {
        "prior_g1_h002_structural_gate": True,
        "case_ci95_strictly_inside_0p3": ci_pass,
        "central_abs_m_below_0p3": bool(abs(m_percent) < 0.3),
        "truth_component_abs_below_0p3": bool(
            abs(contributions["truth_population"]) < 0.3
        ),
        "residual_component_abs_below_0p3": bool(
            abs(contributions["measurement_residual"]) < 0.3
        ),
        "qmc_standard_error_at_most_0p005": bool(qmc_se <= 0.005),
    }
    passed = all(gates.values())
    prior_response = prior["response"]
    return {
        "schema_version": 1,
        "analysis": "minimal shared-R_blend existing g1 amplitude decision",
        "existing_catalogue_generalization_not_fresh_multi_axis_promotion": True,
        "sample": (
            "cases140--239; g1 +/-0.05; 18<r<25.8; 0.37<Re<1.5; "
            "MAG_AUTO<25.8; FLUX_RADIUS>=0.75arcsec; no measured-|e| cut"
        ),
        "response": {
            "measured_R11": float(measured[0]),
            "model_R11": float(model[0]),
            "m_percent": float(m_percent),
            "case_standard_error_percent": case_se,
            "case_ci95_percent": case_ci.tolist(),
            "qmc_standard_error_percent": qmc_se,
            "m_by_qmc_seed_percent": m_by_qmc.tolist(),
        },
        "model_minus_image_response": total_error.tolist(),
        "truth_population_response_error": truth_error.tolist(),
        "measurement_residual_response_error": residual_error.tolist(),
        "m_contribution_percent": contributions,
        "gates": gates,
        "passes_existing_g1_endpoints": passed,
        "decision": (
            "survives_existing_g1_endpoints_only"
            if passed
            else "reject_shared_rblend_candidate_without_new_simulation"
        ),
        "prior_g1_h002": {
            "m_percent": prior_response["m_percent"],
            "case_ci95_percent": prior_response["case_ci95_percent"],
            "truth_contribution_percent": prior["m_contribution_percent"][
                "truth_population"
            ],
            "residual_contribution_percent": prior["m_contribution_percent"][
                "measurement_residual"
            ],
        },
        "next_action": (
            "keep the candidate frozen and prepare only the missing population-level g2 and g0 acceptance evidence"
            if passed
            else "do not simulate; diagnose the smallest prespecified model escalation without response tuning"
        ),
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--response", type=Path, required=True)
    parser.add_argument("--prior-decision", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    result = summarize(_load(args.response), _load(args.prior_decision))
    result["input_artifacts"] = {
        "response": {
            "path": str(args.response.resolve()),
            "sha256": _sha256(args.response),
        },
        "prior_decision": {
            "path": str(args.prior_decision.resolve()),
            "sha256": _sha256(args.prior_decision),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "decision",
                    "passes_existing_g1_endpoints",
                    "response",
                    "m_contribution_percent",
                    "gates",
                    "next_action",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
