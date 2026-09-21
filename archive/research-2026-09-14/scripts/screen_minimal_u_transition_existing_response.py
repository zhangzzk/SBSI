#!/usr/bin/env python3
"""Paired cached response screen of the clean nine-input U-transition ensemble."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys

import numpy as np

from sbsi.selection_model import load_selection_model


CASES = tuple(range(120, 140))
CUT_SLOT = 1
MOMENT_CUT = 6
BOOTSTRAP_SEED = 20260913
BOOTSTRAP_REPLICATES = 10_000
BASE_ROOT = Path(
    "/project/ls-gruen/users/zekang.zhang/sbsi_caches/"
    "coherent_u_crowd_ensemble_v1"
)
CURRENT_ROOT = Path(
    "/project/ls-gruen/users/zekang.zhang/sbsi_caches/"
    "minimal_coherent_u_flow_rblend_ablation_v2"
)
PROPOSED_ROOT = Path(
    "/project/ls-gruen/users/zekang.zhang/sbsi_caches/"
    "minimal_coherent_u_transition_lambda1_v1"
)
CURRENT_HASHES = (
    "bbbd27a86c18bffb3e8b98e46c455ebcd2a27844ead363913bb5289ae19e6685",
    "031498aa4140be4d2dfc7605f5211a1d2379e14c16f3fbc23b79f2f7f9e028cd",
    "dab1e4cddf3b61da4cc6dee810091209b29df4dbf5dcc86cd20d3e23eebbd702",
)
PROPOSED_HASHES = (
    "36a03d3ec192a962d64ce2bef1b8f715ff7490f4eea22eddd46fa0ec1c745e59",
    "0daaa2c04b1aa89a193f05256f11706506f64972407cc40a43fc3de4ab88a120",
    "bdb52b894abe2319d006e7a357ae9018360c9422eb91f4b46e9acb800c5cb2d0",
)
FEATURES = (
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


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import frozen module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def aggregate_metrics(values: list[np.ndarray]) -> dict:
    total = np.asarray(values).sum(axis=(0, 1))
    return {
        "n_leg_rows": int(total[0]),
        "observed_usable_fraction": float(total[1] / total[0]),
        "predicted_usable_fraction": float(total[2] / total[0]),
        "signed_probability_gap": float((total[2] - total[1]) / total[0]),
        "bce": float(total[3] / total[0]),
        "brier": float(total[4] / total[0]),
    }


def component_arrays(actual_r: np.ndarray, model_r: np.ndarray) -> dict[str, np.ndarray]:
    return {
        kind: -100.0 * (model_r[:, index, 0] - actual_r[:, index, 0])
        / model_r[:, 0, 0]
        for index, kind in ((1, "truth"), (2, "residual"))
    }


def run(output: Path, device: str) -> None:
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("the catalogue screen must run through Slurm")
    research_scripts = Path(__file__).resolve().parent
    helpers = load_module(
        research_scripts / "screen_transition_aware_existing_response.py",
        "minimal_u_transition_screen_helpers",
    )
    base = load_module(BASE_ROOT / "worker.py", "minimal_u_transition_cached_base")
    base_config = base.verify()
    candidate = load_module(
        research_scripts / "train_minimal_coherent_u_rblend_ablation.py",
        "minimal_u_transition_candidate_source",
    )
    candidate_protocol = helpers.load_candidate_protocol(candidate)
    training = load_module(
        research_scripts / "train_minimal_u_transition.py",
        "minimal_u_transition_training_source",
    )
    proposed_protocol = training.load_protocol(PROPOSED_ROOT)
    if tuple(proposed_protocol["features"]) != FEATURES:
        raise RuntimeError("proposed feature contract changed")
    review_path = PROPOSED_ROOT / "review.json"
    review = json.loads(review_path.read_text())
    if review.get("response_read") is not False:
        raise RuntimeError("training review read a response")

    current_paths = tuple(
        CURRENT_ROOT / f"models/flow8_rblend/seed{seed}/selected.pt"
        for seed in (20260913, 20260914, 20260915)
    )
    proposed_paths = tuple(
        PROPOSED_ROOT / f"models/seed{seed}/selected.pt"
        for seed in (20260913, 20260914, 20260915)
    )
    if tuple(helpers.digest(path) for path in current_paths) != CURRENT_HASHES:
        raise RuntimeError("current checkpoint hash mismatch")
    if tuple(helpers.digest(path) for path in proposed_paths) != PROPOSED_HASHES:
        raise RuntimeError("proposed checkpoint hash mismatch")
    current_members = [load_selection_model(path, device=device) for path in current_paths]
    proposed_members = [load_selection_model(path, device=device) for path in proposed_paths]
    for member in (*current_members, *proposed_members):
        if tuple(member.preprocessor.feature_names) != FEATURES:
            raise RuntimeError("classifier feature contract mismatch")

    actual_by_case = []
    oracle_by_case = []
    current_by_case = []
    proposed_by_case = []
    current_metrics = []
    proposed_metrics = []
    source_receipts = []
    for case in CASES:
        data, _unused_frames, prepared_receipt, moments, moment_receipt = (
            base.source_case(base_config, case, with_moments=True)
        )
        ids, frames, usable, _candidate_receipt = candidate.verified_case(
            candidate_protocol, case
        )
        np.testing.assert_array_equal(ids, data["ids"])
        np.testing.assert_array_equal(usable, data["U"])

        probabilities = {}
        for name, members in (
            ("current", current_members),
            ("proposed", proposed_members),
        ):
            probabilities[name] = np.mean(
                np.stack(
                    [
                        np.stack(
                            [
                                member.predict_proba(frame, batch_size=65_536)
                                for frame in frames
                            ]
                        )
                        for member in members
                    ]
                ),
                axis=0,
            )
            if probabilities[name].shape != data["U"].shape:
                raise RuntimeError(f"probability shape mismatch: {name} case={case}")

        _, cached = helpers.verify_part(BASE_ROOT / "parts" / f"case{case:03d}.npz")
        actual_by_case.append(cached["actual"][CUT_SLOT])
        oracle_by_case.append(cached["model"][0, CUT_SLOT])
        current_by_case.append(
            base.model_statistics(
                probabilities["current"],
                moments["q"][MOMENT_CUT],
                moments["u"][MOMENT_CUT],
                data["es"],
            )
        )
        proposed_by_case.append(
            base.model_statistics(
                probabilities["proposed"],
                moments["q"][MOMENT_CUT],
                moments["u"][MOMENT_CUT],
                data["es"],
            )
        )
        current_metrics.append(base.metric_sufficient(data["U"], probabilities["current"]))
        proposed_metrics.append(base.metric_sufficient(data["U"], probabilities["proposed"]))
        source_receipts.append(
            {
                "case": case,
                "prepared_receipt": str(prepared_receipt),
                "prepared_receipt_sha256": helpers.digest(prepared_receipt),
                "moment_receipt": str(moment_receipt),
                "moment_receipt_sha256": helpers.digest(moment_receipt),
            }
        )
        print(f"MINIMAL_U_TRANSITION_SCREEN_CASE case={case}", flush=True)

    actual = np.asarray(actual_by_case)
    oracle = np.asarray(oracle_by_case)
    current = np.asarray(current_by_case)
    proposed = np.asarray(proposed_by_case)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    weights = np.vstack(
        (
            np.ones((1, len(CASES)), dtype=int),
            rng.multinomial(
                len(CASES),
                np.full(len(CASES), 1.0 / len(CASES)),
                BOOTSTRAP_REPLICATES,
            ),
        )
    )
    actual_r = base.responses(actual, weights)
    oracle_r = base.responses(oracle, weights)
    current_r = base.responses(current, weights)
    proposed_r = base.responses(proposed, weights)
    current_result = helpers.response_entry(base, actual_r, current_r)
    proposed_result = helpers.response_entry(base, actual_r, proposed_r)
    oracle_result = helpers.response_entry(base, actual_r, oracle_r)
    current_m = 100.0 * (actual_r[:, 0, 0] / current_r[:, 0, 0] - 1.0)
    proposed_m = 100.0 * (actual_r[:, 0, 0] / proposed_r[:, 0, 0] - 1.0)
    current_components = component_arrays(actual_r, current_r)
    proposed_components = component_arrays(actual_r, proposed_r)
    proposed_centers = proposed_result["component_contribution_to_m_pp"]

    result = {
        "analysis": "reused cached-moment response screen of the clean nine-input U-transition ensemble",
        "screen_only_not_promotion": True,
        "cases": list(CASES),
        "shear": {"component": "g1", "minus": -0.02, "plus": 0.02},
        "population": "18<r<25.8, 0.37<Re<1.5; measured magnitude/radius cuts; no measured-|e| cut",
        "proposed_u_transition": proposed_result,
        "current_v3_5_like": current_result,
        "usable_oracle": oracle_result,
        "paired_proposed_minus_current": {
            "delta_m_pp": base.describe(proposed_m - current_m),
            "delta_model_R11": base.describe(proposed_r[:, 0, 0] - current_r[:, 0, 0]),
            "delta_component_contribution_to_m_pp": {
                kind: base.describe(proposed_components[kind] - current_components[kind])
                for kind in ("truth", "residual")
            },
        },
        "probability_metrics": {
            "proposed_against_U": aggregate_metrics(proposed_metrics),
            "current_against_U": aggregate_metrics(current_metrics),
        },
        "descriptive_flags": {
            "proposed_central_abs_m_below_0p3": abs(
                proposed_result["m_percent"]["value"]
            )
            < 0.3,
            "proposed_truth_and_residual_centers_each_below_0p3": all(
                abs(proposed_centers[name]["value"]) < 0.3
                for name in ("truth", "residual")
            ),
            "proposed_case_ci95_entirely_within_0p3": (
                proposed_result["m_percent"]["case_ci95"][0] > -0.3
                and proposed_result["m_percent"]["case_ci95"][1] < 0.3
            ),
        },
        "cached_moments": {
            "qmc_seeds": list(range(7301, 7317)),
            "draws_per_seed": 64,
            "common_random_numbers_between_legs": True,
            "note": "Uses the stored mean over 16 historical QMC groups; no flow evaluation was rerun.",
        },
        "bootstrap": {
            "unit": "simulation case",
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
        },
        "provenance": {
            "screen_script": str(Path(__file__).resolve()),
            "screen_script_sha256": helpers.digest(Path(__file__).resolve()),
            "base_worker": str(BASE_ROOT / "worker.py"),
            "base_worker_sha256": helpers.digest(BASE_ROOT / "worker.py"),
            "training_protocol": str(PROPOSED_ROOT / "protocol.json"),
            "training_protocol_sha256": helpers.digest(PROPOSED_ROOT / "protocol.json"),
            "training_review": str(review_path),
            "training_review_sha256": helpers.digest(review_path),
            "proposed_models": [
                {"path": str(path), "sha256": value}
                for path, value in zip(proposed_paths, PROPOSED_HASHES)
            ],
            "current_models": [
                {"path": str(path), "sha256": value}
                for path, value in zip(current_paths, CURRENT_HASHES)
            ],
            "source_receipts": source_receipts,
        },
        "limitations": [
            "Cases120-139 are a repeatedly reused development endpoint and cannot certify or promote the model.",
            "Only coherent g1 +/-0.02 is available; there is no g2, g=0 additive check, or independent amplitude here.",
            "The fixed measurement flow overlaps this historical case lineage, although neither classifier read cases120-139 during training or selection.",
            "The conditional discordant-pair objective is not a proper marginal-probability score under arbitrary shared-noise dependence.",
            "The interval is conditional on the historical shared image-noise field.",
        ],
    }
    helpers.write_json(output, result)
    print(
        "MINIMAL_U_TRANSITION_SCREEN_PASSED "
        f"m={proposed_result['m_percent']['value']:.6f}% "
        f"se={proposed_result['m_percent']['case_se']:.6f}pp "
        f"output={output}",
        flush=True,
    )


def selftest() -> None:
    actual = np.array([[[1.0], [0.2], [0.8]]])
    model = np.array([[[1.1], [0.3], [0.8]]])
    components = component_arrays(actual, model)
    total = components["truth"] + components["residual"]
    expected = -100.0 * (model[:, 0, 0] - actual[:, 0, 0]) / model[:, 0, 0]
    np.testing.assert_allclose(total, expected)
    print("MINIMAL_U_TRANSITION_SCREEN_SELFTEST_PASSED", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("selftest", "run"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda")
    arguments = parser.parse_args()
    if arguments.mode == "selftest":
        selftest()
    else:
        if arguments.output is None:
            parser.error("--output is required for run")
        run(arguments.output.resolve(), arguments.device)
