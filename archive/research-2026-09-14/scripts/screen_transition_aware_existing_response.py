#!/usr/bin/env python3
"""Reweight cached cases120--139 moments with the old transition detector.

This is a deliberately narrow classifier-only mechanism comparison.  It reads
the already materialized detector frames and lambda10/trial9 flow moments; it
does not generate images, catalogues, or flow samples.  The historical model
predicts raw detection D, whereas the current V3.5-like classifier predicts
usable measurement U, so the output reports calibration against both labels
and must not be interpreted as a drop-in model promotion.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import importlib.util
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from sbsi.selection_model import load_selection_model


CASES = tuple(range(120, 140))
CUT_SLOT = 1  # measured magnitude/radius cuts, no measured-|e| cut
MOMENT_CUT = 6
BOOTSTRAP_SEED = 20260913
BOOTSTRAP_REPLICATES = 10_000
BASE_ROOT = Path(
    "/project/ls-gruen/users/zekang.zhang/sbsi_caches/"
    "coherent_u_crowd_ensemble_v1"
)
CANDIDATE_ROOT = Path(
    "/project/ls-gruen/users/zekang.zhang/sbsi_caches/"
    "minimal_coherent_u_flow_rblend_ablation_v2"
)
MODEL_PATH = Path(
    "/project/ls-gruen/users/zekang.zhang/sbsi_caches/"
    "detection_classifier_transition_lambda1_v1/models/transition_aware.pt"
)
MODEL_SHA256 = "9966cfbc191f11b049bf7419dbdb45d65d1262428889a91bb3c9caf928703455"
CURRENT_MODEL_HASHES = (
    "bbbd27a86c18bffb3e8b98e46c455ebcd2a27844ead363913bb5289ae19e6685",
    "031498aa4140be4d2dfc7605f5211a1d2379e14c16f3fbc23b79f2f7f9e028cd",
    "dab1e4cddf3b61da4cc6dee810091209b29df4dbf5dcc86cd20d3e23eebbd702",
)
EXPECTED_FEATURES = (
    "Re_input_p_scaled",
    "Re_input_s_scaled",
    "r_input_p_scaled",
    "r_input_s_scaled",
    "sersic_n_input_p",
    "sersic_n_input_s",
    "distance_scaled",
    "neighbored",
    "neighbour_log_impact",
    "e1_input_p",
    "e2_input_p",
    "e1_input_s",
    "e2_input_s",
    "e_parallel_p",
    "e_cross_p",
    "e_parallel_s",
    "e_cross_s",
)


def digest(path: str | Path) -> str:
    value = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8_388_608), b""):
            value.update(block)
    return value.hexdigest()


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import frozen module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"preserving existing screen output: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def load_candidate_protocol(candidate) -> dict:
    """Verify the frozen protocol after its experiment code was archived."""

    protocol_path = CANDIDATE_ROOT / "protocol.json"
    protocol = json.loads(protocol_path.read_text())
    replacements = {
        "script": Path(candidate.__file__),
        "job_script": Path(__file__).parents[1]
        / "jobs/job_train_minimal_coherent_u_rblend_ablation.sh",
    }
    for key, expected in protocol["hashes"].items():
        path = replacements.get(key, Path(protocol["paths"][key]))
        if digest(path) != expected:
            raise RuntimeError(f"frozen candidate input changed: {key} {path}")
    return protocol


def verify_part(path: Path) -> tuple[dict, dict[str, np.ndarray]]:
    receipt_path = path.with_suffix(".json")
    receipt = json.loads(receipt_path.read_text())
    case = int(path.stem.removeprefix("case"))
    if receipt.get("case") != case or receipt.get("output_sha256") != digest(path):
        raise RuntimeError(f"cached response-part receipt mismatch: {path}")
    with np.load(path) as stored:
        arrays = {name: stored[name] for name in stored.files}
    if int(arrays["case"]) != case:
        raise RuntimeError(f"cached response-part case mismatch: {path}")
    return receipt, arrays


def response_entry(base, actual_r: np.ndarray, model_r: np.ndarray) -> dict:
    denominator = model_r[:, 0, 0]
    m = 100.0 * (actual_r[:, 0, 0] / denominator - 1.0)
    result = {
        "actual_R11": base.describe(actual_r[:, 0, 0]),
        "model_R11": base.describe(denominator),
        "m_percent": base.describe(m),
        "component_delta_R_model_minus_data": {},
        "component_contribution_to_m_pp": {},
    }
    for kind_index, kind in ((1, "truth"), (2, "residual")):
        delta = model_r[:, kind_index, 0] - actual_r[:, kind_index, 0]
        result["component_delta_R_model_minus_data"][kind] = base.describe(delta)
        result["component_contribution_to_m_pp"][kind] = base.describe(
            -100.0 * delta / denominator
        )
    return result


def aggregate_metrics(values: list[np.ndarray]) -> dict:
    total = np.asarray(values).sum(axis=(0, 1))
    return {
        "n_leg_rows": int(total[0]),
        "observed_fraction": float(total[1] / total[0]),
        "predicted_fraction": float(total[2] / total[0]),
        "signed_probability_gap": float((total[2] - total[1]) / total[0]),
        "bce": float(total[3] / total[0]),
        "brier": float(total[4] / total[0]),
    }


def run(output: Path, device: str) -> None:
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("the catalogue screen must run through Slurm")

    base = load_module(BASE_ROOT / "worker.py", "transition_cached_base")
    base_config = base.verify()
    candidate = load_module(
        Path(__file__).with_name("train_minimal_coherent_u_rblend_ablation.py"),
        "transition_current_candidate_source",
    )
    candidate_protocol = load_candidate_protocol(candidate)
    if digest(MODEL_PATH) != MODEL_SHA256:
        raise RuntimeError("transition-aware checkpoint hash mismatch")
    model = load_selection_model(MODEL_PATH, device=device)
    features = tuple(model.preprocessor.feature_names)
    if features != EXPECTED_FEATURES:
        raise RuntimeError(f"transition-aware feature contract changed: {features}")
    metadata = model.metadata
    if (
        metadata.get("training_objective") != "transition_aware"
        or float(metadata.get("transition_weight", np.nan)) != 1.0
        or metadata.get("response_supervision") is not False
    ):
        raise RuntimeError("transition-aware training metadata changed")
    current_model_paths = tuple(
        CANDIDATE_ROOT / f"models/flow8_rblend/seed{seed}/selected.pt"
        for seed in (20260913, 20260914, 20260915)
    )
    if tuple(digest(path) for path in current_model_paths) != CURRENT_MODEL_HASHES:
        raise RuntimeError("current V3.5-like classifier hash mismatch")
    current_members = [
        load_selection_model(path, device=device) for path in current_model_paths
    ]

    actual_by_case = []
    oracle_by_case = []
    predicted_by_case = []
    current_by_case = []
    u_metrics_by_case = []
    d_metrics_by_case = []
    current_u_metrics_by_case = []
    source_receipts = []
    for case in CASES:
        data, _unused_frames, prepared_receipt, moments, moment_receipt = (
            base.source_case(base_config, case, with_moments=True)
        )
        current_ids, current_frames, current_usable, _candidate_receipt = (
            candidate.verified_case(candidate_protocol, case)
        )
        np.testing.assert_array_equal(current_ids, data["ids"])
        np.testing.assert_array_equal(current_usable, data["U"])
        prepared = base.source_root(base_config, case) / "prepared" / f"case{case:03d}"
        frames = [
            pd.read_feather(
                prepared / f"detector{leg}.feather",
                columns=["input_index", *features],
            )
            for leg in range(2)
        ]
        for frame in frames:
            np.testing.assert_array_equal(
                frame["input_index"].to_numpy(np.int64), data["ids"]
            )
        probabilities = np.stack(
            [model.predict_proba(frame, batch_size=65_536) for frame in frames]
        )
        if probabilities.shape != data["U"].shape:
            raise RuntimeError(f"transition probability shape mismatch: case={case}")
        if not np.isfinite(probabilities).all() or np.any(
            (probabilities < 0.0) | (probabilities > 1.0)
        ):
            raise RuntimeError(f"invalid transition probabilities: case={case}")
        current_probabilities = np.mean(
            np.stack(
                [
                    np.stack(
                        [
                            member.predict_proba(frame, batch_size=65_536)
                            for frame in current_frames
                        ]
                    )
                    for member in current_members
                ]
            ),
            axis=0,
        )
        if current_probabilities.shape != data["U"].shape:
            raise RuntimeError(f"current probability shape mismatch: case={case}")

        _, cached = verify_part(BASE_ROOT / "parts" / f"case{case:03d}.npz")
        actual_by_case.append(cached["actual"][CUT_SLOT])
        oracle_by_case.append(cached["model"][0, CUT_SLOT])
        predicted_by_case.append(
            base.model_statistics(
                probabilities,
                moments["q"][MOMENT_CUT],
                moments["u"][MOMENT_CUT],
                data["es"],
            )
        )
        current_by_case.append(
            base.model_statistics(
                current_probabilities,
                moments["q"][MOMENT_CUT],
                moments["u"][MOMENT_CUT],
                data["es"],
            )
        )
        u_metrics_by_case.append(base.metric_sufficient(data["U"], probabilities))
        d_metrics_by_case.append(base.metric_sufficient(data["D"], probabilities))
        current_u_metrics_by_case.append(
            base.metric_sufficient(data["U"], current_probabilities)
        )
        source_receipts.append(
            {
                "case": case,
                "prepared_receipt": str(prepared_receipt),
                "prepared_receipt_sha256": digest(prepared_receipt),
                "moment_receipt": str(moment_receipt),
                "moment_receipt_sha256": digest(moment_receipt),
            }
        )
        print(f"TRANSITION_EXISTING_SCREEN_CASE case={case}", flush=True)

    actual = np.asarray(actual_by_case)
    oracle = np.asarray(oracle_by_case)
    predicted = np.asarray(predicted_by_case)
    current = np.asarray(current_by_case)
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
    predicted_r = base.responses(predicted, weights)
    current_r = base.responses(current, weights)
    transition_result = response_entry(base, actual_r, predicted_r)
    current_result = response_entry(base, actual_r, current_r)
    oracle_result = response_entry(base, actual_r, oracle_r)
    component_centers = transition_result["component_contribution_to_m_pp"]
    transition_m = 100.0 * (actual_r[:, 0, 0] / predicted_r[:, 0, 0] - 1.0)
    current_m = 100.0 * (actual_r[:, 0, 0] / current_r[:, 0, 0] - 1.0)
    paired_components = {}
    for kind_index, kind in ((1, "truth"), (2, "residual")):
        transition_component = -100.0 * (
            predicted_r[:, kind_index, 0] - actual_r[:, kind_index, 0]
        ) / predicted_r[:, 0, 0]
        current_component = -100.0 * (
            current_r[:, kind_index, 0] - actual_r[:, kind_index, 0]
        ) / current_r[:, 0, 0]
        paired_components[kind] = base.describe(
            transition_component - current_component
        )
    result = {
        "analysis": "reused cached-moment screen of the historical transition-aware raw-D detector",
        "screen_only_not_promotion": True,
        "cases": list(CASES),
        "shear": {"component": "g1", "minus": -0.02, "plus": 0.02},
        "population": "18<r<25.8, 0.37<Re<1.5; measured magnitude/radius cuts; no measured-|e| cut",
        "transition_raw_D_model": transition_result,
        "current_v3_5_like_usable_U_model": current_result,
        "usable_oracle": oracle_result,
        "paired_transition_minus_current": {
            "delta_m_pp": base.describe(transition_m - current_m),
            "delta_model_R11": base.describe(
                predicted_r[:, 0, 0] - current_r[:, 0, 0]
            ),
            "delta_component_contribution_to_m_pp": paired_components,
        },
        "probability_metrics": {
            "against_raw_detection_D": aggregate_metrics(d_metrics_by_case),
            "against_usable_measurement_U": aggregate_metrics(u_metrics_by_case),
            "current_v3_5_like_against_U": aggregate_metrics(
                current_u_metrics_by_case
            ),
        },
        "descriptive_flags": {
            "transition_central_abs_m_below_0p3": abs(
                transition_result["m_percent"]["value"]
            )
            < 0.3,
            "transition_truth_and_residual_centers_each_below_0p3": all(
                abs(component_centers[name]["value"]) < 0.3
                for name in ("truth", "residual")
            ),
            "transition_case_ci95_entirely_within_0p3": (
                transition_result["m_percent"]["case_ci95"][0] > -0.3
                and transition_result["m_percent"]["case_ci95"][1] < 0.3
            ),
        },
        "cached_moments": {
            "qmc_seeds": list(range(7301, 7317)),
            "draws_per_seed": 64,
            "common_random_numbers_between_legs": True,
            "note": "Uses the stored mean over 16 historical QMC groups; no flow evaluation was rerun.",
        },
        "provenance": {
            "base_worker": str(BASE_ROOT / "worker.py"),
            "base_worker_sha256": digest(BASE_ROOT / "worker.py"),
            "model": str(MODEL_PATH),
            "model_sha256": MODEL_SHA256,
            "model_features": list(features),
            "model_metadata": metadata,
            "current_models": [
                {"path": str(path), "sha256": value}
                for path, value in zip(current_model_paths, CURRENT_MODEL_HASHES)
            ],
            "current_protocol": str(CANDIDATE_ROOT / "protocol.json"),
            "current_protocol_sha256": digest(CANDIDATE_ROOT / "protocol.json"),
            "source_receipts": source_receipts,
        },
        "bootstrap": {
            "unit": "simulation case",
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
        },
        "limitations": [
            "The historical transition-aware checkpoint predicts raw detection D, while the likelihood's event and oracle are usable measurement U; this is a mechanism comparison, not a drop-in replacement test.",
            "Cases120-139 are a repeatedly reused development endpoint and cannot certify or promote the model.",
            "Only coherent g1 +/-0.02 is available; there is no g2, g=0 additive check, or independent amplitude here.",
            "The transition model was trained on cases0-24, tuned25-29, and has no case overlap with this screen, but its support and event definition differ from the V3.5-like classifier.",
            "The fixed measurement flow overlaps this historical case lineage.",
            "The interval is conditional on the historical shared image-noise field.",
        ],
    }
    write_json(output, result)
    print(
        "TRANSITION_EXISTING_SCREEN_PASSED "
        f"m={transition_result['m_percent']['value']:.6f}% "
        f"se={transition_result['m_percent']['case_se']:.6f}pp "
        f"output={output}",
        flush=True,
    )


def selftest() -> None:
    base = load_module(BASE_ROOT / "worker.py", "transition_cached_base_test")
    rng = np.random.default_rng(29)
    sums = rng.uniform(0.2, 1.0, size=(5, 4, 2, 3))
    got = base.responses(sums, np.ones((1, 5), dtype=int))[0]
    pooled = sums.sum(axis=0)
    means = pooled[..., 1:] / pooled[..., :1]
    expected = (means[..., 0, :] - means[..., 1, :]) / base.H
    np.testing.assert_allclose(got, expected)
    metrics = aggregate_metrics(
        [base.metric_sufficient(np.array([[0, 1], [1, 0]]), np.full((2, 2), 0.5))]
    )
    np.testing.assert_allclose(metrics["bce"], np.log(2.0))
    np.testing.assert_allclose(metrics["brier"], 0.25)
    print("TRANSITION_EXISTING_SCREEN_SELFTEST_PASSED", flush=True)


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
