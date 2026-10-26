#!/usr/bin/env python3
"""Screen the frozen minimal R_blend detector on cached coherent moments.

This is deliberately a reused g1 +/-0.02 development diagnostic.  It cannot
promote the likelihood or establish 0.3% calibration.  Its purpose is to avoid
new image simulation when the existing cases120--139 moments can already expose
a gross response failure.
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

from sbsi.selection_model import load_selection_model


CASES = tuple(range(120, 140))
CUT_SLOT = 1  # no-|e| among (none, no_e, all)
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
MODEL_SPECS = {
    "flow8": {
        "hashes": (
            "e0a7741a49b74c2941b937f7d7d7d77abd72efdcf750786977ea25a6088a8321",
            "5f1fa744282eca298e52f875c5ccd58d36a28f7cd08c955a3e636038cd17da4d",
            "c809343087cde616c68d0c56fee7bfb5cd92e09fc8169460f48f6e076e26e4ff",
        ),
    },
    "flow8_rblend": {
        "hashes": (
            "bbbd27a86c18bffb3e8b98e46c455ebcd2a27844ead363913bb5289ae19e6685",
            "031498aa4140be4d2dfc7605f5211a1d2379e14c16f3fbc23b79f2f7f9e028cd",
            "dab1e4cddf3b61da4cc6dee810091209b29df4dbf5dcc86cd20d3e23eebbd702",
        ),
    },
}


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


def run(output: Path, device: str, arm: str) -> None:
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("the catalogue screen must run through Slurm")

    base = load_module(BASE_ROOT / "worker.py", "minimal_rblend_cached_base")
    base_config = base.verify()
    candidate = load_module(
        Path(__file__).with_name("train_minimal_coherent_u_rblend_ablation.py"),
        "minimal_rblend_candidate_source",
    )
    candidate_protocol = candidate.load_protocol(CANDIDATE_ROOT)

    model_paths = tuple(
        CANDIDATE_ROOT / f"models/{arm}/seed{seed}/selected.pt"
        for seed in (20260913, 20260914, 20260915)
    )
    expected_hashes = MODEL_SPECS[arm]["hashes"]
    observed_hashes = tuple(digest(path) for path in model_paths)
    if observed_hashes != expected_hashes:
        raise RuntimeError("frozen minimal-detector checkpoint hash mismatch")
    members = [load_selection_model(path, device=device) for path in model_paths]

    actual_by_case = []
    oracle_by_case = []
    candidate_by_case = []
    metrics_by_case = []
    source_receipts = []
    for case in CASES:
        data, _, prepared_receipt, moments, moment_receipt = base.source_case(
            base_config, case, with_moments=True
        )
        ids, frames, usable, _candidate_receipt = candidate.verified_case(
            candidate_protocol, case
        )
        candidate_receipt = candidate.case_root(candidate_protocol, case) / "receipt.json"
        np.testing.assert_array_equal(ids, data["ids"])
        np.testing.assert_array_equal(usable, data["U"])
        probabilities = np.mean(
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
        if probabilities.shape != data["U"].shape:
            raise RuntimeError(f"candidate probability shape mismatch: case={case}")

        _, cached = verify_part(BASE_ROOT / "parts" / f"case{case:03d}.npz")
        actual_by_case.append(cached["actual"][CUT_SLOT])
        oracle_by_case.append(cached["model"][0, CUT_SLOT])
        candidate_by_case.append(
            base.model_statistics(
                probabilities,
                moments["q"][MOMENT_CUT],
                moments["u"][MOMENT_CUT],
                data["es"],
            )
        )
        metrics_by_case.append(base.metric_sufficient(data["U"], probabilities))
        source_receipts.append(
            {
                "case": case,
                "prepared_receipt": str(prepared_receipt),
                "prepared_receipt_sha256": digest(prepared_receipt),
                "moment_receipt": str(moment_receipt),
                "moment_receipt_sha256": digest(moment_receipt),
                "candidate_receipt": str(candidate_receipt),
                "candidate_receipt_sha256": digest(candidate_receipt),
            }
        )
        print(f"MINIMAL_RBLEND_EXISTING_SCREEN_CASE case={case}", flush=True)

    actual = np.asarray(actual_by_case)
    oracle = np.asarray(oracle_by_case)
    predicted = np.asarray(candidate_by_case)
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

    metric_sums = np.asarray(metrics_by_case).sum(axis=(0, 1))
    probability_metrics = {
        "n_leg_rows": int(metric_sums[0]),
        "observed_usable_fraction": float(metric_sums[1] / metric_sums[0]),
        "predicted_usable_fraction": float(metric_sums[2] / metric_sums[0]),
        "signed_probability_gap": float(
            (metric_sums[2] - metric_sums[1]) / metric_sums[0]
        ),
        "bce": float(metric_sums[3] / metric_sums[0]),
        "brier": float(metric_sums[4] / metric_sums[0]),
    }
    candidate_result = response_entry(base, actual_r, predicted_r)
    oracle_result = response_entry(base, actual_r, oracle_r)
    component_centers = candidate_result["component_contribution_to_m_pp"]
    result = {
        "analysis": "reused cached-moment screen of frozen minimal detector",
        "screen_only_not_promotion": True,
        "model_arm": arm,
        "cases": list(CASES),
        "shear": {"component": "g1", "minus": -0.02, "plus": 0.02},
        "population": "18<r<25.8, 0.37<Re<1.5; measured magnitude/radius cuts; no measured-|e| cut",
        "candidate": candidate_result,
        "usable_oracle": oracle_result,
        "probability_metrics": probability_metrics,
        "descriptive_flags": {
            "candidate_central_abs_m_below_0p3": abs(
                candidate_result["m_percent"]["value"]
            )
            < 0.3,
            "candidate_truth_and_residual_centers_each_below_0p3": all(
                abs(component_centers[name]["value"]) < 0.3
                for name in ("truth", "residual")
            ),
            "candidate_case_ci95_entirely_within_0p3": (
                candidate_result["m_percent"]["case_ci95"][0] > -0.3
                and candidate_result["m_percent"]["case_ci95"][1] < 0.3
            ),
        },
        "cached_moments": {
            "qmc_seeds": list(range(7301, 7317)),
            "draws_per_seed": 64,
            "common_random_numbers_between_legs": True,
            "note": "Uses the already stored mean over all 16 historical QMC groups; no flow evaluation was rerun.",
        },
        "provenance": {
            "base_worker": str(BASE_ROOT / "worker.py"),
            "base_worker_sha256": digest(BASE_ROOT / "worker.py"),
            "candidate_protocol": str(CANDIDATE_ROOT / "protocol.json"),
            "candidate_protocol_sha256": digest(CANDIDATE_ROOT / "protocol.json"),
            "candidate_models": [
                {"path": str(path), "sha256": value}
                for path, value in zip(model_paths, observed_hashes)
            ],
            "source_receipts": source_receipts,
        },
        "bootstrap": {
            "unit": "simulation case",
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
        },
        "limitations": [
            "Cases120-139 are a repeatedly reused development endpoint and cannot certify or promote the candidate.",
            "Only coherent g1 +/-0.02 is available; there is no g2, g=0 additive check, or independent amplitude here.",
            "The fixed measurement flow overlaps this historical case lineage, although the nine-input detector never used cases120-139.",
            "The interval is conditional on the historical shared image-noise field.",
        ],
    }
    write_json(output, result)
    print(
        "MINIMAL_RBLEND_EXISTING_SCREEN_PASSED "
        f"arm={arm} "
        f"m={candidate_result['m_percent']['value']:.6f}% "
        f"se={candidate_result['m_percent']['case_se']:.6f}pp "
        f"output={output}",
        flush=True,
    )


def selftest() -> None:
    base = load_module(BASE_ROOT / "worker.py", "minimal_rblend_cached_base_test")
    rng = np.random.default_rng(19)
    sums = rng.uniform(0.2, 1.0, size=(5, 4, 2, 3))
    weights = np.ones((1, 5), dtype=int)
    got = base.responses(sums, weights)[0]
    pooled = sums.sum(axis=0)
    means = pooled[..., 1:] / pooled[..., :1]
    expected = (means[..., 0, :] - means[..., 1, :]) / base.H
    np.testing.assert_allclose(got, expected)
    print("MINIMAL_RBLEND_EXISTING_SCREEN_SELFTEST_PASSED", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("selftest", "run"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--arm", choices=tuple(MODEL_SPECS), default="flow8_rblend")
    arguments = parser.parse_args()
    if arguments.mode == "selftest":
        selftest()
    else:
        if arguments.output is None:
            parser.error("--output is required for run")
        run(arguments.output.resolve(), arguments.device, arguments.arm)
