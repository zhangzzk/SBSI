#!/usr/bin/env python3
"""Audit V3.5-like classifier response error across train/tune/screen cases.

This reads the frozen classifier and existing 16 x 64 flow moments only.  The
comparison to the same-moment U oracle isolates whether the classifier's
response-weighting error is stable enough to justify a training-case moment
constraint; it does not train or select a model.
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


SPLITS = {
    "train": tuple(range(40, 100)),
    "tune": tuple(range(100, 120)),
    "screen": tuple(range(120, 140)),
}
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
SEEDS = (20260913, 20260914, 20260915)
MODEL_PATHS = tuple(
    CANDIDATE_ROOT / f"models/flow8_rblend/seed{seed}/selected.pt"
    for seed in SEEDS
)
MODEL_HASHES = (
    "bbbd27a86c18bffb3e8b98e46c455ebcd2a27844ead363913bb5289ae19e6685",
    "031498aa4140be4d2dfc7605f5211a1d2379e14c16f3fbc23b79f2f7f9e028cd",
    "dab1e4cddf3b61da4cc6dee810091209b29df4dbf5dcc86cd20d3e23eebbd702",
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
        raise RuntimeError(f"cannot import module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, value: dict) -> None:
    if path.exists():
        raise FileExistsError(f"preserving existing result: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def actual_statistics(base, data: dict) -> np.ndarray:
    keep = data["U"].astype(bool) & data["C"][MOMENT_CUT].astype(bool)
    return np.asarray(
        [
            base.sufficient(keep, data["y"]),
            base.sufficient(keep, data["es"]),
            base.sufficient(keep, data["y"] - data["es"]),
        ]
    )


def describe_entry(base, actual_r: np.ndarray, model_r: np.ndarray) -> dict:
    denominator = model_r[:, 0, 0]
    m = 100.0 * (actual_r[:, 0, 0] / denominator - 1.0)
    components = {}
    for index, name in ((1, "truth"), (2, "residual")):
        components[name] = -100.0 * (
            model_r[:, index, 0] - actual_r[:, index, 0]
        ) / denominator
    return {
        "actual_R11": base.describe(actual_r[:, 0, 0]),
        "model_R11": base.describe(denominator),
        "m_percent": base.describe(m),
        "component_contribution_to_m_pp": {
            name: base.describe(value) for name, value in components.items()
        },
    }


def paired_entry(base, candidate_r: np.ndarray, oracle_r: np.ndarray,
                 candidate: dict, oracle: dict) -> dict:
    return {
        "candidate_minus_oracle_R11": base.describe(
            candidate_r[:, 0, 0] - oracle_r[:, 0, 0]
        ),
        "candidate_minus_oracle_truth_R11": base.describe(
            candidate_r[:, 1, 0] - oracle_r[:, 1, 0]
        ),
        "candidate_minus_oracle_residual_R11": base.describe(
            candidate_r[:, 2, 0] - oracle_r[:, 2, 0]
        ),
        "candidate_minus_oracle_m_pp": float(
            candidate["m_percent"]["value"] - oracle["m_percent"]["value"]
        ),
        "candidate_minus_oracle_component_pp": {
            name: float(
                candidate["component_contribution_to_m_pp"][name]["value"]
                - oracle["component_contribution_to_m_pp"][name]["value"]
            )
            for name in ("truth", "residual")
        },
    }


def run(output: Path, device: str) -> None:
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("the audit must run through Slurm")
    if tuple(digest(path) for path in MODEL_PATHS) != MODEL_HASHES:
        raise RuntimeError("V3.5-like classifier hash mismatch")
    base = load_module(BASE_ROOT / "worker.py", "classifier_generalization_base")
    config = base.verify()
    candidate_source = Path(__file__).with_name(
        "train_minimal_coherent_u_rblend_ablation.py"
    )
    candidate = load_module(candidate_source, "classifier_generalization_source")
    protocol = json.loads((CANDIDATE_ROOT / "protocol.json").read_text())
    members = [load_selection_model(path, device=device) for path in MODEL_PATHS]
    model_names = [*(f"seed{seed}" for seed in SEEDS), "ensemble"]
    by_case = {}

    for split, cases in SPLITS.items():
        records = []
        for case in cases:
            data, _, prepared_receipt, moments, moment_receipt = base.source_case(
                config, case, with_moments=True
            )
            ids, frames, usable, _ = candidate.verified_case(protocol, case)
            np.testing.assert_array_equal(ids, data["ids"])
            np.testing.assert_array_equal(usable, data["U"])
            member_probabilities = [
                np.stack(
                    [member.predict_proba(frame, batch_size=65_536) for frame in frames]
                )
                for member in members
            ]
            probabilities = [
                *member_probabilities,
                np.mean(np.stack(member_probabilities), axis=0),
            ]
            record = {
                "case": case,
                "actual": actual_statistics(base, data),
                "oracle": base.model_statistics(
                    data["U"],
                    moments["q"][MOMENT_CUT],
                    moments["u"][MOMENT_CUT],
                    data["es"],
                )[:3],
                "models": [
                    base.model_statistics(
                        probability,
                        moments["q"][MOMENT_CUT],
                        moments["u"][MOMENT_CUT],
                        data["es"],
                    )[:3]
                    for probability in probabilities
                ],
                "metrics": [
                    base.metric_sufficient(data["U"], probability)
                    for probability in probabilities
                ],
                "prepared_receipt_sha256": digest(prepared_receipt),
                "moment_receipt_sha256": digest(moment_receipt),
            }
            records.append(record)
            print(f"CLASSIFIER_RESPONSE_AUDIT split={split} case={case}", flush=True)
        by_case[split] = records

    results = {}
    for split_index, (split, cases) in enumerate(SPLITS.items()):
        records = by_case[split]
        rng = np.random.default_rng(BOOTSTRAP_SEED + split_index)
        weights = np.vstack(
            [
                np.ones((1, len(cases)), dtype=int),
                rng.multinomial(
                    len(cases),
                    np.full(len(cases), 1.0 / len(cases)),
                    BOOTSTRAP_REPLICATES,
                ),
            ]
        )
        actual_r = base.responses(np.asarray([r["actual"] for r in records]), weights)
        oracle_r = base.responses(np.asarray([r["oracle"] for r in records]), weights)
        oracle_result = describe_entry(base, actual_r, oracle_r)
        model_results = {}
        for model_index, name in enumerate(model_names):
            model_r = base.responses(
                np.asarray([r["models"][model_index] for r in records]), weights
            )
            entry = describe_entry(base, actual_r, model_r)
            metric = np.asarray([r["metrics"][model_index] for r in records]).sum(
                axis=(0, 1)
            )
            entry["probability_metrics"] = {
                "n_leg_rows": int(metric[0]),
                "observed_usable_fraction": float(metric[1] / metric[0]),
                "predicted_usable_fraction": float(metric[2] / metric[0]),
                "signed_probability_gap": float((metric[2] - metric[1]) / metric[0]),
                "bce": float(metric[3] / metric[0]),
                "brier": float(metric[4] / metric[0]),
            }
            entry["relative_to_oracle"] = paired_entry(
                base, model_r, oracle_r, entry, oracle_result
            )
            model_results[name] = entry
        results[split] = {
            "cases": list(cases),
            "oracle": oracle_result,
            "models": model_results,
        }

    payload = {
        "analysis": "V3.5-like classifier response generalization audit",
        "screen_only_not_promotion": True,
        "selection": "MAG_AUTO<25.8; FLUX_RADIUS>=0.75 arcsec; no measured-|e| cut",
        "splits": results,
        "interpretation_rule": (
            "A stable same-sign candidate-minus-U-oracle truth-response error across "
            "train, tune, and screen supports testing a training-only oracle-moment "
            "constraint. A train-only improvement or sign instability rejects it."
        ),
        "provenance": {
            "base_worker": str(BASE_ROOT / "worker.py"),
            "base_worker_sha256": digest(BASE_ROOT / "worker.py"),
            "candidate_source": str(candidate_source),
            "candidate_source_sha256": digest(candidate_source),
            "candidate_protocol": str(CANDIDATE_ROOT / "protocol.json"),
            "candidate_protocol_sha256": digest(CANDIDATE_ROOT / "protocol.json"),
            "models": [
                {"path": str(path), "sha256": value}
                for path, value in zip(MODEL_PATHS, MODEL_HASHES)
            ],
        },
        "bootstrap": {
            "unit": "simulation case within split",
            "replicates": BOOTSTRAP_REPLICATES,
            "seed_start": BOOTSTRAP_SEED,
        },
        "limitations": [
            "The screen split is repeatedly reused development data.",
            "All splits share the fixed flow and historical simulation lineage.",
            "Only coherent g1 +/-0.02 and the declared 0.75-arcsec sample are tested.",
        ],
    }
    write_json(output, payload)
    print(f"CLASSIFIER_RESPONSE_AUDIT_COMPLETE output={output}", flush=True)


def selftest() -> None:
    candidate_r = np.arange(18, dtype=float).reshape(2, 3, 3)
    oracle_r = candidate_r - 2
    np.testing.assert_allclose(candidate_r - oracle_r, 2)
    print("CLASSIFIER_RESPONSE_AUDIT_SELFTEST_PASSED")


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
            parser.error("--output is required")
        run(arguments.output.resolve(), arguments.device)
