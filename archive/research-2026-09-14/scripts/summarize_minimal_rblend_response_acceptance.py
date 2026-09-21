#!/usr/bin/env python
"""Build one compact, verified decision from four frozen response endpoints."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict:
    def reject_constant(value: str):
        raise ValueError(f"non-finite JSON constant {value} in {path}")

    try:
        return json.loads(path.read_text(), parse_constant=reject_constant)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read JSON {path}: {error}") from error


def _expected_endpoints(protocol: dict) -> tuple[tuple[str, float], ...]:
    values = set()
    for leg in protocol["simulation"]["legs"]:
        for component in ("g1", "g2"):
            amplitude = abs(float(leg[component]))
            if amplitude:
                values.add((component, amplitude))
    return tuple(sorted(values, key=lambda item: (item[0], item[1])))


def summarize(endpoint_paths: list[Path], protocol_path: Path) -> dict:
    protocol = _load_json(protocol_path)
    if protocol.get("protocol") != "minimal-rblend-fresh-acceptance-v1":
        raise ValueError("wrong acceptance protocol")
    expected = _expected_endpoints(protocol)
    if len(endpoint_paths) != len(expected):
        raise RuntimeError("endpoint-result count does not match the protocol")

    protocol_sha = _sha256(protocol_path)
    numerical = protocol["response_evaluation"]["qmc_numerical_gate"]
    component_limit = float(
        protocol["response_evaluation"]["component_robustness_gate"][
            "maximum_absolute_central_contribution_percent"
        ]
    )
    band = np.asarray([-0.3, 0.3], dtype=np.float64)
    case_count = int(protocol["simulation"]["case_count"])
    records = {}
    artifacts = []
    for path in endpoint_paths:
        payload = _load_json(path)
        if payload.get("analysis") != "minimal-rblend blockwise response acceptance":
            raise RuntimeError(f"wrong endpoint analysis: {path}")
        if payload.get("protocol", {}).get("sha256") != protocol_sha:
            raise RuntimeError(f"endpoint/protocol hash mismatch: {path}")
        endpoint = payload.get("endpoint", {})
        key = (endpoint.get("shear_component"), float(endpoint.get("h", np.nan)))
        if key not in expected or key in records:
            raise RuntimeError(f"missing, duplicate, or unexpected endpoint: {key}")
        if endpoint.get("case_count") != case_count:
            raise RuntimeError(f"endpoint case count mismatch: {path}")

        acceptance = payload["response_acceptance"]
        expanded = np.asarray(
            acceptance["case_ci95_expanded_by_qmc_95_half_width_percent"],
            dtype=np.float64,
        )
        qmc_se = float(acceptance["qmc_standard_error_percent"])
        values = np.asarray(
            [
                acceptance["m_percent"],
                acceptance["case_bootstrap_standard_error_percent"],
                *acceptance["case_bootstrap_ci95_percent"],
                qmc_se,
                acceptance["qmc_95_half_width_percent"],
                *expanded,
            ],
            dtype=np.float64,
        )
        if not np.isfinite(values).all() or expanded.shape != (2,):
            raise RuntimeError(f"non-finite or malformed endpoint result: {path}")
        if acceptance.get("equivalence_band_percent") != band.tolist():
            raise RuntimeError(f"endpoint equivalence band drift: {path}")
        if not np.isclose(
            acceptance["qmc_standard_error_limit_percent"],
            numerical["maximum_standard_error_percent"],
            rtol=0,
            atol=0,
        ) or not np.isclose(
            acceptance["qmc_student_t_95_multiplier"],
            numerical["student_t_95_multiplier"],
            rtol=0,
            atol=0,
        ):
            raise RuntimeError(f"endpoint numerical gate drift: {path}")

        qmc_pass = qmc_se <= float(numerical["maximum_standard_error_percent"])
        endpoint_pass = bool(
            qmc_pass and expanded[0] > band[0] and expanded[1] < band[1]
        )
        if acceptance.get("passes_qmc_numerical_gate") is not qmc_pass:
            raise RuntimeError(f"stored QMC decision mismatch: {path}")
        if acceptance.get("passes_marginal_ci95_equivalence") is not endpoint_pass:
            raise RuntimeError(f"stored equivalence decision mismatch: {path}")

        contributions = payload["response_components"][
            "diagonal_m_contribution_percent"
        ]
        component_values = np.asarray(
            [
                contributions["truth_population"],
                contributions["measurement_residual"],
            ],
            dtype=np.float64,
        )
        if not np.isfinite(component_values).all():
            raise RuntimeError(f"non-finite endpoint decomposition: {path}")
        if not np.isclose(
            contributions["total"], acceptance["m_percent"], rtol=1e-12, atol=1e-12
        ):
            raise RuntimeError(f"endpoint decomposition does not close: {path}")
        observed_component_max = float(np.max(np.abs(component_values)))
        component_pass = bool(observed_component_max < component_limit)
        stored_component_gate = payload["response_components"].get(
            "robustness_gate", {}
        )
        expected_component_gate = {
            "maximum_absolute_central_contribution_percent": component_limit,
            "observed_maximum_absolute_central_contribution_percent": (
                observed_component_max
            ),
            "opposite_sign_central_contributions": bool(
                component_values[0] * component_values[1] < 0.0
            ),
            "passes": component_pass,
        }
        for field, value in expected_component_gate.items():
            stored = stored_component_gate.get(field)
            if isinstance(value, float):
                try:
                    matches = bool(np.isclose(float(stored), value, rtol=0, atol=0))
                except (TypeError, ValueError):
                    matches = False
            else:
                matches = stored is value
            if not matches:
                raise RuntimeError(
                    f"stored component-robustness decision mismatch: {path}"
                )
        endpoint_pass = bool(endpoint_pass and component_pass)
        records[key] = {
            "shear_component": key[0],
            "h": key[1],
            "m_percent": acceptance["m_percent"],
            "case_standard_error_percent": acceptance[
                "case_bootstrap_standard_error_percent"
            ],
            "case_ci95_percent": acceptance["case_bootstrap_ci95_percent"],
            "qmc_standard_error_percent": qmc_se,
            "qmc_95_half_width_percent": acceptance["qmc_95_half_width_percent"],
            "expanded_ci95_percent": expanded.tolist(),
            "equivalence_clearance_percent": float(
                min(expanded[0] - band[0], band[1] - expanded[1])
            ),
            "passes_response_equivalence": bool(
                qmc_pass and expanded[0] > band[0] and expanded[1] < band[1]
            ),
            "component_robustness_gate": expected_component_gate,
            "passes": endpoint_pass,
            "cross_response": payload["cross_response"],
            "m_decomposition_percent": contributions,
            "row_accounting": payload["row_accounting"],
        }
        artifacts.append(
            {"path": str(path.resolve()), "sha256": _sha256(path), "endpoint": list(key)}
        )

    if set(records) != set(expected):
        raise RuntimeError("endpoint set does not equal the frozen protocol")
    ordered = [records[key] for key in expected]
    passed = all(record["passes"] for record in ordered)
    return {
        "schema_version": 1,
        "analysis": "minimal-rblend four-endpoint response decision",
        "protocol": {
            "path": str(protocol_path.resolve()),
            "sha256": protocol_sha,
            "name": protocol["protocol"],
        },
        "decision": "pass_to_full_inference" if passed else "reject_candidate",
        "passes_all_four_endpoints": passed,
        "failed_endpoints": [
            {"shear_component": item["shear_component"], "h": item["h"]}
            for item in ordered
            if not item["passes"]
        ],
        "equivalence_band_percent": band.tolist(),
        "case_count": case_count,
        "endpoint_results": ordered,
        "input_artifacts": artifacts,
        "decision_rule": protocol["response_evaluation"]["gate"],
        "next_action": (
            "run the frozen full-inference gate without changing the candidate"
            if passed
            else "end this candidate without response tuning; consult the prespecified escalation"
        ),
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint-result", type=Path, action="append", required=True)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=REPOSITORY_ROOT / "configs/calibration_acceptance_minimal_rblend.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    result = summarize(args.endpoint_result, args.protocol)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: result[key] for key in (
        "decision", "passes_all_four_endpoints", "failed_endpoints"
    )}, indent=2))


if __name__ == "__main__":
    main()
