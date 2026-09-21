#!/usr/bin/env python
"""Summarize a Flow-E coupling ladder and select a density-admissible screen arm."""

import argparse
import hashlib
import json
from pathlib import Path


LADDER = (
    ("t0p5", 0.5),
    ("t1", 1.0),
    ("t2", 2.0),
    ("t5", 5.0),
    ("t10", 10.0),
    ("t25", 25.0),
    ("t75", 75.0),
    ("t200", 200.0),
)


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path):
    return json.loads(Path(path).read_text())


def _point(tag, weight, audit_path):
    audit_path = Path(audit_path)
    payload = _load(audit_path)
    model = Path(payload["model"])
    if not model.is_file():
        raise FileNotFoundError(model)
    return {
        "tag": tag,
        "coupling_weight": weight,
        "model": str(model.resolve()),
        "model_sha256": _sha256(model),
        "audit": str(audit_path.resolve()),
        "audit_sha256": _sha256(audit_path),
        "n_rows": int(payload["n_rows"]),
        "validation_cases": payload["validation_cases"],
        "nll_balanced": float(payload["nll_balanced"]),
        "shape_response_loss": float(payload["shape_response_loss"]),
        "magnitude_derivative_loss": float(payload["magnitude_derivative_loss"]),
        "log_size_derivative_loss": float(payload["log_size_derivative_loss"]),
        "photometric_derivative_loss": float(payload["photometric_derivative_loss"]),
    }


def summarize(root, baseline_audit, endpoint_audits, base_likelihood, output,
              likelihood_output, *, nll_margin=0.05, shape_factor=1.05):
    root = Path(root)
    output = Path(output)
    likelihood_output = Path(likelihood_output)
    for path in (output, likelihood_output):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}")

    baseline = _point("original_t0", 0.0, baseline_audit)
    points = []
    for tag, weight in LADDER:
        audit = root / f"heldout_derivative_losses_E_r450_{tag}_s501.json"
        points.append(_point(tag, weight, audit))

    reference_rows = baseline["n_rows"]
    reference_cases = baseline["validation_cases"]
    for point in points:
        if point["n_rows"] != reference_rows or point["validation_cases"] != reference_cases:
            raise ValueError(f"audit population mismatch for {point['tag']}")
        point["delta_nll_from_original"] = point["nll_balanced"] - baseline["nll_balanced"]
        point["shape_loss_ratio_to_original"] = (
            point["shape_response_loss"] / baseline["shape_response_loss"]
        )
        point["photometric_loss_ratio_to_original"] = (
            point["photometric_derivative_loss"]
            / baseline["photometric_derivative_loss"]
        )
        point["density_admissible"] = point["delta_nll_from_original"] <= nll_margin
        point["shape_admissible"] = point["shape_loss_ratio_to_original"] <= shape_factor
        point["eligible"] = point["density_admissible"] and point["shape_admissible"]

    # Mark the two-dimensional density/photometric Pareto frontier. Shape response is
    # retained as a separate hard gate instead of being hidden in a scalar objective.
    for point in points:
        point["pareto_nll_photometric"] = not any(
            other is not point
            and other["nll_balanced"] <= point["nll_balanced"]
            and other["photometric_derivative_loss"] <= point["photometric_derivative_loss"]
            and (
                other["nll_balanced"] < point["nll_balanced"]
                or other["photometric_derivative_loss"] < point["photometric_derivative_loss"]
            )
            for other in points
        )

    eligible = [point for point in points if point["eligible"]]
    if eligible:
        selected = min(
            eligible,
            key=lambda point: (point["photometric_derivative_loss"], point["nll_balanced"]),
        )
        basis = "minimum photometric loss subject to held-out density and shape gates"
    else:
        selected = min(points, key=lambda point: point["nll_balanced"])
        basis = "exploratory minimum-NLL arm; no retrain met both preregistered gates"

    endpoints = [_load(path) for path in endpoint_audits]
    result = {
        "purpose": "density-preserving Flow-E photometric-coupling screen",
        "selection_is_not_promotion": True,
        "gates": {
            "maximum_balanced_nll_degradation": nll_margin,
            "maximum_shape_response_loss_ratio": shape_factor,
        },
        "baseline": baseline,
        "ladder": points,
        "historical_high_weight_endpoints": endpoints,
        "selection_basis": basis,
        "selected": selected,
    }

    likelihood = _load(base_likelihood)
    likelihood["release"] = "v3.3-flow-retrain-screen"
    likelihood["description"] = (
        "Exploratory density-admissible Flow-E coupling-ladder selection with the "
        "unchanged transition-aware detector; not a promoted likelihood."
    )
    likelihood["measurement_model"]["path"] = selected["model"]
    likelihood["measurement_model"]["sha256"] = selected["model_sha256"]

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    likelihood_output.write_text(json.dumps(likelihood, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--baseline-audit", type=Path, required=True)
    parser.add_argument("--endpoint-audit", type=Path, action="append", default=[])
    parser.add_argument("--base-likelihood", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--likelihood-output", type=Path, required=True)
    parser.add_argument("--nll-margin", type=float, default=0.05)
    parser.add_argument("--shape-factor", type=float, default=1.05)
    args = parser.parse_args()
    summarize(
        args.root,
        args.baseline_audit,
        args.endpoint_audit,
        args.base_likelihood,
        args.output,
        args.likelihood_output,
        nll_margin=args.nll_margin,
        shape_factor=args.shape_factor,
    )


if __name__ == "__main__":
    main()
