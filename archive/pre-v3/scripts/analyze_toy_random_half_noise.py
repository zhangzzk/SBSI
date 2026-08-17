"""Aggregate independent-noise replicates of the random-half response toy."""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np


CONTRASTS = {
    "production_minus_anchor": (
        "cross_amplitude_contrasts",
        "production_half_forward_hi_minus_anchor_coherent_lo",
    ),
    "coherent_amplitude_hi_minus_lo": (
        "cross_amplitude_contrasts", "coherent_hi_minus_lo",
    ),
}


def mean_sem(values):
    values = np.asarray(values, dtype=float)
    return {
        "mean": float(values.mean()),
        "seed_sem": float(values.std(ddof=1) / np.sqrt(len(values))),
        "n_noise_seeds": int(len(values)),
        "values": values.tolist(),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inputs", required=True, help="glob for per-noise-seed JSON files")
    ap.add_argument("--output-json", required=True)
    args = ap.parse_args()
    paths = sorted(glob.glob(args.inputs))
    if len(paths) < 4:
        raise RuntimeError(f"need at least four noise seeds, found {len(paths)}")
    docs = [json.load(open(path, encoding="utf-8")) for path in paths]
    scenarios = sorted(docs[0]["scenarios"])
    if any(sorted(doc["scenarios"]) != scenarios for doc in docs):
        raise RuntimeError("scenario sets differ across noise seeds")

    out = {
        "inputs": paths,
        "n_noise_seeds": len(paths),
        "scenarios": {},
        "gate": {
            "rule": "production-minus-anchor <= -0.005 and >3 seed SEM in >=2 crowded scenes",
            "carrier_scale": 0.005,
        },
    }
    passing = []
    for name in scenarios:
        rows = {}
        for label, (parent, key) in CONTRASTS.items():
            vals = [
                doc["scenarios"][name][parent][key]["difference"] for doc in docs
            ]
            rows[label] = mean_sem(vals)
        for g in ("0.05", "0.2"):
            item = f"half_forward_minus_central_g{g}"
            vals = [
                doc["scenarios"][name]["amplitudes"][g]["contrasts"]
                ["half_forward_minus_half_central"]["difference"]
                for doc in docs
            ]
            rows[item] = mean_sem(vals)
            item = f"half_central_minus_coherent_g{g}"
            vals = [
                doc["scenarios"][name]["amplitudes"][g]["contrasts"]
                ["half_central_minus_coherent"]["difference"]
                for doc in docs
            ]
            rows[item] = mean_sem(vals)
        main = rows["production_minus_anchor"]
        passes = (
            name != "one_control"
            and main["mean"] <= -0.005
            and main["seed_sem"] > 0
            and abs(main["mean"]) / main["seed_sem"] > 3
        )
        rows["passes_carrier_gate"] = bool(passes)
        passing.append(name) if passes else None
        out["scenarios"][name] = rows
        print(
            f"{name:<20} production-anchor "
            f"{main['mean']:+.6f} +/- {main['seed_sem']:.6f}; "
            f"carrier_gate={'PASS' if passes else 'fail'}"
        )
    out["gate"]["passing_crowded_scenes"] = passing
    out["gate"]["overall_pass"] = len(passing) >= 2

    target = os.path.abspath(args.output_json)
    if os.path.exists(target):
        raise FileExistsError(target)
    with open(target, "x", encoding="utf-8") as handle:
        json.dump(out, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"overall carrier gate: {'PASS' if out['gate']['overall_pass'] else 'FAIL'}")
    print(f"wrote {target}")


if __name__ == "__main__":
    main()
