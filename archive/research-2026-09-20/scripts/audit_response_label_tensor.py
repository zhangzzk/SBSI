"""Audit directional response information discarded by scalar emulator labels.

Uses the already frozen BlendEMU preparation, without fitting any model.
Random-direction labels identify 2*d_a*u_b in expectation; finite forward
amplitude and both-leg usability remain limitations. This is not a coherent
small-shear calibration measurement.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from scripts.train_fixed_g0_flow import write_json


def directional_labels(parallel, cross, angle_degrees):
    angle = np.deg2rad(np.asarray(angle_degrees, dtype=float)) * 2
    c, s = np.cos(angle), np.sin(angle)
    direction = np.stack((c, s), axis=-1)
    delta = np.stack((parallel*c-cross*s, parallel*s+cross*c), axis=-1)
    return 2*delta[..., :, None]*direction[..., None, :], direction


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("full response-label scan requires Slurm")
    if args.output.exists():
        raise FileExistsError(args.output)
    manifest_path = args.prepared / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    parts = {}
    for part in manifest["parts"]:
        parts.setdefault(part["case"], []).append(part)
    reports = []
    for case, blocks in sorted(parts.items()):
        sums = np.zeros((3, 9))
        primary_ids = [[] for _ in range(3)]
        for part in blocks:
            path = args.prepared / part["path"]
            if hashlib.sha256(path.read_bytes()).hexdigest() != part["sha256"]:
                raise ValueError(f"prepared part hash mismatch: {path}")
            with np.load(path) as data:
                x, y, cross = data["x"], data["y"].astype(float), data["null"].astype(float)
                tensor, u = directional_labels(y, cross, data["shear_angle"])
                if not np.isfinite(tensor).all():
                    raise ValueError("invalid directional labels")
                # The tensor trace/2 must reproduce the existing scalar label.
                np.testing.assert_allclose(np.trace(tensor, axis1=1, axis2=2)/2, y, atol=1e-12)
                cuts = (np.ones(len(y), dtype=bool),
                        (x[:, 7] > np.log(3)) & (x[:, 8] < 25.8),
                        (x[:, 7] >= np.log(3.75)) & (x[:, 8] < 25.8))
                gram = u[:, :, None]*u[:, None, :]
                for i, keep in enumerate(cuts):
                    sums[i, 0] += keep.sum()
                    sums[i, 1:5] += tensor[keep].sum(0).reshape(-1)
                    sums[i, 5:9] += gram[keep].sum(0).reshape(-1)
                    primary_ids[i].append(data["primary"][keep])
        counts = [len(np.unique(np.concatenate(ids))) for ids in primary_ids]
        reports.append({"case": case, "sums": sums, "represented_primary_counts": counts})
        if case % 20 == 0:
            print(f"RESPONSE_TENSOR case={case} rows={int(sums[0, 0])}", flush=True)
    summaries = {}
    rng = np.random.default_rng(20260917)
    for split, lo, hi in (("fit", 40, 200), ("development", 0, 40)):
        selected = [row for row in reports if lo <= row["case"] < hi]
        values = np.asarray([row["sums"] for row in selected])
        counts = np.asarray([row["represented_primary_counts"] for row in selected])
        sample = rng.integers(len(selected), size=(10000, len(selected)))
        rows = []
        for i, cut in enumerate(("uncut", "radius_gt_3_mag_lt_25.8", "radius_ge_3.75_mag_lt_25.8")):
            total = values[:, i].sum(0)
            tensor = total[1:5].reshape(2, 2)/counts[:, i].sum()
            boot = values[sample, i, 1:5].sum(1)/counts[sample, i].sum(1)[:, None]
            halfdiff = (boot[:, 0]-boot[:, 3])/2
            rows.append({"cut": cut, "pairs": int(total[0]),
                         "represented_primaries": int(counts[:, i].sum()),
                         "tensor_per_represented_primary": tensor,
                         "tensor_case_bootstrap_se": boot.std(0, ddof=1).reshape(2, 2),
                         "R11_minus_scalar": float((tensor[0, 0]-tensor[1, 1])/2),
                         "R11_minus_scalar_case_se": float(halfdiff.std(ddof=1)),
                         "R11_minus_scalar_case_ci95": np.quantile(halfdiff, [.025, .975]),
                         "direction_gram_per_pair": total[5:9].reshape(2, 2)/total[0]})
        summaries[split] = rows
    write_json(args.output, {"manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                            "summaries": summaries, "cases": reports,
                            "estimand": "2*d_a*u_b from forward h=0.2 random-direction labels; sum over pairs divided by represented primary count",
                            "limitations": ["finite forward h=0.2", "both-leg label usability",
                                            "primaries without prepared pairs excluded",
                                            "pair sum contains other-neighbour instrument noise",
                                            "different population from native coherent-shear scenes",
                                            "case bootstrap excludes model-training uncertainty"]})
    print("RESPONSE_TENSOR_COMPLETE", json.dumps(summaries, default=lambda x: x.tolist()), flush=True)


if __name__ == "__main__":
    main()
