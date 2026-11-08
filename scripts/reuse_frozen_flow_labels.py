#!/usr/bin/env python
"""Reuse exact retained model-independent labels after recording regeneration.

No label values are edited. Compare fresh targets/masses/means, retain the
normal-matrix roundoff diagnostic, and copy the original hash-verified banks.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil

import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--regenerated", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    source, regenerated, root = args.reference, args.regenerated, args.root
    banks = json.loads((source / "joint/per_leg_guard_target_banks.json").read_text())
    if json.loads((source / "status/labels.json").read_text())["state"] != "completed":
        raise ValueError("retained source label stage did not complete")
    if digest(root / "harmonized/domain/manifest.json") != banks["domain_manifest_sha256"]:
        raise ValueError("label domain differs")
    if digest(root / "source_snapshot/scripts/per_leg_guard_response.py") != banks["source_sha256"]:
        raise ValueError("label recipe differs")
    manifest = json.loads((root / "harmonized/domain/manifest.json").read_text())
    differences = {}
    for split in ("train", "validation"):
        name = f"per_leg_guard_{split}_targets.json"
        if digest(source / "joint" / name) != banks["files"][name]:
            raise ValueError("retained label hash differs")
        original = json.loads((source / "joint" / name).read_text())
        fresh = json.loads((regenerated / "joint" / name).read_text())
        if original["cases"] != fresh["cases"] or original["cases"] != manifest["split"][f"{split}_cases"]:
            raise ValueError("case split differs")
        differences[split] = {}
        for key in ("target", "baseline_mass", "baseline_mean_shape", "shape_normal"):
            old, new = np.asarray(original["per_leg"][key]), np.asarray(fresh["per_leg"][key])
            if not np.isfinite(new).all() or old.shape != new.shape:
                raise ValueError("nonfinite or misaligned regenerated labels")
            differences[split][key] = dict(max_abs=float(np.max(abs(old-new))),
                max_relative=float(np.max(abs(old-new)/np.maximum(abs(old), 1e-300))))
            if key != "shape_normal":
                np.testing.assert_allclose(old, new, rtol=0, atol=2e-12)
        # Copy exact retained bytes including the normal matrix. This is not
        # accepting the regenerated matrix under a relaxed numeric tolerance.
    copied = {}
    for name in (*banks["files"], "per_leg_guard_target_banks.json", "validation_usability_response.json"):
        destination = root / "joint" / name
        if destination.exists():
            raise FileExistsError(f"preserving existing labels: {destination}")
        shutil.copy2(source / "joint" / name, destination)
        copied[name] = digest(destination)
        if copied[name] != digest(source / "joint" / name):
            raise ValueError("copied label bytes differ")
    receipt = dict(state="completed", job_id=os.environ.get("SLURM_JOB_ID"),
        source=str(source), regenerated=str(regenerated), differences=differences,
        copied_sha256=copied, domain_manifest_sha256=banks["domain_manifest_sha256"],
        method="exact retained model-independent banks; regenerated normal-matrix replay failed original absolute threshold",
        model_dependent_references_reused=False)
    (root / "status/labels.json").write_text(json.dumps(receipt, indent=2)+"\n")
    (root / "label_source_amendment.json").write_text(json.dumps(receipt, indent=2)+"\n")
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
