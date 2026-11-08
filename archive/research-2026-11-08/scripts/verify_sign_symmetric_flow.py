#!/usr/bin/env python
"""Verify frozen orchestration, source, domain manifests and retained artifacts."""
import hashlib
import json
from pathlib import Path
import sys


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    root = Path(sys.argv[1]).resolve()
    frozen = json.loads((root / "launch_inputs.json").read_text())
    for name, expected in frozen["files"].items():
        if digest(root / name) != expected:
            raise ValueError(f"frozen orchestration changed: {name}")
    experiment = json.loads((root / "experiment.json").read_text())
    for name, expected in experiment["checkout_hashes"].items():
        if digest(Path(experiment["checkout"]) / name) != expected:
            raise ValueError(f"source changed: {name}")
    for name, expected in frozen["domain_manifests"].items():
        if digest(root / name) != expected:
            raise ValueError(f"domain manifest changed: {name}")
    for name, expected in frozen["fixed_artifacts"].items():
        if digest(name) != expected:
            raise ValueError(f"retained artifact changed: {name}")
    import sbsi
    if Path(sbsi.__file__).resolve().parent.parent != Path(experiment["checkout"]):
        raise ValueError(f"wrong installed SBSI: {sbsi.__file__}")
    print("SIGN_SYMMETRY_FROZEN_INPUTS_VERIFIED", flush=True)


if __name__ == "__main__":
    main()
