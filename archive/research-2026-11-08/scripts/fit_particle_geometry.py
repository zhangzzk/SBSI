#!/usr/bin/env python
"""Fit and freeze physical proposal geometry from completed independent pilots.

Run this CPU fit through the local scheduler. It reads coordinates only; score
outputs do not enter the fit. It never changes the pilot outputs or target.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from sbsi.particle_geometry import fit_particle_geometry


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--multiscale", action="store_true",
                        help="Use fixed SD multipliers .25/1/4 with probabilities .2/.6/.2")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    arrays, sources, seeds, identities = [], [], [], []
    for supplied in args.runs:
        path = supplied.resolve()
        protocol_path, result_path = path / "protocol.json", path / "result.json"
        particles_path = path / "particles.npz"
        protocol = json.loads(protocol_path.read_text())
        result = json.loads(result_path.read_text())
        protocol_hash, particle_hash = digest(protocol_path), digest(particles_path)
        if (result.get("status") != "completed"
                or result.get("protocol_sha256") != protocol_hash
                or result.get("particles_sha256") != particle_hash
                or result.get("observations") != protocol["observations"]):
            raise ValueError(f"incomplete or inconsistent pilot receipt: {path}")
        identity = {key: protocol[key] for key in (
            "observations", "fixed_shear", "reference_protocol_sha256", "observation_sha256")}
        if identities and identity != identities[0]:
            raise ValueError("pilot observations, order, shear, or reference identity differ")
        identities.append(identity)
        seeds.append(protocol["seed"])
        with np.load(particles_path, allow_pickle=False) as saved:
            coordinates = saved["coordinates"]
        if coordinates.shape[:2] != (len(protocol["observations"]), protocol["particles"]):
            raise ValueError(f"pilot particle dimensions differ from protocol: {path}")
        arrays.append(coordinates)
        sources.append({"path": str(path), "seed": protocol["seed"],
                        "protocol_sha256": protocol_hash, "particles_sha256": particle_hash,
                        "result_sha256": digest(result_path)})
    import sbsi.particle_geometry as implementation
    metadata = {
        **identities[0], "pilot_seeds": seeds, "pilot_runs": sources,
        "implementation_sha256": {"particle_geometry.py": digest(implementation.__file__),
                                  "fit_particle_geometry.py": digest(__file__)},
        "production_contract": "Freeze before fresh production; reject every pilot seed via validate_production.",
    }
    geometry = fit_particle_geometry(arrays, metadata, multiscale=args.multiscale)
    geometry.save(args.output)
    print(json.dumps({"output": str(args.output.resolve()), "sha256": digest(args.output),
                      "observations": metadata["observations"], "pilot_seeds": seeds,
                      "phase_resultants": geometry.metadata["phase_resultants"],
                      "multiscale": args.multiscale}, indent=2))


if __name__ == "__main__":
    main()
