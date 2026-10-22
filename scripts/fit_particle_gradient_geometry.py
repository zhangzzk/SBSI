#!/usr/bin/env python
"""Freeze a zero-shear likelihood-gradient metric from independent pilot runs.

Submit through the scheduler; CUDA is the default. This reads the retained
V3.9 models, evaluates a separate float64 diagnostic, and writes only a fresh
proposal artifact. The production F32 likelihood and native prior are unchanged.
"""

import argparse
import hashlib
import importlib
import json
from pathlib import Path

import numpy as np

from sbsi.particle_geometry import geometry_target_identity
from sbsi.particle_gradient_geometry import fit_gradient_geometry, physical_likelihood_gradients


def digest(path):
    checksum = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            checksum.update(block)
    return checksum.hexdigest()


def load_pilots(paths):
    """Read completed immutable pilots, rejecting incompatible identities."""
    runs, sources, identity = [], [], None
    for supplied in paths:
        path = Path(supplied).resolve()
        protocol = json.loads((path / "protocol.json").read_text())
        result = json.loads((path / "result.json").read_text())
        protocol_hash, particle_hash = digest(path / "protocol.json"), digest(path / "particles.npz")
        if (result.get("status") != "completed"
                or result.get("protocol_sha256") != protocol_hash
                or result.get("particles_sha256") != particle_hash
                or result.get("observations") != protocol.get("observations")):
            raise ValueError(f"incomplete or inconsistent pilot receipt: {path}")
        current = {key: protocol[key] for key in (
            "observations", "fixed_shear", "reference", "reference_protocol_sha256", "observation_sha256")}
        current.update(geometry_target_identity(protocol))
        if current["fixed_shear"] != [0.0, 0.0]:
            raise ValueError("gradient-metric pilots require exactly zero shear")
        if identity is not None and current != identity:
            raise ValueError("pilot observations, shear, or reference identities differ")
        identity = current
        with np.load(path / "particles.npz", allow_pickle=False) as saved:
            x, values = saved["coordinates"], saved["summaries"]
        shape = (len(protocol["observations"]), protocol["particles"])
        if x.shape != (*shape, 10) or values.shape != (*shape, 9):
            raise ValueError("pilot coordinates/summaries do not align with the protocol")
        runs.append((x, values))
        sources.append({"path": str(path), "seed": protocol["seed"],
                        "protocol_sha256": protocol_hash, "particles_sha256": particle_hash,
                        "result_sha256": digest(path / "result.json")})
    if not runs:
        raise ValueError("at least one pilot run is required")
    seeds = [source["seed"] for source in sources]
    if len(set(seeds)) != len(seeds):
        raise ValueError("independent pilots require distinct seeds")
    return runs, {**identity, "pilot_seeds": seeds, "pilot_runs": sources}


def native_shape_parameters():
    """Load the same F24 row0 overrides as build_prior, without cosmology work."""
    from galsbi import load
    config = importlib.import_module("galsbi.configs.config_Fischbacher+24_intrinsic")
    row = load.load_abc_posterior("Fischbacher+24")[0]
    names = ("sersic_n_min", "sersic_n_max", "ell_beta_emax")
    values = {name: float(row[name] if name in row.dtype.names else getattr(config, name)) for name in names}
    provenance = {"model": "Fischbacher+24", "model_index": 0,
                  "config_path": str(Path(config.__file__).resolve()),
                  "config_sha256": digest(config.__file__), "values": values}
    return values, provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=4096)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.batch_size < 1:
        raise ValueError("positive diagnostic batch size required")
    runs, metadata = load_pilots(args.runs)
    reference = Path(metadata["reference"])
    if (digest(reference / "protocol.json") != metadata["reference_protocol_sha256"]
            or digest(reference / "observations.parquet") != metadata["observation_sha256"]):
        raise ValueError("pilot reference changed")
    reference_protocol = json.loads((reference / "protocol.json").read_text())
    if not reference_protocol.get("prior", "").startswith("fixed F24 row0"):
        raise ValueError("this gradient-metric launcher requires the retained F24 row0 prior")
    # Heavy optional dependencies are imported only for the scheduled fit.
    import pandas as pd
    import torch
    from sbsi.measurement_model import load_measurement_model
    from sbsi.models import V39_LIKE, load_detection_classifier
    from sbsi.summary_gradients import SummaryGradients
    import sbsi.particle_gradient_geometry as geometry_source
    import sbsi.summary_gradients as gradient_source

    torch.set_num_threads(2)
    V39_LIKE.validate()
    parameters, parameter_source = native_shape_parameters()
    observations = pd.read_parquet(reference / "observations.parquet").iloc[metadata["observations"]]
    flow = load_measurement_model(V39_LIKE.flow_checkpoints[0], device=args.device)
    detector = load_detection_classifier(V39_LIKE, device=args.device)
    flow.model.double()
    for member in getattr(detector, "members", (detector,)):
        member.model.double()
    evaluator = SummaryGradients(flow, detector, batch_size=args.batch_size)
    physical, log_kernel_ranges = [], []
    for index, (x, values) in enumerate(runs):
        owners = np.repeat(np.arange(len(observations)), x.shape[1])
        observed = observations.iloc[owners].reset_index(drop=True)
        density, raw_gradient = evaluator.evaluate(values.reshape(-1, 9), observed, [0, 0], wrt="context")
        gradient = physical_likelihood_gradients(x.reshape(-1, 10), values.reshape(-1, 9),
                                                 raw_gradient, **parameters)
        physical.append(gradient.reshape(*x.shape[:2], 5))
        log_kernel_ranges.append([float(density.min()), float(density.max())])
        print("GRADIENT_PILOT", index, "rows", len(gradient), "max_abs", float(np.max(np.abs(gradient))), flush=True)
    circular = np.mean([np.exp(1j*x[..., 9]).mean(axis=1) for x, _ in runs], axis=0)
    centres = np.where(np.abs(circular) <= 64*np.finfo(float).eps, 0.0, np.angle(circular)) % (2*np.pi)
    metadata.update({
        "parameter_source": parameter_source,
        "gradient_evaluation": "F64 F8+C8 log-K diagnostic on frozen pilot states; production F32 acceptance untouched",
        "gradient_batch_size": args.batch_size, "gradient_device": args.device,
        "diagnostic_log_kernel_ranges_by_run": log_kernel_ranges,
        "phase_centres": centres.tolist(), "phase_resultants": np.abs(circular).tolist(),
        "implementation_sha256": {str(Path(p).resolve()): digest(p) for p in (
            __file__, geometry_source.__file__, gradient_source.__file__,
            Path(__file__).resolve().parents[1] / "configs/models_v3_9_like.json")},
        "production_contract": "Freeze before fresh production; reject every pilot seed via validate_production.",
    })
    geometry = fit_gradient_geometry(physical, metadata, phase_centres=centres)
    geometry.save(args.output)
    print(json.dumps({"output": str(args.output.resolve()), "sha256": digest(args.output),
                      "pilot_seeds": metadata["pilot_seeds"], "observations": metadata["observations"],
                      "metric_condition_numbers": geometry.metadata["metric_scaled_condition_numbers"]}, indent=2))


if __name__ == "__main__":
    main()
