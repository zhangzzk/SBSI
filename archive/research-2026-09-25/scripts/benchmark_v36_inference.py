#!/usr/bin/env python
"""Bounded V3.6 density-kernel benchmark, not a shear/calibration result."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

from sbsi.catalogue_disk_response import CatalogueDiskResponse
from sbsi.crowding import FLOW_FEATURES
from sbsi.disk_response_transport import transported_physical_log_prob
from sbsi.measurement_model import load_measurement_model
from sbsi.models import V36_LIKE, load_disk_response
from sbsi.models import load_detection_classifier
from sbsi.crowding import FEATURES, classifier_features
from sbsi.shear_map import apply_shear_to_ellipticity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--observations", type=int, default=16)
    parser.add_argument("--draws", type=int, default=16384)
    args = parser.parse_args()
    if args.output.exists() or args.observations < 1 or args.draws < 1:
        raise ValueError("new output and positive observation/draw counts required")
    torch.set_num_threads(4)
    V36_LIKE.validate()
    metadata = json.loads(args.pairs.with_suffix(".json").read_text())
    for path, digest in ((args.parent, metadata["anchor_sha256"]), (args.pairs, metadata["sha256"])):
        if sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"benchmark source hash differs: {path}")
    parent = np.load(args.parent)
    print(f"PARENT keys={parent.files}", flush=True)
    ids, context = parent["input_index"], parent["context"]
    pairs = pd.read_feather(args.pairs)
    flow = load_measurement_model(V36_LIKE.flow_checkpoints[0], device="cuda")
    response = CatalogueDiskResponse.from_pairs(ids, pairs, load_disk_response(V36_LIKE, device="cuda", nthread=4))
    raw = pd.DataFrame(context, columns=FLOW_FEATURES)
    all_context = flow.context_tensor(raw)
    rng = np.random.default_rng(20260920)
    torch.manual_seed(20260920)
    observed = flow.sample(raw.iloc[rng.choice(len(raw), args.observations)], n_samples=1)[:, 0]
    # Uniform atoms deliberately include poor candidates; report this limitation.
    atoms = rng.integers(len(ids), size=(args.observations, args.draws))
    radius = np.broadcast_to(observed[:, None, 2], atoms.shape)
    flux = np.broadcast_to(observed[:, None, 3], atoms.shape)
    torch.cuda.synchronize()
    start = time.monotonic()
    b, report = response.predict(atoms, radius, flux)
    torch.cuda.synchronize()
    report["response_seconds"] = time.monotonic() - start
    print(f"RESPONSE {json.dumps(report)}", flush=True)
    target = torch.as_tensor(np.repeat(observed, args.draws, axis=0), device="cuda", dtype=torch.float64)
    coefficients = torch.as_tensor(b.ravel(), device="cuda", dtype=torch.float64)
    context_tensor = all_context[torch.as_tensor(atoms.ravel(), device="cuda")]
    means = torch.as_tensor(flow.target_transform.means, device="cuda", dtype=torch.float64)
    scales = torch.as_tensor(flow.target_transform.scales, device="cuda", dtype=torch.float64)
    timings = []
    with torch.no_grad():
        for iteration in range(10):
            torch.cuda.synchronize()
            start = time.monotonic()
            total = 0.
            for lo in range(0, len(target), 65536):
                hi = lo + 65536
                velocity = coefficients[lo:hi, None] * torch.tensor([.02, .001], device="cuda")
                values = transported_physical_log_prob(
                    target[lo:hi], velocity,
                    lambda physical: flow.log_prob_tensor((physical-means)/scales, context_tensor[lo:hi]),
                )
                total += float(values.sum())
            torch.cuda.synchronize()
            timings.append(time.monotonic() - start)
            print(f"DENSITY iteration={iteration} seconds={timings[-1]:.4f} checksum={total}", flush=True)
    report.update({
        "status": "benchmark_complete", "gpu": torch.cuda.get_device_name(),
        "observations": args.observations, "draws": args.draws, "prior_subset_atoms": len(ids),
        "density_seconds": timings, "warm_stencil_seconds": sum(timings[1:]),
        "four_gpu_500k_kernel_hours": (report["response_seconds"] + sum(timings[1:])) * 500000 / args.observations / 4 / 3600,
        "limitations": ["Uniform atoms from one cached scene; not a production proposal.",
                        "Eager density kernel, no compilation; same contexts reused for timing only.",
                        "Excludes prior/cache construction, selection normalization, classifier states, retrieval, I/O and queue time.",
                        "Response coefficients computed once and reused across the nine stencil calls."],
    })
    # Measure the cache-building work separately, using the original CRN
    # convention. This remains a timing probe, not a population normalizer.
    detector = load_detection_classifier(V36_LIKE, device="cuda")
    n_cache = min(8192, len(ids))
    cache_context = context[:n_cache]
    cache_ids = ids[:n_cache]
    cache_pairs = pairs[pairs.index_input_p.isin(cache_ids)]
    beta, fwhm = metadata["conditions"]["moffat_beta"], metadata["conditions"]["psf_fwhm"]
    psf_radius = fwhm/2*np.sqrt((2**(1/(beta-1))-1)/(2**(1/beta)-1))
    torch.manual_seed(8201)
    torch.cuda.synchronize()
    start = time.monotonic()
    flow.sample(raw.iloc[:n_cache], n_samples=128, qmc=True)
    torch.cuda.synchronize()
    report["proposal_cache_seconds"] = time.monotonic()-start
    cache_timings = []
    for dx, dy in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
        shape = np.column_stack(apply_shear_to_ellipticity(cache_context[:, 0], cache_context[:, 1],
                                                         .02+.001*dx, .001*dy))
        torch.cuda.synchronize()
        start = time.monotonic()
        feature = classifier_features(cache_ids, cache_context, shape, cache_pairs, psf_radius)
        probability = detector.predict_proba(pd.DataFrame(feature, columns=FEATURES))
        frame = raw.iloc[:n_cache].copy()
        frame["e1_input_p"], frame["e2_input_p"] = shape[:, 0], shape[:, 1]
        torch.manual_seed(8101)
        draws = flow.sample(frame, n_samples=64, qmc=True)
        selected = (draws[..., 2] > 3.) & (draws[..., 3] > 10**(.4*(30.-25.8)))
        checksum = float(np.mean(probability*selected.mean(1)))
        torch.cuda.synchronize()
        cache_timings.append(time.monotonic()-start)
        print(f"CACHE node=({dx},{dy}) seconds={cache_timings[-1]:.4f} timing_checksum={checksum}", flush=True)
    report["cache_probe_rows"] = n_cache
    report["classifier_and_normalizer_seconds"] = cache_timings
    report["four_gpu_140m_cache_hours"] = (report["proposal_cache_seconds"]+sum(cache_timings))*140000000/n_cache/4/3600
    report["limitations"].append("Cache projection assumes 140m atoms; actual uncut count and production I/O still need verification.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"BENCHMARK_COMPLETE {args.output}", flush=True)


if __name__ == "__main__":
    main()
