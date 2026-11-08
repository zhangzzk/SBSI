#!/usr/bin/env python
"""Audit physical generation, density, inverse and scores on real F8 contexts."""
import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sbsi.disk_precision import physical_log_prob_with_gap
from sbsi.measurement_model import build_flow, load_measurement_model, save_measurement_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--flow", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("real-context audit requires scheduler allocation")
    from sbsi.fixed_g0_domain import FLOW_FEATURES
    from scripts.train_fixed_g0_flow import build_transforms_and_config, load_case
    data = load_case(args.root / "base/domain/flow/g0/case000.npz")
    raw = data["context"][:1024].copy()
    torch.manual_seed(20261026)
    if args.flow is None:
        condition, target, config = build_transforms_and_config(dict(
            context_raw=raw, target_raw=data["target"][:1024]))
        model = build_flow(config)
        # Avoid the trivial identity initialization when testing equivariance.
        with torch.no_grad():
            for layer in model.base.layers:
                layer.net[-1].weight.normal_(std=.003)
                layer.net[-1].bias.normal_(std=.003)
        path = args.root / "preflight_model.pt"
        save_measurement_model(path, model, condition, target, config,
            metadata={"purpose": "random nonidentity preflight, not a trained checkpoint"})
    else:
        path = args.flow
    bundle = load_measurement_model(path, device="cuda")
    from sbsi.flow_sign_symmetric import SignSymmetricPhysicalDiskFlow
    if not isinstance(bundle.model, SignSymmetricPhysicalDiskFlow):
        raise ValueError("audit requires the sign-symmetric candidate")
    raw[:6, :2] = [[0, 0], [0, .1], [.1, 0], [1e-8, -1e-8], [-1e-8, 1e-8], [.1, -.1]]
    context = bundle.context_tensor(pd.DataFrame(raw, columns=FLOW_FEATURES))
    z = torch.randn(len(raw), 4, device="cuda") * .5
    model = bundle.model
    with torch.no_grad():
        y, ld, gap = model.forward_physical(z, context, return_shape_log_gap=True)
        density = physical_log_prob_with_gap(model, y, gap, context)
        means = torch.cat((model.disk_shape_means, model.physical_means))
        scales = torch.cat((model.disk_shape_scales, model.physical_scales))
        recovered, inverse_ld = model.inverse((y-means)/scales, context)
        inverse_error = float((recovered-z).abs().max())
        ld_error = float((inverse_ld+ld-scales.log().sum()).abs().max())
        if inverse_error > 2e-4 or ld_error > 2e-4:
            raise ValueError(f"inverse replay failed: {inverse_error}, {ld_error}")
    context = context.detach().requires_grad_(True)
    score = torch.autograd.grad(physical_log_prob_with_gap(model, y.detach(), gap.detach(), context).sum(), context)[0]
    records = []
    for signs in itertools.product((-1., 1.), repeat=2):
        action = torch.tensor([*signs, 1., 1.], device="cuda")
        context_action = torch.tensor([*signs, *([1.] * (context.shape[1]-2))], device="cuda")
        mirrored_context = (context.detach()*context_action).requires_grad_(True)
        with torch.no_grad():
            ym, ldm, gapm = model.forward_physical(z*action, mirrored_context, return_shape_log_gap=True)
        mirrored_density = physical_log_prob_with_gap(model, ym.detach(), gapm.detach(), mirrored_context)
        mirrored_score = torch.autograd.grad(mirrored_density.sum(), mirrored_context)[0]
        errors = dict(generation=float((ym-y*action).abs().max()),
            logdet=float((ldm-ld).abs().max()), log_gap=float((gapm-gap).abs().max()),
            density=float((mirrored_density.detach()-density).abs().max()),
            score=float((mirrored_score-score*context_action).abs().max()))
        if not all(np.isfinite(v) and v <= 1e-5 for v in errors.values()):
            raise ValueError(f"reflection {signs} failed: {errors}")
        records.append(dict(signs=signs, maximum_errors=errors))
    result = dict(status="passed", flow=str(path.resolve()),
        flow_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), device=torch.cuda.get_device_name(),
        observations=len(raw), cases="case000 contexts with explicit zero/near-zero shape probes",
        inverse_max_error=inverse_error, inverse_logdet_max_error=ld_error,
        reflections=records, full_joint_symmetry_claim=False,
        limitation="C8 unchanged; architecture audit is not a calibration or convergence result")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
