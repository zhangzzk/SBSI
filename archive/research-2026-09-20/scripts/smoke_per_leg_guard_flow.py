"""Check per-leg gradient replay on a frozen physical flow; never update weights."""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sbsi.fixed_g0_domain import FLOW_FEATURES
from sbsi.flow_guard_response import independent_guard_score, make_guard_cuts, sampled_guard_response_backward
from sbsi.measurement_model import load_measurement_model
from scripts.audit_harmonized_usability_response import load_aligned
from scripts.per_leg_guard_response import PerLegGuardResponsePopulation
from scripts.train_fixed_g0_flow import sha256, write_json
from scripts.train_full_domain_guard_flow import MAGNITUDE_THRESHOLDS, RADIUS_THRESHOLDS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--domain-root', type=Path, required=True)
    parser.add_argument('--flow', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('gradient smoke requires Slurm')
    if args.output.exists():
        raise FileExistsError(args.output)
    manifest_path = args.domain_root/'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    case = manifest['split']['validation_cases'][0]
    legs, provenance = load_aligned(args.domain_root, manifest, case)
    rows = np.sort(np.random.default_rng(2026091805).choice(len(legs['g0']['usable']), 2048, replace=False))
    flow_hash = sha256(args.flow)
    bundle = load_measurement_model(args.flow, device=torch.device('cpu'))
    model = bundle.model
    model.eval()
    context = []
    for leg in ('g0', 'g05'):
        path = args.domain_root/f'usability/{leg}/case{case:03d}.npz'
        if sha256(path) != provenance[leg]['usability_sha256']:
            raise ValueError('context changed during smoke')
        with np.load(path) as data:
            raw = pd.DataFrame(data['context'][rows], columns=FLOW_FEATURES)
        context.append(torch.as_tensor(bundle.condition_preprocessor.transform_frame(raw), dtype=torch.float32))
    cuts = make_guard_cuts(RADIUS_THRESHOLDS, MAGNITUDE_THRESHOLDS)
    pop = PerLegGuardResponsePopulation(*context, legs['g0']['values'][rows], legs['g05']['values'][rows],
        legs['g05']['gamma'][rows], legs['g0']['usable'][rows], legs['g05']['usable'][rows], cuts,
        radius_softness=.05, magnitude_softness=.05, response_scale=.01)
    count, chunk, draws, seed = 128, 64, 4, 2026091806
    model.zero_grad(set_to_none=True)
    replay = sampled_guard_response_backward(model, pop, count, draws, chunk_size=chunk, seed=seed, independent_pairs=True)
    parameters = dict(model.named_parameters())
    saved = {name: None if p.grad is None else p.grad.detach().clone() for name, p in parameters.items()}
    model.zero_grad(set_to_none=True)
    rng = np.random.default_rng(seed)
    indices = [torch.as_tensor(rng.integers(len(rows), size=count)) for _ in range(2)]
    means = []
    for stream in (0, 1):
        pieces = [pop.contributions(model, indices[stream][start:start+chunk], draws, seed+100000000*stream+block)
                  for block, start in enumerate(range(0, count, chunk))]
        means.append(torch.cat(pieces).mean(0))
    direct = independent_guard_score(*means, pop.target, pop.response_scale)
    direct.backward()
    np.testing.assert_allclose(replay['loss'], float(direct.detach()), rtol=1e-10, atol=1e-10)
    norms, differences, none = 0., 0., []
    for name, p in parameters.items():
        old, new = saved[name], p.grad
        if (old is None) != (new is None):
            raise RuntimeError(f'gradient participation differs: {name}')
        if old is None:
            none.append(name)
            continue
        if not bool(torch.isfinite(old).all() and torch.isfinite(new).all()):
            raise ValueError(f'nonfinite physical-flow gradient: {name}')
        norms += float(new.double().square().sum())
        differences += float((new.double()-old.double()).square().sum())
    relative = np.sqrt(differences)/max(np.sqrt(norms), 1e-30)
    tolerance = 1e-12 if all(p.dtype == torch.float64 for p in parameters.values()) else 1e-6
    if not np.isfinite(relative) or relative > tolerance or norms == 0:
        raise RuntimeError(f'gradient replay error {relative} exceeds {tolerance} or gradient vanishes')
    if sha256(args.flow) != flow_hash:
        raise ValueError('frozen checkpoint changed')
    write_json(args.output, {'status': 'passed', 'smoke': True, 'optimizer_steps': 0, 'case': case,
        'parent_rows': len(rows), 'usable_counts': pop.statistics['usable_counts'],
        'row_seed': 2026091805, 'latent_row_seed': seed, 'pairs_per_group': count, 'draws': draws,
        'independent_row_groups': True, 'direct_loss': float(direct.detach()),
        'replay_loss': replay['loss'], 'gradient_l2': float(np.sqrt(norms)),
        'gradient_relative_l2_error': float(relative), 'gradient_tolerance': tolerance,
        'parameters_without_gradient': none, 'flow_sha256': flow_hash,
        'domain_manifest_sha256': sha256(manifest_path), 'source_provenance': provenance,
        'script_sha256': sha256(__file__), 'guard_adapter_sha256': sha256(PerLegGuardResponsePopulation.__init__.__code__.co_filename),
        'limitations': ['small operational subset; response score is not a population estimate',
                        'no optimization, checkpoint selection or calibration claim']})
    print(f'PHYSICAL_FLOW_PER_LEG_REPLAY_PASSED relative_gradient_error={relative:.3g} tolerance={tolerance:.3g}', flush=True)


if __name__ == '__main__':
    main()
