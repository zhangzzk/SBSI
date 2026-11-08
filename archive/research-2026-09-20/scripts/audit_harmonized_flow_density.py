"""Evaluate one frozen flow on old and harmonized measurement targets."""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sbsi.fixed_g0_domain import FLOW_FEATURES
from sbsi.measurement_model import load_measurement_model
from scripts.train_fixed_g0_flow import load_case, sha256, write_json


@torch.no_grad()
def main():
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('flow density audit requires Slurm')
    cache = Path('/project/ls-gruen/users/zekang.zhang/sbsi_caches')
    root = cache/'output_conditioned_response_20260917_v1'
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--domain-root', type=Path, default=root/'harmonized_pilot_domain')
    parser.add_argument('--output', type=Path, default=root/'harmonized_pilot_density.json')
    parser.add_argument('--cases', type=int, nargs='+', default=[0, 40, 41])
    args = parser.parse_args()
    if len(set(args.cases)) != len(args.cases) or not args.cases or any(c < 0 or c >= 200 for c in args.cases):
        raise ValueError('unique training-domain cases0--199 required')
    output = args.output
    if output.exists():
        raise FileExistsError(output)
    domains = [cache/'full_domain_true_mag26_guard_v1/domain', args.domain_root]
    manifests = [json.loads((path/'manifest.json').read_text()) for path in domains]
    flow = cache/'full_domain_true_mag26_guard_v1/flow/guard/selected.pt'
    bundle = load_measurement_model(flow, device=torch.device('cuda'))
    reports = []
    for case in args.cases:
        for leg in ('g0', 'g05'):
            ids, losses = [], []
            for domain, manifest in zip(domains, manifests):
                path = domain/f'flow/{leg}/case{case:03d}.npz'
                if sha256(path) != manifest['flow'][f'{leg}_cases'][str(case)]['sha256']:
                    raise ValueError('changed flow-domain targets')
                rows = load_case(path)
                context = bundle.context_tensor(pd.DataFrame(rows['context'], columns=FLOW_FEATURES))
                target = torch.as_tensor(bundle.target_transform.transform_array(rows['target']), device='cuda')
                loss = -bundle.log_prob_tensor(target, context, batch_size=8192).cpu().numpy()
                if not np.isfinite(loss).all():
                    raise ValueError('nonfinite density on valid targets')
                ids.append(rows['input_index'])
                losses.append(loss)
            shared, oi, ni = np.intersect1d(ids[0], ids[1], return_indices=True)
            paired = losses[1][ni]-losses[0][oi]
            report = {'case': case, 'leg': leg,
                      'split': 'validation' if case in manifests[1]['split']['validation_cases'] else 'train',
                      'old_rows': len(ids[0]), 'new_rows': len(ids[1]), 'shared_rows': len(shared),
                      'old_only': len(ids[0])-len(shared), 'new_only': len(ids[1])-len(shared),
                      'old_full_nll': float(losses[0].mean()), 'new_full_nll': float(losses[1].mean()),
                      'paired_nll_difference': float(paired.mean()),
                      'paired_object_se': float(paired.std(ddof=1)/np.sqrt(len(shared)))}
            reports.append(report)
            print(f'HARMONIZED_DENSITY {report}', flush=True)
    write_json(output, {'reports': reports, 'cases': args.cases, 'flow_sha256': sha256(flow),
                        'domain_manifest_sha256': [sha256(path/'manifest.json') for path in domains],
                        'script_sha256': sha256(__file__),
                        'density_measure': 'same frozen checkpoint standardized physical-target measure in both arms',
                        'limitations': ['reused training-domain scenes, with original train/validation split; not calibration acceptance',
                                        'paired object errors condition on each scene',
                                        'a different data likelihood is not a trained-model improvement']})
    print(f'HARMONIZED_DENSITY_COMPLETE {output}', flush=True)


if __name__ == '__main__':
    main()
