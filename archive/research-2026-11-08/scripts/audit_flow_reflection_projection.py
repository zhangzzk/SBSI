#!/usr/bin/env python
"""Compare retained density to its exact four-reflection mixture on validation.

This is a diagnostic reference, not an inference wrapper or training target.
It tests whether projecting the retained density onto the requested symmetry
itself incurs the likelihood loss seen in the new single-evaluation flow.
"""
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


def main():
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument('--root', type=Path, required=True)
    a.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    args = a.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('scheduler allocation required')
    torch.set_num_threads(4)
    p = json.loads((args.root / 'diagnosis_protocol.json').read_text())
    flow = p['density_models']['old']
    if sha256(flow) != p['hashes'][flow]:
        raise ValueError('retained flow changed')
    bundle = load_measurement_model(flow, device=torch.device(args.device))
    bundle.model.eval()
    records = []
    for case in p['density_cases']:
        for leg in ('g0', 'g05'):
            path = Path(p['training']) / 'harmonized/domain/flow' / leg / f'case{case:03d}.npz'
            data = load_case(path)
            lps = []
            with torch.no_grad():
                for signs in ((1, 1), (-1, 1), (1, -1), (-1, -1)):
                    raw_x, raw_y = data['context'].copy(), data['target'].copy()
                    raw_x[:, :2] *= signs
                    raw_y[:, :2] *= signs
                    x = bundle.context_tensor(pd.DataFrame(raw_x, columns=FLOW_FEATURES))
                    y = torch.as_tensor(bundle.target_transform.transform_array(raw_y), device=args.device)
                    logp = bundle.log_prob_tensor(y, x, batch_size=8192).cpu()
                    logp -= np.log(bundle.target_transform.scales).sum()
                    if not bool(torch.isfinite(logp).all()):
                        raise ValueError('nonfinite reflected density')
                    lps.append(logp)
            lp = torch.stack(lps)
            mixture = torch.logsumexp(lp, dim=0) - np.log(4.)
            delta = (lp[0] - mixture).numpy()
            bins = np.digitize(data['context'][:, 3], [24, 25.5, 27.5])
            record = dict(case=case, leg=leg, rows=len(delta), source_sha256=sha256(path),
                old_physical_nll=float(-lp[0].mean()), projected_physical_nll=float(-mixture.mean()),
                projected_minus_old_nll=float(delta.mean()),
                individual_reflection_nll=[float(-v.mean()) for v in lp],
                magnitude_bins=[dict(rows=int((bins == k).sum()),
                    projected_minus_old_nll=float(delta[bins == k].mean())) for k in range(4)])
            records.append(record)
            write_json(args.root / 'reflection_projection.json', dict(records=records, complete=False,
                script_sha256=sha256(__file__), diagnostic_only=True))
            print('REFLECTION_PROJECTION', json.dumps(record), flush=True)
    write_json(args.root / 'reflection_projection.json', dict(records=records, complete=True,
        script_sha256=sha256(__file__), diagnostic_only=True))


if __name__ == '__main__':
    main()
