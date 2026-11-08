#!/usr/bin/env python
"""Stage and physical-density diagnostics using a frozen study protocol."""
import argparse
import importlib.util
import json
import os
from pathlib import Path

import numpy as np


def module(path):
    spec = importlib.util.spec_from_file_location('forward_components', path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def density(root, p, ev):
    import pandas as pd
    import torch
    from sbsi.fixed_g0_domain import FLOW_FEATURES
    from sbsi.measurement_model import load_measurement_model
    from scripts.train_fixed_g0_flow import load_case
    models = {name: load_measurement_model(path, device=torch.device('cuda'))
              for name, path in p['density_models'].items()}
    for b in models.values():
        b.model.eval()
    rows = []
    for case in p['density_cases']:
        for leg in ('g0', 'g05'):
            path = Path(p['training']) / 'harmonized/domain/flow' / leg / f'case{case:03d}.npz'
            data = load_case(path)
            frame = pd.DataFrame(data['context'], columns=FLOW_FEATURES)
            raw = data['target']
            cut = (raw[:, 2] > 3.) & (raw[:, 3] > 10**(-.4*(25.8-30)))
            bins = np.digitize(data['context'][:, 3], [24, 25.5, 27.5])
            record = dict(case=case, leg=leg, rows=len(raw), selected=int(cut.sum()),
                          source_sha256=ev.digest(path), models={})
            with torch.no_grad():
                for name, b in models.items():
                    x = b.context_tensor(frame)
                    y = torch.as_tensor(b.target_transform.transform_array(raw), device='cuda')
                    lp = b.log_prob_tensor(y, x, batch_size=8192).cpu().numpy()
                    # Model log_prob is in standardized target coordinates.
                    lp -= np.log(b.target_transform.scales).sum()
                    if not np.isfinite(lp).all():
                        raise ValueError('nonfinite physical density')
                    record['models'][name] = dict(physical_nll=float(-lp.mean()),
                        selected_physical_nll=float(-lp[cut].mean()),
                        by_magnitude=[dict(rows=int((bins == k).sum()),
                            physical_nll=float(-lp[bins == k].mean())) for k in range(4)])
            rows.append(record)
            ev.write(root / 'density.json', dict(records=rows, complete=False,
                definition='negative log density per physical e1,e2,radius_pixels,flux volume'))
            print('DENSITY_CASE', case, leg, {k:v['physical_nll'] for k,v in record['models'].items()}, flush=True)
    ev.write(root / 'density.json', dict(records=rows, complete=True,
        definition='negative log density per physical e1,e2,radius_pixels,flux volume'))


def stage_summary(root, p, ev):
    out = {}
    for stage in p['stages']:
        records = [json.loads((root / stage / 'cases' / f'case{case:03d}.json').read_text())
                   for case in p['cases']]
        totals = {k: np.array([r['sums'][k] for r in records]).sum(axis=(0, 2)) for k in records[0]['sums']}
        response = {k: (v[0, 1:]/v[0, 0] - v[1, 1:]/v[1, 0])/.04 for k,v in totals.items()}
        for prefix in ('old', 'new'):
            np.testing.assert_allclose(response[f'{prefix}_flow_C8'] + response[f'{prefix}_blend_C8'],
                                       response[f'{prefix}_C8'], rtol=0, atol=1e-11)
        out[stage] = dict(responses={k:v.tolist() for k,v in response.items()},
            bias_percent={k:float(100*(response['measured'][0]/response[k][0]-1))
                          for k in ('old_C8', 'new_C8', 'old_true_flags', 'new_true_flags')})
    ev.write(root / 'stage_summary.json', dict(cases=p['cases'], stages=out,
        purpose='stage attribution only; not checkpoint selection or calibration acceptance'))
    print('STAGE_SUMMARY', json.dumps(out), flush=True)


def main():
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument('--root', type=Path, required=True)
    args = a.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('scheduler allocation required')
    root = args.root
    p = json.loads((root / 'diagnosis_protocol.json').read_text())
    ev = module(root / 'evaluate_components.py')
    for path, expected in p['hashes'].items():
        if ev.digest(path) != expected:
            raise ValueError(f'diagnostic input changed: {path}')
    density(root, p, ev)
    for stage, flow in p['stages'].items():
        local = root / stage
        protocol = ev.verify(local)
        if protocol['flow'] != flow or protocol['cases'] != p['cases']:
            raise ValueError('stage protocol mismatch')
        ev.evaluate(local, protocol, 0)
    stage_summary(root, p, ev)


if __name__ == '__main__':
    main()
