#!/usr/bin/env python
"""Compare frozen candidate sufficient sums with independent per-leg normalization."""
import argparse
import gc
import json
import os
from pathlib import Path

import numpy as np

from sbsi.disk_inference_store import file_hash, write_json

CACHE = Path('/project/ls-gruen/users/zekang.zhang/sbsi_caches')
SOURCES = {
    'retained_narrow': ['scene_classifier_geometry_20260918_v1/smooth_joint_main.json'],
    'retained_reproduction': ['joint_m_mag265_control_20260922_v1/main_r26flow_reproduction.json'],
    'mag265': ['joint_m_mag265_control_20260922_v1/main_r265flow_control.json'],
    'mag265_full_parent': ['constgold_truth_parent_unbounded_20260922_v1/measurement/main_unbounded_r265flow.json'],
    'unbounded': ['joint_m_unbounded_retrain_20260923_v1/main_unbounded_retrained_flow_classifier.json'],
    'unbounded_rows': ['joint_m_unbounded_retrain_20260923_v1/rows/main_rows_a.json',
                       'joint_m_unbounded_retrain_20260923_v1/rows/main_rows_b.json'],
    'fluxsize_ep08': ['joint_m_fluxsize_ep08_20260923_v1/rows/main_rows_a.json',
                     'joint_m_fluxsize_ep08_20260923_v1/rows/main_rows_b.json'],
    'guard8_ep02': ['joint_m_guardstrata_20260923_v1/rows/main_rows_a.json',
                   'joint_m_guardstrata_20260923_v1/rows/main_rows_b.json'],
    'closecrowd': ['joint_m_closecrowd_20260924_v1/rows/main_rows_a.json',
                  'joint_m_closecrowd_20260924_v1/rows/main_rows_b.json'],
    'closecrowd_guard_ep03': ['joint_m_cc_gstrata_20260924_v1/rows/main_rows_a.json',
                            'joint_m_cc_gstrata_20260924_v1/rows/main_rows_b.json'],
}
ARMS = ('modeled_smooth', 'actual_usable_flags', 'matched_usable')
CUTS = ('mag25.8_radius_gt0.60', 'mag25.8_radius_ge0.75')


def normalized_response(sums, h=.02):
    """Sums axes: leg (+,-), optional samples/cases, [count, e1, e2]."""
    sums = np.asarray(sums, dtype=float)
    if sums.shape[0] != 2 or sums.shape[-1] != 3 or not np.isfinite(sums).all():
        raise ValueError('finite two-leg sufficient sums required')
    if np.any(sums[..., 0] <= 0) or h <= 0:
        raise ValueError('positive mass and shear step required')
    mean = sums[..., 1:] / sums[..., :1]
    return (mean[0] - mean[1]) / (2*h)


def extract(entry, source, n):
    legs = []
    for leg in ('plus', 'minus'):
        v = entry[source][leg]
        den = np.asarray(v['denominator'], float).reshape(n, -1)
        num = np.asarray(v['numerator'], float).reshape(n, -1)
        if den.shape != (n, 1) or num.shape != (n, 2):
            raise ValueError('unexpected sufficient sum shape')
        legs.append(np.column_stack((den, num)))
    return np.stack(legs)


def estimate(data, model, weights):
    dr, mr = normalized_response(data.sum(1)), normalized_response(model.sum(1))
    db = normalized_response(np.einsum('bc,lck->lbk', weights, data))
    mb = normalized_response(np.einsum('bc,lck->lbk', weights, model))
    boot = 100*(db[:, 0]/mb[:, 0]-1)
    pooled = dict(data_response=dr.tolist(), model_response=mr.tolist(),
                  m_percent=float(100*(dr[0]/mr[0]-1)),
                  case_se_pp=float(boot.std(ddof=1)),
                  ci95_pp=np.quantile(boot, [.025, .975]).tolist(),
                  selected_data=data.sum(1)[:, 0].tolist(),
                  selected_model=model.sum(1)[:, 0].tolist(),
                  selected_count_ratio=(model.sum(1)[:, 0]/data.sum(1)[:, 0]).tolist())
    return pooled, boot


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('saved science outputs must be reduced under Slurm')
    if args.output.exists():
        raise FileExistsError(args.output)
    outputs, arrays, bootstrap = {}, {}, {}
    weights = np.random.default_rng(20260924).multinomial(80, np.full(80, 1/80), size=10000)
    for name, files in SOURCES.items():
        stored, cases, provenance = {}, [], []
        for relative in files:
            path = CACHE/relative
            d = json.loads(path.read_text())
            n = len(d['cases'])
            if d['h'] != .02 or d.get('shear_axis', 1) != 1:
                raise ValueError('inconsistent shear convention')
            cases.extend(d['cases'])
            inp = d['inputs']
            ex = d.get('experimental_response')
            provenance.append(dict(path=str(path), sha256=file_hash(path), cases=d['cases'],
                population=d['population'], draws=d['draws'], seed=d['seed'],
                inputs={k:v for k,v in inp.items() if k in ('flow', 'flow_sha256',
                    'classifier_arms', 'script_sha256', 'parent_manifest_sha256')},
                response_metadata=None if ex is None else ex['metadata'],
                composition=None if ex is None else ex.get('composition'),
                case_reports=d['case_reports']))
            for cut in CUTS:
                for arm in ARMS:
                    key = f'truth_parent/{cut}/{arm}'
                    if key not in d['per_case_sufficient']:
                        continue
                    e = d['per_case_sufficient'][key]
                    for source in ('measured', 'model'):
                        stored.setdefault((cut, arm, source), []).append(extract(e, source, n))
            del d
            gc.collect()
        if sorted(cases) != list(range(40, 120)):
            raise ValueError(f'{name}: missing/duplicate cases')
        order = np.argsort(cases)
        arrays[name] = {k:np.concatenate(v, axis=1)[:, order] for k,v in stored.items()}
        result, boots = {}, {}
        for arm in ARMS:
            for cut in (*CUTS, 'shell_060_075'):
                if (CUTS[0], arm, 'measured') not in arrays[name]:
                    continue
                vals = []
                for src in ('measured', 'model'):
                    v = arrays[name][CUTS[0], arm, src]-arrays[name][CUTS[1], arm, src] if cut == 'shell_060_075' else arrays[name][cut, arm, src]
                    vals.append(v)
                result[f'{cut}/{arm}'], boots[f'{cut}/{arm}'] = estimate(*vals, weights)
        outputs[name] = dict(sources=provenance, results=result)
        bootstrap[name] = boots
        print(name, json.dumps({k:v for k,v in result.items() if k.startswith(CUTS[0])}), flush=True)
    paired = {}
    for name in SOURCES:
        if name == 'unbounded_rows':
            continue
        paired[name] = {}
        for arm in ARMS:
            key = f'{CUTS[0]}/{arm}'
            identity = np.array_equal(arrays[name][CUTS[0], arm, 'measured'],
                                      arrays['unbounded_rows'][CUTS[0], arm, 'measured'])
            if identity:
                shift = bootstrap[name][key]-bootstrap['unbounded_rows'][key]
                paired[name][arm] = dict(measured_sums_identical=True,
                    delta_m_pp=outputs[name]['results'][key]['m_percent']-outputs['unbounded_rows']['results'][key]['m_percent'],
                    case_se_pp=float(shift.std(ddof=1)), ci95_pp=np.quantile(shift, [.025, .975]).tolist())
            else:
                paired[name][arm] = dict(measured_sums_identical=False,
                    warning='Different measured sums; not a controlled model-only shift.')
    write_json(args.output, dict(script_sha256=file_hash(__file__), models=outputs,
        paired_vs_unbounded_rows=paired, cases=list(range(40,120)),
        convention='Per-leg normalized selected means, h=0.02 g1, m=100*(Rdata/Rmodel-1); primary measured radius>0.6 arcsec and MAG_AUTO<25.8.',
        limitations=['Saved development scenes, no independent acceptance.',
            'Matched-usable arm is diagnostic only, not the requested estimator.',
            'Parent populations and component identities must be read from each source.',
            'Uncertainties exclude latent and training variation.']))


if __name__ == '__main__':
    main()
