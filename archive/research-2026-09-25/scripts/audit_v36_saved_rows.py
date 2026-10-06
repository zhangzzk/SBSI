#!/usr/bin/env python
"""Reconcile saved row-level diagnostics with normalized per-leg calibration."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path

import numpy as np

from sbsi.disk_inference_store import file_hash, write_json
from scripts.compare_v36_saved_models import CACHE, SOURCES, normalized_response

EDGES = np.array([-np.inf, 24., 25.5, 27.5, np.inf])
ARMS = ('measured', 'modeled_smooth', 'actual_usable_flags', 'model_common_usable')


def response_bin_contributions(sums, h=.02):
    """Exact allocation of global normalized response to additive bin sums."""
    s = np.asarray(sums, float)
    total = s.sum(axis=-2)
    dbar = total[..., 0].mean(axis=0)
    abar = total[..., 1:].mean(axis=0)
    product = total[0, ..., 0]*total[1, ..., 0]
    hnum = (s[0, ..., 1:]-s[1, ..., 1:])/(2*h)
    bden = (s[0, ..., 0]-s[1, ..., 0])/(2*h)
    contributions = (dbar[..., None, None]*hnum-abar[..., None, :]*bden[..., None])/product[..., None, None]
    np.testing.assert_allclose(contributions.sum(axis=-2), normalized_response(total, h), atol=2e-12, rtol=0)
    return contributions


def reduce_case(path):
    path = Path(path)
    case = int(path.stem[4:])
    with np.load(path) as z:
        ids = z['input_index']
        if len(np.unique(ids)) != len(ids):
            raise ValueError('duplicate identities')
        parent = z['cohort_truth_parent']
        mag = z['context'][:, 3]
        bins = np.digitize(mag, EDGES[1:-1])
        cut_names = z['cut_names'].tolist()
        ki = cut_names.index('mag25.8_radius_gt0.60')
        common = z['usable_plus'] & z['usable_minus']
        rows = {a:[] for a in ARMS}
        counts = dict(parent=int(parent.sum()), only_plus=int((parent & z['usable_plus'] & ~z['usable_minus']).sum()),
                      only_minus=int((parent & z['usable_minus'] & ~z['usable_plus']).sum()))
        for leg in ('plus', 'minus'):
            values, usable = z[f'measured_{leg}'], z[f'usable_{leg}']
            selected = usable & (values[:,2] > 3.) & (values[:,3] > 10**(.4*(30-25.8)))
            mass = z[f'mass_{leg}'][:,ki]
            num = z[f'flow_{leg}'][:,ki]+z[f'blend_{leg}'][:,ki]
            for arm in ARMS:
                if arm == 'measured':
                    weight = parent & selected
                    row = np.column_stack((weight.astype(float), np.where(weight[:,None], values[:,:2], 0.)))
                else:
                    weight = parent*(z[f'probability_modeled_smooth_{leg}'] if arm == 'modeled_smooth' else
                                     usable if arm == 'actual_usable_flags' else common)
                    row = np.column_stack((weight*mass, weight[:,None]*num))
                if not np.isfinite(row).all():
                    raise ValueError('nonfinite sufficient sums')
                rows[arm].append(np.array([row[bins == i].sum(0) for i in range(len(EDGES)-1)]).tolist())
    return dict(case=case, source=str(path), sha256=file_hash(path), counts=counts, sums=rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', choices=['unbounded_rows','closecrowd','closecrowd_guard_ep03'], required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--workers', type=int, default=4)
    args = p.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or args.output.exists():
        raise ValueError('scheduler allocation and new output required')
    root = (CACHE/SOURCES[args.model][0]).parent
    files = sorted(root.glob('cases_*/case*.npz'))
    if sorted(int(f.stem[4:]) for f in files) != list(range(40,120)):
        raise ValueError('expected exactly 80 cases')
    records = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for r in pool.map(reduce_case, files):
            records.append(r)
            print(f'ROW_REPLAY {args.model} case={r["case"]}', flush=True)
    records.sort(key=lambda r:r['case'])
    sums = {a:np.array([r['sums'][a] for r in records]).transpose(1,0,2,3) for a in ARMS}
    # Validate exact per-case sums against the original evaluator, independently
    # of the later faulty diagnostic. No learned model is invoked here.
    key = 'truth_parent/mag25.8_radius_gt0.60/'
    replay = {}
    for relative in SOURCES[args.model]:
        path = CACHE/relative
        d = json.loads(path.read_text())
        replay[str(path)] = file_hash(path)
        for arm in ARMS[:-1]:
            source, branch = ('measured', 'modeled_smooth') if arm == 'measured' else ('model', arm)
            for li, leg in enumerate(('plus','minus')):
                e = d['per_case_sufficient'][key+branch][source][leg]
                expected = np.column_stack((np.asarray(e['denominator']).reshape(-1), np.asarray(e['numerator']).reshape(-1,2)))
                actual = sums[arm][li, np.array(d['cases'])-40].sum(1)
                np.testing.assert_allclose(actual, expected, rtol=2e-10, atol=2e-6)
        del d
    total = {a:v.sum(1) for a,v in sums.items()}
    rng = np.random.default_rng(20260924)
    weights = rng.multinomial(80,np.full(80,1/80),size=10000)
    boots = {a:np.einsum('bc,lckv->lbkv', weights,v) for a,v in sums.items()}
    data_parts = response_bin_contributions(total['measured'])
    data_boot = response_bin_contributions(boots['measured'])
    result = {}
    for arm in ARMS[1:]:
        model_r = normalized_response(total[arm].sum(1))[0]
        model_rb = normalized_response(boots[arm].sum(2))[:,0]
        contrib = 100*(data_parts-response_bin_contributions(total[arm]))[:,0]/model_r
        cb = 100*(data_boot-response_bin_contributions(boots[arm]))[...,0]/model_rb[:,None]
        result[arm] = dict(m_percent=float(contrib.sum()), case_se_pp=float(cb.sum(1).std(ddof=1)),
            bin_contributions_pp=contrib.tolist(), bin_case_se_pp=cb.std(0,ddof=1).tolist(),
            sum_absolute_contributions_pp=float(np.abs(contrib).sum()),
            selected_fractions_data=(total['measured'][:,:,0].sum(0)/total['measured'][:,:,0].sum()).tolist(),
            selected_fractions_model=(total[arm][:,:,0].sum(0)/total[arm][:,:,0].sum()).tolist())
    raw = {a:(v[0,:,1].sum()-v[1,:,1].sum())/.04 for a,v in total.items()}
    wrong = 100*(raw['measured']/raw['model_common_usable']-1)
    out = dict(model=args.model, bins=['r<24','24<=r<25.5','25.5<=r<27.5','r>=27.5'],
        results=result, historical_unnormalized_mixed_usability_m_percent=float(wrong),
        per_case=records, original_evaluator_replay=replay, script_sha256=file_hash(__file__),
        limitations=['Truth bins diagnose error; all bins retained in headline.',
            'Exact additive allocation of normalized responses, not causal self/neighbour attribution.',
            'Existing development scenes; excludes integration and training uncertainty.'])
    write_json(args.output,out)
    print(json.dumps({k:v for k,v in out.items() if k not in ('per_case','original_evaluator_replay')},indent=2),flush=True)


if __name__ == '__main__':
    main()
