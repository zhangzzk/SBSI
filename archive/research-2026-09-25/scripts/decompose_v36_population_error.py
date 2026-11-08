#!/usr/bin/env python
"""Exact algebraic allocation of frozen uncut-response error, not a correction.

Separates within-bright/faint responses, selected population composition, and
finite-step selection migration. Truth strata are diagnostic only.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from sbsi.disk_inference_store import file_hash, write_json
from scripts.audit_v36_cut_transfer import pooled_response


def mixture(bright, faint, h):
    bright, faint = np.asarray(bright, float), np.asarray(faint, float)
    if (bright.shape != (2, 3) or faint.shape != (2, 3)
            or not np.isfinite([bright, faint]).all()
            or np.any(bright[:, 0] <= 0) or np.any(faint[:, 0] <= 0)
            or not np.isfinite(h) or h <= 0):
        raise ValueError('positive mass, finite two-leg sufficient sums and step required')
    total = bright+faint
    f = faint[:, 0]/total[:, 0]
    mean_f = faint[:, 1:]/faint[:, :1]
    mean_b = bright[:, 1:]/bright[:, :1]
    rf, rb = pooled_response(faint, h), pooled_response(bright, h)
    migration = (f[0]-f[1])/(2*h)*(mean_f.mean(0)-mean_b.mean(0))
    response = pooled_response(total, h)
    np.testing.assert_allclose(response, f.mean()*rf+(1-f.mean())*rb+migration, atol=2e-12, rtol=0)
    return dict(fraction=f, mean_fraction=f.mean(), faint_response=rf, bright_response=rb,
                migration=migration, response=response)


def allocation(data_bright, data_faint, model_bright, model_faint, h):
    data, model = mixture(data_bright, data_faint, h), mixture(model_bright, model_faint, h)
    fraction = (data['mean_fraction']+model['mean_fraction'])/2
    terms = dict(
        bright_response=(1-fraction)*(data['bright_response']-model['bright_response']),
        faint_response=fraction*(data['faint_response']-model['faint_response']),
        selected_composition=(data['mean_fraction']-model['mean_fraction'])*(
            (data['faint_response']+model['faint_response'])/2-
            (data['bright_response']+model['bright_response'])/2),
        selection_migration=data['migration']-model['migration'])
    residual = data['response']-model['response']-sum(terms.values())
    np.testing.assert_allclose(residual, 0., atol=2e-12, rtol=0)
    return data, model, terms, residual


def report_part(records, arm):
    def pooled(source, group):
        values = [r['measured'] if source == 'measured' else r['model'][arm] for r in records]
        result = np.array([[v[leg][group] for v in values] for leg in ('plus', 'minus')]).sum(1)
        return result
    for source in ('measured', 'model'):
        np.testing.assert_allclose(pooled(source, 'all_bright')+pooled(source, 'faint'),
                                   pooled(source, 'all'), rtol=2e-12, atol=1e-9)
    d, m, terms, residual = allocation(pooled('measured', 'all_bright'), pooled('measured', 'faint'),
                                      pooled('model', 'all_bright'), pooled('model', 'faint'), .02)
    def serial(values):
        return {k: np.asarray(v).tolist() for k, v in values.items()}
    return dict(data=serial(d), model=serial(m), delta_response=serial(terms),
        contribution_to_m11_pp={k: float(100*v[0]/m['response'][0]) for k, v in terms.items()},
        m11_percent=float(100*(d['response'][0]/m['response'][0]-1)), residual=residual.tolist())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--summary', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError('refusing overwrite')
    summary = json.loads(args.summary.read_text())
    if summary['h'] != .02 or summary['axis'] != 1:
        raise ValueError('this diagnostic requires the frozen g1 +/-0.02 pilot')
    records = []
    for name, expected in summary['source_sha256'].items():
        if file_hash(name) != expected:
            raise ValueError('frozen sufficient sums changed')
        records.append(json.loads(Path(name).read_text()))
    if sorted(r['case'] for r in records) != sorted(summary['cases']):
        raise ValueError('case membership mismatch')
    report = dict(summary_sha256=file_hash(args.summary), source_sha256=summary['source_sha256'],
        script_sha256=file_hash(__file__), h=.02, axis=1,
        allocation='Symmetric product-difference identity using mean data/model selected fraction and responses; diagnostic algebra, not a causal component intervention.',
        definition='m11=100*(Rdata/Rmodel-1); fixed measured radius>0.6 arcsec and MAG_AUTO<25.8; no truth cuts.',
        pooled={arm: report_part(records, arm) for arm in ('predicted_usable', 'actual_usable')},
        per_case={str(r['case']): {arm: report_part([r], arm) for arm in ('predicted_usable', 'actual_usable')}
                  for r in records},
        limitations=['Two development scenes, not independent calibration or a population error budget.',
            'Within-faint response error combines flow, response transport and selection effects.',
            'Holding a model component to observed labels is a diagnostic; it is not deployable inference.',
            'No truth-dependent production weight, numerical correction, or model change.'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, report)
    print(json.dumps(report['pooled'], indent=2), flush=True)


if __name__ == '__main__':
    main()
