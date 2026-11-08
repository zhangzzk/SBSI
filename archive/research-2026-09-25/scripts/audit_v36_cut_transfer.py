#!/usr/bin/env python
"""Reanalyse frozen V3.6 response sums; isolate the 0.60--0.75 arcsec shell.

No new flow draws, model fitting, cuts for production, or archive imports.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from sbsi.disk_inference_store import file_hash, write_json


LOW = 'mag25.8_radius_gt0.60'
HIGH = 'mag25.8_radius_ge0.75'


def sums(record, cut, arm, source):
    entry = record['per_case_sufficient'][f'truth_parent/{cut}/{arm}'][source]
    return np.array([np.column_stack((np.asarray(entry[leg]['denominator'])[:, 0],
                                     np.asarray(entry[leg]['numerator'])[:, 0]))
                     for leg in ('plus', 'minus')])


def pooled_response(sufficient, h):
    return (sufficient[0, ..., 1:]/sufficient[0, ..., :1]
            - sufficient[1, ..., 1:]/sufficient[1, ..., :1])/(2*h)


def decompose(total, high, h):
    """Exact finite-step quotient identity, including selection migration."""
    shell = total-high
    if np.any(shell[..., 0] <= 0) or np.any(high[..., 0] <= 0):
        raise ValueError('both measured radius regions must have positive mass')
    f = shell[..., 0]/total[..., 0]
    means_shell = shell[..., 1:]/shell[..., :1]
    means_high = high[..., 1:]/high[..., :1]
    shell_term = f.mean(0)[..., None]*pooled_response(shell, h)
    high_term = (1-f.mean(0))[..., None]*pooled_response(high, h)
    migration = ((f[0]-f[1])/(2*h))[..., None]*(means_shell.mean(0)-means_high.mean(0))
    residual = pooled_response(total, h)-shell_term-high_term-migration
    np.testing.assert_allclose(residual, 0, atol=2e-12, rtol=0)
    return dict(shell_fraction_per_leg=f.tolist(), shell_response=pooled_response(shell, h).tolist(),
                high_response=pooled_response(high, h).tolist(),
                shell_contribution=shell_term.tolist(), high_contribution=high_term.tolist(),
                selection_migration_contribution=migration.tolist(), residual=residual.tolist())


def analyse(record, arm, weights):
    h, axis = record['h'], record['shear_axis']-1
    measured = {c: sums(record, c, arm, 'measured') for c in (LOW, HIGH)}
    model = {c: sums(record, c, arm, 'model') for c in (LOW, HIGH)}
    for values in (measured, model):
        values['shell'] = values[LOW]-values[HIGH]
    biases, responses, boot = {}, {}, {}
    for cut in (LOW, HIGH, 'shell'):
        rm = pooled_response(measured[cut].sum(1), h)[axis]
        rp = pooled_response(model[cut].sum(1), h)[axis]
        measured_boot = np.einsum('bc,lck->lbk', weights, measured[cut])
        model_boot = np.einsum('bc,lck->lbk', weights, model[cut])
        b = 100*(pooled_response(measured_boot, h)[:, axis]/pooled_response(model_boot, h)[:, axis]-1)
        biases[cut] = dict(m_percent=float(100*(rm/rp-1)), case_bootstrap_se_pp=float(np.std(b, ddof=1)),
                           case_bootstrap_ci95_percent=np.quantile(b, [.025, .975]).tolist())
        responses[cut] = dict(measured=float(rm), model=float(rp))
        boot[cut] = b
    delta = boot[HIGH]-boot[LOW]
    return dict(bias=biases, response=responses,
                paired_radius_change=dict(delta_m_pp=biases[HIGH]['m_percent']-biases[LOW]['m_percent'],
                    se_pp=float(np.std(delta, ddof=1)), ci95_pp=np.quantile(delta, [.025, .975]).tolist()),
                decomposition={name: decompose(values[LOW].sum(1), values[HIGH].sum(1), h)
                               for name, values in [('measured', measured), ('model', model)]})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--latent-repeat', action='store_true', help='Use the frozen seed7302 repeats and their receipts')
    args = p.parse_args()
    if args.output.exists():
        raise ValueError('new output required')
    report = dict(script_sha256=file_hash(__file__), bootstrap_seed=20260924, replicates=10000,
        convention='m=100*(Rmeasured/Rmodel-1); coherent antithetic extraction; pooled per-leg selected means; original per-leg U and cuts.',
        shell='MAG_AUTO<25.8 and 0.60<FLUX_RADIUS<0.75 arcsec; subtraction of nested selected sufficient sums.',
        limitations=['Reanalysis of development endpoints, not independent validation.',
            'Case bootstrap excludes finite-draw and training uncertainty.',
            'Response ratios within a measured shell do not by themselves identify which model component is wrong.',
            'Four-scene endpoints have weak case-bootstrap precision.'], endpoints={})
    for name in ('main', 'independent', 'fresh_g1', 'fresh_g2'):
        tag = '_seed7302' if args.latent_repeat else ''
        receipt_tag = '_latent_comparison' if args.latent_repeat else '_comparison'
        path = args.root/f'smooth_joint_{name}{tag}.json'
        receipt_path = args.root/f'smooth_joint_{name}{receipt_tag}.json'
        receipt = json.loads(receipt_path.read_text())
        digest = file_hash(path)
        if digest != receipt['candidate_sha256' if args.latent_repeat else 'result_sha256']:
            raise ValueError('frozen result does not match retained receipt')
        record = json.loads(path.read_text())
        n = len(record['cases'])
        weights = np.random.default_rng(20260924).multinomial(n, np.full(n, 1/n), size=10000)
        arms = {arm: analyse(record, arm, weights) for arm in ('modeled_smooth', 'actual_usable_flags')}
        report['endpoints'][name] = dict(path=str(path), sha256=digest, receipt_sha256=file_hash(receipt_path),
            cases=record['cases'], h=record['h'], axis=record['shear_axis'], draws=record['draws'], latent_seed=record['seed'], arms=arms)
        print(name, json.dumps(arms['modeled_smooth']['bias']), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, report)


if __name__ == '__main__':
    main()
