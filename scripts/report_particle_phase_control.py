#!/usr/bin/env python
"""Independent-population Stein residual and precision-measure diagnostics.

This standalone report does not modify sampling or any existing score/Z gate.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from sbsi.phase_control import (FrozenPhaseControl, digest, load_phase_diagnostic,
                                summarize_phase_populations)


def serial(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {k: serial(v) for k, v in value.items()}
    if isinstance(value, list):
        return [serial(v) for v in value]
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--diagnostics', type=Path, nargs='+', required=True)
    parser.add_argument('--coefficients', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if (args.output.exists() or len(args.diagnostics) < 2
            or len(set(p.resolve() for p in args.diagnostics)) != len(args.diagnostics)):
        raise ValueError('fresh output and at least two distinct diagnostics required')
    reports, arrays = zip(*[load_phase_diagnostic(path) for path in args.diagnostics])
    if len(set(r['seed'] for r in reports)) != len(reports):
        raise ValueError('independent populations require distinct seeds')
    if any(r['target_identity'] != reports[0]['target_identity'] or r['observations'] != reports[0]['observations']
           or r['stride'] != reports[0]['stride'] for r in reports[1:]):
        raise ValueError('diagnostic targets, observation order, or sampling strides differ')
    frozen = FrozenPhaseControl.load(args.coefficients) if args.coefficients else None
    if frozen is not None:
        for report in reports:
            frozen.validate_production(report['seed'], report['observations'], report['target_identity'])
    rows = summarize_phase_populations([r['log_normalizer'] for r in reports],
        [a['phase_control'] for a in arrays], [a['raw_scores'] for a in arrays],
        [a['delta_log_kernel'] for a in arrays], None if frozen is None else frozen.coefficients)
    for oid, row in zip(reports[0]['observations'], rows):
        row['observation'] = oid
    output = dict(schema='sbsi-independent-phase-diagnostics-v1', observations=serial(rows),
        seeds=[r['seed'] for r in reports], target_identity=reports[0]['target_identity'],
        moment_columns=['T', 'raw_g1', 'raw_g2']+([] if frozen is None else ['corrected_g1', 'corrected_g2']),
        precision_measure='F64 diagnostic: Z64hat=Z32hat*mean(exp(logK64-saved_logK32)); whole-run weights retained',
        coefficients=None if frozen is None else dict(path=str(args.coefficients.resolve()),
            sha256=digest(args.coefficients), metadata=frozen.metadata),
        sources={str(path.resolve()):digest(path) for path in args.diagnostics},
        script_sha256=digest(__file__),
        raw_score_and_normalizer_gates_unchanged=True, accepted_sampler=False,
        limitations=['T has known zero expectation for the intended continuous F64 posterior.',
                     'The raw F32-sampled expectation need not be exactly zero; both measures are retained.',
                     'Delta-method MCSE uses whole independent populations; finite runs can miss rare modes.',
                     'These diagnostics do not establish posterior stationarity or certify F32/F64 equivalence.'])
    with args.output.open('x') as handle:
        json.dump(output, handle, indent=2, allow_nan=False); handle.write('\n')
    print('PHASE_POPULATION_REPORT', args.output, flush=True)


if __name__ == '__main__':
    main()
