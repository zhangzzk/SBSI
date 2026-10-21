#!/usr/bin/env python
"""Freeze coefficients from the four declared v1 phase diagnostics only."""
import argparse
import json
from pathlib import Path

from sbsi.phase_control import (FrozenPhaseControl, digest, fit_frozen_phase_control,
                                load_phase_diagnostic)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--diagnostics', type=Path, nargs=4, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or len(set(p.resolve() for p in args.diagnostics)) != 4:
        raise ValueError('fresh output and four distinct diagnostics required')
    loaded = [load_phase_diagnostic(path) for path in args.diagnostics]
    reports, arrays = zip(*loaded)
    # This experiment authorizes only these development pilots; do not refit
    # on the evaluated v5/v6/v7 groups or turn held-out data into training.
    if [Path(r['run']).name for r in reports] != [f'dev_v1_{i}' for i in range(4)]:
        raise ValueError('n15 fitting requires ordered dev_v1_0 through dev_v1_3 only')
    if any(r['target_identity'] != reports[0]['target_identity'] or r['observations'] != reports[0]['observations']
           for r in reports[1:]):
        raise ValueError('pilot target/derivative identities differ')
    metadata = dict(observations=reports[0]['observations'], pilot_seeds=[r['seed'] for r in reports],
        target_identity=reports[0]['target_identity'], pilot_diagnostics=[dict(path=str(path.resolve()),
            sha256=digest(path), arrays_sha256=digest(path.with_suffix('.npz')),
            run=report['run'], original_sources=report['source_hashes'])
            for path, report in zip(args.diagnostics, reports)],
        fit_script_sha256=digest(__file__), estimand='continuous float64 endpoint; raw float32-sampled use remains a qualified diagnostic')
    frozen = fit_frozen_phase_control([a['raw_scores'] for a in arrays], [a['phase_control'] for a in arrays], metadata)
    frozen.save(args.output)
    print(json.dumps(dict(output=str(args.output.resolve()), sha256=digest(args.output),
                          coefficients=frozen.coefficients.tolist(), pilot_seeds=metadata['pilot_seeds']), indent=2))


if __name__ == '__main__':
    main()
