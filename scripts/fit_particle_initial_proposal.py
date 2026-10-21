#!/usr/bin/env python
"""Fit exactly four completed independent pilots; run this CPU fit via Slurm."""
import argparse
import json
from pathlib import Path

from sbsi.particle_initial_proposal import FrozenParticleProposal, digest, fit_initial_proposal_from_runs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', nargs=4, type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    fitted = fit_initial_proposal_from_runs(args.runs)
    import sbsi.particle_initial_proposal as fit_source
    import sbsi.galsbi_particle_proposal as density_source
    metadata = fitted.metadata
    metadata['fit_implementation_sha256'] = {
        Path(path).name: digest(path) for path in (fit_source.__file__, density_source.__file__, __file__)}
    fitted = FrozenParticleProposal(fitted.proposals, metadata)
    fitted.save(args.output)
    print(json.dumps(dict(output=str(args.output.resolve()), sha256=digest(args.output),
        observations=metadata['observations'], pilot_seeds=metadata['pilot_seeds'],
        weights=[p.weights.tolist() for p in fitted.proposals],
        phase_concentrations=[p.phase_concentrations.tolist() for p in fitted.proposals]), indent=2))


if __name__ == '__main__':
    main()
