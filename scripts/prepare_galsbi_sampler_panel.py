"""Freeze a measured-property stratified held-out panel before sampler results."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from check_galsbi_continuous_prior import digest, save


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('fresh held-out panel required')
    protocol = json.loads((args.reference/'protocol.json').read_text())
    receipt = json.loads((args.reference/'result.json').read_text())
    if (not receipt['preparation_checks_pass']
            or receipt['observations_sha256'] != digest(args.reference/'observations.parquet')):
        raise ValueError('validated unchanged reference required')
    observed = pd.read_parquet(args.reference/'observations.parquet')
    magnitude = protocol['conditions']['zero_point']-2.5*np.log10(observed.iloc[:, 3].to_numpy())
    radius = observed.iloc[:, 2].to_numpy()
    usable = (radius > 3)&(radius < 100)&(np.square(observed.iloc[:, :2]).sum(1).to_numpy() < .99**2)
    development = [58165, 0, 1, 8]
    usable[development] = False
    rng = np.random.default_rng(2026102204)
    bands = []
    for lower, upper in [(None, 21), (21, 23), (23, 24), (24, 25), (25, 25.8)]:
        mask = usable & (magnitude < upper)
        if lower is not None:
            mask &= magnitude >= lower
        candidates = np.flatnonzero(mask)
        chosen = rng.choice(candidates, 4, replace=False)
        bands.append(dict(lower=lower, upper=upper, eligible=len(candidates), observations=chosen.tolist(),
                          measured_magnitude=magnitude[chosen].tolist()))
    save(args.output, dict(seed=2026102204, bands=bands, excluded_development=development,
        reference=str(args.reference.resolve()), reference_protocol_sha256=digest(args.reference/'protocol.json'),
        observations_sha256=receipt['observations_sha256'], script_sha256=digest(__file__),
        selection='Four measured-property-selected rows per band, without inference or generating-truth fields.',
        use='Held out from development; tune no sampler setting on these results.'))
    print('HELDOUT_PANEL_READY', args.output, 'objects', sum(len(b['observations']) for b in bands), flush=True)


if __name__ == '__main__':
    main()
