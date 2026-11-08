"""Measure sign-symmetry violations in the actual finite FS2 sampling prior.

This audits the exact catalogue distribution sampled by BlendEMU. It does not
assume catalogue rows are independent observations from an isotropic universe,
and it does not randomize angles or alter any training/simulation data.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8388608), b''):
            digest.update(block)
    return digest.hexdigest()


MAG_EDGES = np.array([0., 18., 20., 22., 24., 26., 29.])
SIZE_EDGES = np.array([0., .1, .3, .5, 1., 2., 10.])
ANGLE_BINS = 36


def angular_summary(histogram, shape_sums, mass):
    if mass <= 0:
        return {'mass': float(mass), 'mean_shape': None, 'orbit_tv': None}
    probability = np.asarray(histogram, dtype=float)/mass
    indices = np.arange(ANGLE_BINS)
    orbit = np.stack([probability, probability[-indices % ANGLE_BINS],
                      probability[(ANGLE_BINS//2-indices) % ANGLE_BINS],
                      probability[(ANGLE_BINS//2+indices) % ANGLE_BINS]]).mean(0)
    return {'mass': float(mass), 'mean_shape': (np.asarray(shape_sums)/mass).tolist(),
            'angle_probability': probability.tolist(),
            'orbit_tv': float(np.abs(probability-orbit).sum()/2),
            'max_bin_probability_delta': float(np.abs(probability-orbit).max())}


def accumulate(q, angle_degrees, mag, circular_size, sums):
    arrays = [np.asarray(x, dtype=float) for x in (q, angle_degrees, mag, circular_size)]
    q, angle, mag, size = arrays
    if (len({x.shape for x in arrays}) != 1 or q.ndim != 1
            or not all(np.isfinite(x).all() for x in arrays)
            or ((q <= 0) | (q >= 1)).any()):
        raise ValueError('aligned finite physical prior arrays required')
    phi = np.deg2rad(2*angle)
    amp = (1-q)/(1+q)
    shape = np.column_stack((amp*np.cos(phi), amp*np.sin(phi)))
    width = 2*np.pi/ANGLE_BINS
    angle_index = np.floor(np.remainder(phi+width/2, 2*np.pi)/width).astype(int)
    mi = np.searchsorted(MAG_EDGES, mag, side='right')-1
    si = np.searchsorted(SIZE_EDGES, size, side='right')-1
    if ((mi < 0) | (mi >= len(MAG_EDGES)-1) | (si < 0) | (si >= len(SIZE_EDGES)-1)).any():
        raise ValueError('prior row outside declared magnitude/size strata')
    strata = mi*(len(SIZE_EDGES)-1)+si
    nstrata = (len(MAG_EDGES)-1)*(len(SIZE_EDGES)-1)
    for kind, weight in [('number', np.ones(len(q))), ('flux', 10**(-.4*(mag-30)))]:
        entry = sums.setdefault(kind, {'hist': np.zeros((nstrata, ANGLE_BINS)),
                                       'mass': np.zeros(nstrata), 'shape': np.zeros((nstrata, 2))})
        entry['hist'] += np.bincount(strata*ANGLE_BINS+angle_index, weights=weight,
                                     minlength=nstrata*ANGLE_BINS).reshape(nstrata, ANGLE_BINS)
        entry['mass'] += np.bincount(strata, weights=weight, minlength=nstrata)
        for axis in (0, 1):
            entry['shape'][:, axis] += np.bincount(strata, weights=weight*shape[:, axis], minlength=nstrata)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--catalogue', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('finite catalogue-prior scan requires Slurm')
    if args.output.exists():
        raise FileExistsError(args.output)
    from blendemu import catalog
    source_sha256 = sha256(args.catalogue)
    frame, density = catalog.load_fs2_catalog(str(args.catalogue), mag_col='lsst_r_mag', mag_cut=29.)
    sums = {}
    for start in range(0, len(frame), 250000):
        chunk = frame.iloc[start:start+250000]
        accumulate(chunk['BA'], chunk['angle'], chunk['sdss_r'],
                   chunk['Re_arcsec'].to_numpy()*np.sqrt(chunk['BA'].to_numpy()), sums)
    result = {}
    for kind, entry in sums.items():
        rows = []
        for mi in range(len(MAG_EDGES)-1):
            for si in range(len(SIZE_EDGES)-1):
                index = mi*(len(SIZE_EDGES)-1)+si
                rows.append({'magnitude_interval': MAG_EDGES[mi:mi+2].tolist(),
                             'circular_size_interval_arcsec': SIZE_EDGES[si:si+2].tolist(),
                             **angular_summary(entry['hist'][index], entry['shape'][index], entry['mass'][index])})
        result[kind] = {'full': angular_summary(entry['hist'].sum(0), entry['shape'].sum(0), entry['mass'].sum()),
                        'strata': rows}
    report = {'rows': len(frame), 'density_per_deg2': float(density), 'results': result,
               'runtime': {'python': sys.version, 'numpy': np.__version__, 'pandas': pd.__version__},
               'angle_bins': ANGLE_BINS, 'spin2_angle_bin_centres_radians': (np.arange(ANGLE_BINS)*2*np.pi/ANGLE_BINS).tolist(),
               'source_sha256': source_sha256, 'loader_sha256': sha256(catalog.__file__),
               'script_sha256': sha256(__file__),
               'interpretation': 'exact finite sampling-prior asymmetry, no iid-row significance or isotropy acceptance test',
               'limitations': ['coarse angle/magnitude/size strata do not establish full conditional scene symmetry',
                               'flux weighting is descriptive and can be dominated by rare bright objects',
                               'simulation sampling positions, detection and measurement are separate checks',
                               'no angle randomization or calibration correction is applied']}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({kind: value['full'] for kind, value in result.items()}, indent=2), flush=True)
    print(f'SCENE_PRIOR_SYMMETRY_COMPLETE rows={len(frame)} output={args.output}', flush=True)


if __name__ == '__main__':
    main()
