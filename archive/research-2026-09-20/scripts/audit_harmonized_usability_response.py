"""Measure random-direction forward response with paired versus per-leg usability.

This label-only diagnostic uses the corrected validation domain, not ConstGold.
All estimands keep the random-direction forward extraction on both sides.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np

from sbsi.flow_guard_response import numpy_guard_weights
from scripts.train_fixed_g0_flow import load_case, sha256, write_json

CUTS = ({'name': 'uncut', 'radius_min': None, 'magnitude_max': None},
        {'name': 'mag25.8_radius_gt0.60', 'radius_min': 3., 'magnitude_max': 25.8})


def weights(values, hard):
    if not hard:
        return numpy_guard_weights(values, CUTS, radius_softness=.05, magnitude_softness=.05)
    result = np.ones((len(values), len(CUTS)))
    result[:, 1] = ((values[:, 2] > 3.) & (30.-2.5*np.log10(values[:, 3]) < 25.8))
    return result


def sufficient(zero, shear, gamma, usable_zero, usable_shear, *, hard, paired):
    """Additive raw moments; absent measurements contribute zero mass, not shapes."""
    zero, shear, gamma = map(lambda x: np.asarray(x, dtype=float), (zero, shear, gamma))
    u0, ug = map(lambda x: np.asarray(x, dtype=bool), (usable_zero, usable_shear))
    n = len(gamma)
    if zero.shape != (n, 4) or shear.shape != zero.shape or gamma.shape != (n, 2) or u0.shape != (n,) or ug.shape != (n,):
        raise ValueError('aligned full-parent arrays required')
    if not np.isfinite(gamma).all():
        raise ValueError('finite full-parent shear required')
    for values, usable in ((zero, u0), (shear, ug)):
        if not np.isfinite(values[usable]).all() or (values[usable, 2:] <= 0).any():
            raise ValueError('usable measurements must be finite with positive photometry')
    parent = u0 & ug if paired else np.ones(n, dtype=bool)
    if not parent.any():
        raise ValueError('empty response population')
    legs = []
    for values, usable in ((zero, u0), (shear, ug)):
        take = usable & parent
        w = weights(values[take], hard)
        legs.append({'mass': w.sum(0), 'shape': np.einsum('nc,no->co', w, values[take, :2]),
                     'mass_gamma': np.einsum('nc,ni->ci', w, gamma[take]),
                     'shape_gamma': np.einsum('nc,no,ni->coi', w, values[take, :2], gamma[take])})
    return {'n': np.asarray(parent.sum()), 'normal': gamma[parent].T@gamma[parent],
            'mass_zero': legs[0]['mass'], 'shape_zero': legs[0]['shape'],
            'delta_mass_gamma': legs[1]['mass_gamma']-legs[0]['mass_gamma'],
            'delta_shape_gamma': legs[1]['shape_gamma']-legs[0]['shape_gamma']}


def response(stats):
    n = stats['n']
    mass = stats['mass_zero']
    if np.any(mass <= 0) or np.any(np.linalg.cond(stats['normal']) > 1e8):
        raise ValueError('positive cut mass and stable shear design required')
    mean = stats['shape_zero']/mass[..., None]
    cross = stats['delta_shape_gamma']-mean[..., None]*stats['delta_mass_gamma'][..., None, :]
    cross = cross/(mass/n[..., None])[..., None, None]
    return np.einsum('...coi,...ij->...coj', cross, np.linalg.inv(stats['normal']))


def aggregate(rows, case_weights):
    return {key: np.einsum('bc,c...->b...', case_weights, np.asarray([row[key] for row in rows])) for key in rows[0]}


def describe(value):
    return {'estimate': value[0], 'case_bootstrap_se': value[1:].std(0, ddof=1),
            'case_bootstrap_ci95': np.moveaxis(np.quantile(value[1:], [.025, .975], axis=0), 0, -1)}


def load_aligned(root, manifest, case):
    legs, provenance = {}, {}
    for leg in ('g0', 'g05'):
        path = root/f'flow/{leg}/case{case:03d}.npz'
        if sha256(path) != manifest['flow'][f'{leg}_cases'][str(case)]['sha256']:
            raise ValueError('changed prepared flow labels')
        valid = load_case(path)
        parent_path = root/f'usability/{leg}/case{case:03d}.npz'
        with np.load(parent_path) as stored:
            full = {k: stored[k] for k in ('input_index', 'gamma', 'usable')}
        ids = full['input_index']
        if len(np.unique(ids)) != len(ids):
            raise ValueError('duplicate truth identities')
        np.testing.assert_array_equal(ids[full['usable']], valid['input_index'])
        np.testing.assert_array_equal(full['gamma'][full['usable']], valid['gamma'])
        values = np.full((len(ids), 4), np.nan)
        values[full['usable']] = valid['target']
        legs[leg] = full | {'values': values}
        provenance[leg] = {'flow_sha256': sha256(path), 'usability_sha256': sha256(parent_path)}
    np.testing.assert_array_equal(legs['g0']['input_index'], legs['g05']['input_index'])
    np.testing.assert_array_equal(legs['g0']['gamma'], 0.)
    np.testing.assert_allclose(np.linalg.norm(legs['g05']['gamma'], axis=1), .05, atol=1e-12)
    return legs, provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--domain-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('full label audit requires Slurm')
    if args.output.exists():
        raise FileExistsError(args.output)
    manifest_path = args.domain_root/'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    cases = manifest['split']['validation_cases']
    if len(cases) < 2 or len(set(cases)) != len(cases):
        raise ValueError('distinct validation cases required')
    records, raw = [], {f'{width}/{population}': [] for width in ('soft', 'hard') for population in ('paired', 'per_leg')}
    for case in cases:
        legs, provenance = load_aligned(args.domain_root, manifest, case)
        z, g = legs['g0'], legs['g05']
        u0, ug = z['usable'], g['usable']
        records.append({'case': case, 'parent': len(u0), 'both': int((u0 & ug).sum()),
                        'zero_only': int((u0 & ~ug).sum()), 'shear_only': int((~u0 & ug).sum()),
                        'neither': int((~u0 & ~ug).sum()), 'provenance': provenance})
        for key, rows in raw.items():
            width, population = key.split('/')
            rows.append(sufficient(z['values'], g['values'], g['gamma'], u0, ug,
                                   hard=width == 'hard', paired=population == 'paired'))
        print(f'USABILITY_RESPONSE_CASE {records[-1]}', flush=True)
    rng = np.random.default_rng(20260914)
    case_weights = np.concatenate([np.ones((1, len(cases))), rng.multinomial(len(cases), np.ones(len(cases))/len(cases), size=10000)])
    estimates = {key: response(aggregate(rows, case_weights)) for key, rows in raw.items()}
    differences = {f'{width}/per_leg_minus_paired': describe(estimates[f'{width}/per_leg']-estimates[f'{width}/paired']) for width in ('soft', 'hard')}
    differences['per_leg/hard_minus_soft'] = describe(estimates['hard/per_leg']-estimates['soft/per_leg'])
    for key, value in estimates.items():
        print(f'USABILITY_RESPONSE_RESULT {key} primary={value[0,1].tolist()}', flush=True)
    write_json(args.output, {'cases': cases, 'cuts': CUTS, 'records': records,
        'responses': {key: describe(value) for key, value in estimates.items()}, 'differences': differences,
        'per_case_sufficient': raw, 'domain_manifest_sha256': sha256(manifest_path),
        'script_sha256': sha256(__file__), 'bootstrap_seed': 20260914,
        'extraction': 'forward random-direction h0.05 influence response; matched versus full-parent per-leg usable mass',
        'limitations': ['validation labels reused; not joint flow-plus-emulator acceptance',
                        'changes in usability and selection are included in per-leg forward response',
                        'forward finite-step influence response is not antithetic coherent-shear response',
                        'no missing measurements are imputed and no empirical offset is applied']})


if __name__ == '__main__':
    main()
