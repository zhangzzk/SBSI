"""Separate selected-mass and centering effects in frozen response audits.

This is an identity for the recorded finite-step influence estimator, not a
correction to a model or an extrapolation to infinitesimal shear.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np

from scripts import audit_harmonized_usability_response as labels
from scripts.train_fixed_g0_flow import sha256, write_json


def decompose(predicted, observed):
    """Return terms in R_reference - R_own for identical parent/shear designs."""
    if any(not np.isfinite(value).all() for raw in (predicted, observed) for value in raw.values()):
        raise ValueError('finite sufficient statistics required')
    # Model moments sum GPU-sized blocks; labels sum an entire scene. Permit
    # reduction roundoff in the design only, while requiring exact row counts.
    normal_q, normal_d = predicted['normal'], observed['normal']
    design_error = np.linalg.norm(normal_q-normal_d, axis=(-2, -1))
    design_scale = np.linalg.norm(normal_d, axis=(-2, -1))
    if (not np.array_equal(predicted['n'], observed['n'])
            or np.any(design_error > 1e-12*design_scale)):
        raise ValueError('identical parent and shear design required')
    mass_q, mass_d = predicted['mass_zero'], observed['mass_zero']
    own = labels.response(predicted)
    reference = labels.response(dict(predicted, mass_zero=mass_d, shape_zero=observed['shape_zero']))
    mass_ratio_error = mass_q/mass_d-1
    mean_offset = predicted['shape_zero']/mass_q[..., None]-observed['shape_zero']/mass_d[..., None]
    selection_slope = np.einsum('...ci,...ij->...cj', predicted['delta_mass_gamma'],
                               np.linalg.inv(predicted['normal']))
    selection_slope *= (predicted['n'][..., None]/mass_d)[..., None]
    mass_term = mass_ratio_error[..., None, None]*own
    centering_term = mean_offset[..., :, None]*selection_slope[..., None, :]
    difference = reference-own
    error = np.max(np.abs(difference-mass_term-centering_term))
    np.testing.assert_allclose(difference, mass_term+centering_term, rtol=1e-10, atol=2e-12)
    return {'own_response': own, 'measured_reference_response': reference,
            'reference_minus_own': difference, 'selected_mass_term': mass_term,
            'centering_term': centering_term, 'selected_mass_relative_error': mass_ratio_error,
            'selected_mean_shape_offset': mean_offset,
            'selection_slope_over_measured_mass': selection_slope}, float(error)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('bootstrap decomposition requires Slurm')
    if args.output.exists():
        raise FileExistsError(args.output)
    audit_hash = sha256(args.audit)
    audit = json.loads(args.audit.read_text())
    cases = audit['cases']
    if len(cases) < 2 or len(cases) != len(set(cases)):
        raise ValueError('at least two distinct cases required')
    if audit['label_helper_sha256'] != sha256(labels.__file__):
        raise ValueError('changed label estimator')
    rng = np.random.default_rng(20260914)
    weights = np.concatenate([np.ones((1, len(cases))),
        rng.multinomial(len(cases), np.ones(len(cases))/len(cases), size=10000)])
    summary, max_error = {}, 0.
    for mode, records in audit['per_case_observed'].items():
        observed = labels.aggregate(records, weights)
        summary[mode] = {}
        for arm, groups in audit['per_case_model_groups'].items():
            integrated = [labels.aggregate(group[mode], weights) for group in groups]
            predicted = {key: np.mean([raw[key] for raw in integrated], axis=0) for key in observed}
            terms, error = decompose(predicted, observed)
            max_error = max(max_error, error)
            summary[mode][arm] = {key: labels.describe(value) for key, value in terms.items()}
            print(f'GUARD_NORMALIZATION mode={mode} arm={arm} max_identity_error={error:.3g}', flush=True)
    if audit_hash != sha256(args.audit):
        raise ValueError('audit changed during decomposition')
    write_json(args.output, {'smoke': audit['smoke'], 'cases': cases, 'summary': summary,
        'max_identity_error': max_error, 'audit_sha256': audit_hash,
        'flow_sha256': audit['flow_sha256'], 'domain_manifest_sha256': audit['domain_manifest_sha256'],
        'script_sha256': sha256(__file__), 'label_helper_sha256': sha256(labels.__file__),
        'bootstrap_seed': 20260914, 'bootstrap_replicates': 10000,
        'limitations': ['finite-step forward influence response, not coherent joint calibration',
                        'case uncertainty conditions on the recorded latent draws',
                        'smoke subsets are operational only' if audit['smoke'] else 'reused validation cases',
                        'algebraic attribution only; no model modification, fitting or empirical correction']})


if __name__ == '__main__':
    main()
