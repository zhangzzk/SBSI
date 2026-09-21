"""Experimental population-consistent guard for per-leg measurement densities.

All truth-parent contexts are retained. Each measurement and model expectation
uses its own leg's observed usability. This adapter reuses the existing CRN
integrator and independent-row gradient replay without changing legacy models.
"""
import numpy as np
import torch

from sbsi.flow_guard_response import numpy_guard_weights, physical_context_draws, torch_guard_weights


def fit_per_leg_guard_response(zero, shear, gamma, usable_zero, usable_shear, cuts, *,
                               radius_softness, magnitude_softness, zero_point=30., chunk_size=100000):
    zero, shear, gamma = (np.asarray(x, dtype=np.float64) for x in (zero, shear, gamma))
    u0, ug = (np.asarray(x) for x in (usable_zero, usable_shear))
    n = len(gamma)
    cuts = tuple(dict(cut) for cut in cuts)
    if (not n or gamma.shape != (n, 2) or zero.shape != (n, 4) or shear.shape != zero.shape
            or u0.shape != (n,) or ug.shape != (n,) or u0.dtype != bool or ug.dtype != bool
            or not np.isfinite(gamma).all() or chunk_size < 1 or not cuts):
        raise ValueError('aligned full-parent data, boolean per-leg usability and finite shear required')
    for values, usable in ((zero, u0), (shear, ug)):
        if not np.isfinite(values[usable]).all() or (values[usable, 2:] <= 0).any():
            raise ValueError('usable targets must be finite with positive photometry')
    normal = gamma.T@gamma
    if not np.isfinite(normal).all() or np.linalg.cond(normal) > 1e8:
        raise ValueError('stable two-component full-parent shear design required')
    moments = []
    for values, usable in ((zero, u0), (shear, ug)):
        mass = np.zeros(len(cuts))
        shape = np.zeros((len(cuts), 2))
        mass_gamma = np.zeros((len(cuts), 2))
        shape_gamma = np.zeros((len(cuts), 2, 2))
        for start in range(0, n, chunk_size):
            take = usable[start:start+chunk_size]
            block = values[start:start+chunk_size][take]
            design = gamma[start:start+chunk_size][take]
            w = numpy_guard_weights(block, cuts, radius_softness=radius_softness,
                                    magnitude_softness=magnitude_softness, zero_point=zero_point)
            mass += w.sum(0)
            shape += np.einsum('nc,no->co', w, block[:, :2])
            mass_gamma += np.einsum('nc,ni->ci', w, design)
            shape_gamma += np.einsum('nc,no,ni->coi', w, block[:, :2], design)
        moments.append({'mass': mass, 'shape': shape, 'mass_gamma': mass_gamma, 'shape_gamma': shape_gamma})
    baseline_mass = moments[0]['mass']/n
    if not np.isfinite(baseline_mass).all() or (baseline_mass <= 0).any():
        raise ValueError('positive baseline selected mass required for every guard')
    baseline_mean = moments[0]['shape']/moments[0]['mass'][:, None]
    cross = moments[1]['shape_gamma']-moments[0]['shape_gamma']
    cross -= baseline_mean[..., None]*(moments[1]['mass_gamma']-moments[0]['mass_gamma'])[:, None, :]
    response = (cross/baseline_mass[:, None, None])@np.linalg.inv(normal)
    return {'target': response.reshape(len(cuts), 4), 'baseline_mass': baseline_mass,
            'baseline_mean_shape': baseline_mean, 'shape_normal': normal, 'truth_parent_rows': n,
            'usable_counts': {'zero': int(u0.sum()), 'shear': int(ug.sum()), 'both': int((u0 & ug).sum()),
                              'zero_only': int((u0 & ~ug).sum()), 'shear_only': int((~u0 & ug).sum())},
            'cuts': cuts, 'radius_softness': float(radius_softness), 'magnitude_softness': float(magnitude_softness),
            'zero_point': float(zero_point), 'population': 'full truth parent with own-leg usability; no intersection conditioning'}


class PerLegGuardResponsePopulation:
    """Adapter for existing independent-row response integration/gradient replay.

``pairs`` associates the two contexts of every truth parent, including parents
unusable in either or both legs. It does not select a both-usable population.
Use ``independent_pairs=True`` with sampled_guard_response_backward.
"""
    def __init__(self, context_zero, context_shear, measured_zero, measured_shear, gamma,
                 usable_zero, usable_shear, cuts, *, radius_softness, magnitude_softness,
                 response_scale, zero_point=30.):
        if (context_zero.ndim != 2 or context_shear.shape != context_zero.shape
                or context_zero.device != context_shear.device
                or not bool(torch.isfinite(context_zero).all() and torch.isfinite(context_shear).all())):
            raise ValueError('aligned finite full-parent contexts on the same device required')
        stats = fit_per_leg_guard_response(measured_zero, measured_shear, gamma, usable_zero, usable_shear, cuts,
                                           radius_softness=radius_softness, magnitude_softness=magnitude_softness,
                                           zero_point=zero_point)
        n, device = stats['truth_parent_rows'], context_zero.device
        if len(context_zero) != n:
            raise ValueError('contexts must cover every truth parent')
        self.context = torch.cat((context_zero, context_shear), dim=0)
        self.pairs = torch.column_stack((torch.arange(n, device=device), torch.arange(n, device=device)+n))
        self.usable = torch.as_tensor(np.column_stack((usable_zero, usable_shear)), dtype=torch.bool, device=device)
        self.cuts = stats['cuts']
        self.radius_softness, self.magnitude_softness = float(radius_softness), float(magnitude_softness)
        self.zero_point = float(zero_point)
        self.target = torch.as_tensor(stats['target'], dtype=torch.float64, device=device)
        self.baseline_mass = torch.as_tensor(stats['baseline_mass'], dtype=torch.float64, device=device)
        self.baseline_mean = torch.as_tensor(stats['baseline_mean_shape'], dtype=torch.float64, device=device)
        self.design = torch.as_tensor(np.asarray(gamma)@np.linalg.inv(stats['shape_normal'])*n, dtype=torch.float64, device=device)
        scale = np.asarray(response_scale, dtype=np.float64)
        if scale.ndim == 0:
            scale = np.full(len(self.cuts), float(scale))
        if scale.shape != (len(self.cuts),) or not np.isfinite(scale).all() or (scale <= 0).any():
            raise ValueError('finite positive per-cut response scales required')
        self.response_scale = torch.as_tensor(scale[:, None], dtype=torch.float64, device=device)
        self.statistics = stats

    def contributions(self, model, indices, draws, seed):
        pair = self.pairs[indices]
        generated = physical_context_draws(model, [self.context[pair[:, leg]] for leg in (0, 1)], draws, seed)
        influences = []
        for leg, values in enumerate(generated):
            w = torch_guard_weights(values, self.cuts, radius_softness=self.radius_softness,
                                    magnitude_softness=self.magnitude_softness, zero_point=self.zero_point)
            contribution = (w[..., None]*(values[..., None, :2]-self.baseline_mean[None, None])
                            /self.baseline_mass[None, None, :, None]).mean(1)
            influences.append(contribution*self.usable[indices, leg, None, None])
        delta = influences[1]-influences[0]
        result = torch.einsum('nco,ni->ncoi', delta, self.design[indices]).reshape(len(indices), len(self.cuts), 4)
        if not bool(torch.isfinite(result).all()):
            raise ValueError('nonfinite per-leg generated response contribution')
        return result


def main():
    """Prepare a reference target bank only; never fit or select a model."""
    import argparse
    import json
    import os
    from pathlib import Path

    from sbsi.flow_guard_response import fit_guard_response, make_guard_cuts
    from scripts.audit_harmonized_usability_response import load_aligned
    from scripts.train_fixed_g0_flow import sha256, write_json
    from scripts.train_full_domain_guard_flow import MAGNITUDE_THRESHOLDS, RADIUS_THRESHOLDS

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--domain-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--split', choices=('train', 'validation'), default='validation')
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('full target construction requires Slurm')
    if args.output.exists():
        raise FileExistsError(args.output)
    manifest_path = args.domain_root/'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    cases = manifest['split'][f'{args.split}_cases']
    parts = {key: [] for key in ('zero', 'shear', 'gamma', 'u0', 'ug')}
    provenance = []
    for case in cases:
        legs, receipt = load_aligned(args.domain_root, manifest, case)
        parts['zero'].append(legs['g0']['values'])
        parts['shear'].append(legs['g05']['values'])
        parts['gamma'].append(legs['g05']['gamma'])
        parts['u0'].append(legs['g0']['usable'])
        parts['ug'].append(legs['g05']['usable'])
        provenance.append({'case': case, 'sources': receipt})
        print(f'PER_LEG_GUARD_LABELS case={case} parents={len(legs["g0"]["usable"])}', flush=True)
    zero, shear, gamma, u0, ug = (np.concatenate(parts.pop(key)) for key in ('zero', 'shear', 'gamma', 'u0', 'ug'))
    cuts = make_guard_cuts(RADIUS_THRESHOLDS, MAGNITUDE_THRESHOLDS)
    options = {'radius_softness': .05, 'magnitude_softness': .05}
    per_leg = fit_per_leg_guard_response(zero, shear, gamma, u0, ug, cuts, **options)
    both = u0 & ug
    paired = fit_guard_response(zero[both], shear[both], gamma[both], cuts, **options)
    write_json(args.output, {'split': args.split, 'cases': cases, 'per_leg': per_leg, 'paired': paired,
        'domain_manifest_sha256': sha256(manifest_path), 'provenance': provenance,
        'script_sha256': sha256(__file__), 'label_alignment_helper_sha256': sha256(load_aligned.__code__.co_filename),
        'legacy_guard_helper_sha256': sha256(fit_guard_response.__code__.co_filename),
        'limitations': ['training-label target bank, no model update or selection',
                        'soft forward0.05 response, not coherent antithetic joint acceptance',
                        'per-leg target must only be combined with per-leg model contributions',
                        'no confidence interval here; primary-cut case uncertainty is in the separate label audit']})
    print('PER_LEG_GUARD_LABELS_COMPLETE', args.output, flush=True)


if __name__ == '__main__':
    main()
