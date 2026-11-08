"""Audit frozen single flows against identically selected per-leg response labels."""
import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

from sbsi.fixed_g0_domain import FLOW_FEATURES
from sbsi.flow_guard_response import physical_context_draws, torch_guard_weights
from sbsi.measurement_model import load_measurement_model
from scripts import audit_harmonized_usability_response as labels
from scripts.train_fixed_g0_flow import sha256, write_json


MODES = tuple(f'{width}/{population}' for width in ('soft', 'hard') for population in ('paired', 'per_leg'))


def row_moments(values, hard):
    """Average over latent draws before applying the observed per-leg usability."""
    if values.ndim != 3 or values.shape[-1] != 4 or not bool(torch.isfinite(values).all()):
        raise ValueError('finite row/draw/four-output samples required')
    if bool((values[..., 2:] < 0).any()):
        raise ValueError('negative generated photometry')
    if hard:
        # A softplus underflow to zero remains in the uncut population.
        magnitude = 30.-2.5*torch.log10(values[..., 3])
        selected = (values[..., 2] > 3.) & (magnitude < 25.8)
        weight = torch.stack((torch.ones_like(magnitude), selected.to(values.dtype)), -1)
    else:
        weight = torch_guard_weights(values, labels.CUTS, radius_softness=.05, magnitude_softness=.05)
    mass = weight.mean(1)
    shape = (weight[..., None]*values[..., None, :2]).mean(1)
    return mass.cpu().numpy(), shape.cpu().numpy()


def moment_sufficient(zero, shear, gamma, u0, ug, paired):
    """The same design, selection and normalization contract as the label audit."""
    n = len(gamma)
    if gamma.shape != (n, 2) or u0.shape != (n,) or ug.shape != (n,):
        raise ValueError('aligned population and shear required')
    parent = u0 & ug if paired else np.ones(n, dtype=bool)
    stats = []
    for (mass, shape), usable in ((zero, u0), (shear, ug)):
        if mass.shape != (n, len(labels.CUTS)) or shape.shape != (*mass.shape, 2):
            raise ValueError('aligned output moments required')
        if not np.isfinite(mass).all() or not np.isfinite(shape).all():
            raise ValueError('finite output moments required')
        take = usable & parent
        stats.append({'mass': mass[take].sum(0), 'shape': shape[take].sum(0),
                      'mass_gamma': np.einsum('nc,ni->ci', mass[take], gamma[take]),
                      'shape_gamma': np.einsum('nco,ni->coi', shape[take], gamma[take])})
    return {'n': np.asarray(parent.sum()), 'normal': gamma[parent].T@gamma[parent],
            'mass_zero': stats[0]['mass'], 'shape_zero': stats[0]['shape'],
            'delta_mass_gamma': stats[1]['mass_gamma']-stats[0]['mass_gamma'],
            'delta_shape_gamma': stats[1]['shape_gamma']-stats[0]['shape_gamma']}


def response_conventions(predicted, observed):
    own = labels.response(predicted)
    measured_reference = dict(predicted, mass_zero=observed['mass_zero'], shape_zero=observed['shape_zero'])
    return {'own_normalization': own, 'measured_normalization': labels.response(measured_reference)}


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--domain-root', type=Path, required=True)
    parser.add_argument('--flow', action='append', required=True, help='name=checkpoint')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('flow integration requires Slurm')
    if args.device == 'cpu' and not args.smoke:
        raise ValueError('CPU is reserved for bounded operational smoke')
    if args.output.exists():
        raise FileExistsError(args.output)
    paths = dict(item.split('=', 1) for item in args.flow)
    if len(paths) != len(args.flow):
        raise ValueError('distinct model names required')
    manifest_path = args.domain_root/'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    cases = manifest['split']['validation_cases'][:2] if args.smoke else manifest['split']['validation_cases']
    if len(cases) < 2 or len(set(cases)) != len(cases):
        raise ValueError('distinct validation cases required')
    draws, groups, batch = (4, 2, 256) if args.smoke else (8, 2, 2048)
    seed = 2026091804000000
    hashes = {arm: sha256(path) for arm, path in paths.items()}
    bundles = {arm: load_measurement_model(path, device=torch.device(args.device)) for arm, path in paths.items()}
    for bundle in bundles.values():
        bundle.model.eval()
    observed = {mode: [] for mode in MODES}
    predicted = {arm: [{mode: [] for mode in MODES} for _ in range(groups)] for arm in paths}
    provenance = []
    for case in cases:
        started = time.monotonic()
        legs, receipt = labels.load_aligned(args.domain_root, manifest, case)
        size = min(512, len(legs['g0']['usable'])) if args.smoke else len(legs['g0']['usable'])
        z, g = ({key: value[:size] for key, value in legs[leg].items()} for leg in ('g0', 'g05'))
        gamma, u0, ug = g['gamma'], z['usable'], g['usable']
        contexts = {}
        for leg in ('g0', 'g05'):
            path = args.domain_root/f'usability/{leg}/case{case:03d}.npz'
            if sha256(path) != receipt[leg]['usability_sha256']:
                raise ValueError('usability context changed during read')
            with np.load(path) as data:
                np.testing.assert_array_equal(data['input_index'], legs[leg]['input_index'])
                contexts[leg] = pd.DataFrame(data['context'][:size], columns=FLOW_FEATURES)
        for mode in MODES:
            width, population = mode.split('/')
            observed[mode].append(labels.sufficient(z['values'], g['values'], gamma, u0, ug,
                                                   hard=width == 'hard', paired=population == 'paired'))
        for arm, bundle in bundles.items():
            tensors = [torch.as_tensor(bundle.condition_preprocessor.transform_frame(contexts[leg]), dtype=torch.float32, device=args.device) for leg in ('g0', 'g05')]
            for group in range(groups):
                total = {mode: None for mode in MODES}
                for block, start in enumerate(range(0, size, batch)):
                    stop = min(size, start+batch)
                    generated = physical_context_draws(bundle.model, [t[start:stop] for t in tensors], draws,
                                                        seed+case*10000019+group*100000000003+block)
                    for width in ('soft', 'hard'):
                        moments = [row_moments(values, width == 'hard') for values in generated]
                        for population in ('paired', 'per_leg'):
                            mode = f'{width}/{population}'
                            raw = moment_sufficient(*moments, gamma[start:stop], u0[start:stop], ug[start:stop], population == 'paired')
                            if total[mode] is None:
                                total[mode] = raw
                            else:
                                for key in raw:
                                    total[mode][key] += raw[key]
                    if block == 0 or (block+1) % 25 == 0:
                        print(f'PER_LEG_FLOW arm={arm} case={case} group={group} rows={stop}/{size}', flush=True)
                for mode in MODES:
                    predicted[arm][group][mode].append(total[mode])
        provenance.append({'case': case, 'rows': size, 'sources': receipt})
        print(f'PER_LEG_FLOW_CASE_DONE case={case} seconds={time.monotonic()-started:.1f}', flush=True)
    rng = np.random.default_rng(20260914)
    cw = np.concatenate([np.ones((1, len(cases))), rng.multinomial(len(cases), np.ones(len(cases))/len(cases), size=10000)])
    summary = {}
    for mode in MODES:
        obs = labels.aggregate(observed[mode], cw)
        target = labels.response(obs)
        summary[mode] = {'measured': labels.describe(target), 'models': {}}
        for arm in paths:
            integrated = [labels.aggregate(raw[mode], cw) for raw in predicted[arm]]
            mean = {key: np.mean([raw[key] for raw in integrated], axis=0) for key in obs}
            conventions = response_conventions(mean, obs)
            result = {}
            for convention, value in conventions.items():
                group_response = np.asarray([response_conventions(raw, obs)[convention][0] for raw in integrated])
                result[convention] = {'response': labels.describe(value), 'model_minus_measured': labels.describe(value-target),
                                      'integration_group_responses': group_response,
                                      'integration_group_se': group_response.std(0, ddof=1)/np.sqrt(groups)}
            result['baseline_selected_mass_model_minus_measured'] = labels.describe((mean['mass_zero']-obs['mass_zero'])/obs['n'][:, None])
            summary[mode]['models'][arm] = result
            print(f'PER_LEG_FLOW_RESULT mode={mode} arm={arm} primary_model={conventions["own_normalization"][0,1].tolist()} measured={target[0,1].tolist()}', flush=True)
    if hashes != {arm: sha256(path) for arm, path in paths.items()}:
        raise ValueError('model artifact changed during audit')
    write_json(args.output, {'smoke': args.smoke, 'cases': cases, 'draws': draws, 'groups': groups, 'batch_size': batch,
        'latent_seed': seed, 'common_random_numbers': 'same full-parent rows and latents across both legs, cuts and frozen models',
        'flow_sha256': hashes, 'domain_manifest_sha256': sha256(manifest_path), 'script_sha256': sha256(__file__),
        'label_helper_sha256': sha256(labels.__file__), 'provenance': provenance, 'summary': summary,
        'per_case_observed': observed, 'per_case_model_groups': predicted,
        'limitations': ['operational 512-row/two-case subset only' if args.smoke else 'full reused validation population',
                        'no emulator; random-direction forward0.05 response, not coherent antithetic joint response',
                        'actual per-leg U is fixed, so classifier error is not included',
                        'case bootstrap conditions on latent integration; two groups give only a rough integration error estimate',
                        'measured normalization is a diagnostic influence convention, never a deployed correction',
                        'no calibration claim or model selection']})


if __name__ == '__main__':
    main()
