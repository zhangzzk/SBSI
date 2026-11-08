"""Diagnose discrete shape-symmetry violations on frozen held-out g0 data.

This is a diagnostic of a symmetry hypothesis, not a new inference model.
Square-pixel rotations/reflections induce four distinct sign actions on spin-2
shapes. The circular PSF is necessary but does not by itself prove the complete
simulation and measurement pipeline has these conditional symmetries.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sbsi.fixed_g0_domain import FLOW_FEATURES, FLOW_TARGETS
from sbsi.measurement_model import load_measurement_model
from scripts.train_fixed_g0_flow import sha256, write_json


ACTIONS = ((1, 1), (1, -1), (-1, 1), (-1, -1))


def sign_action(context, target, signs):
    if tuple(signs) not in ACTIONS:
        raise ValueError("unknown shape sign action")
    context, target = np.asarray(context), np.asarray(target)
    if (context.ndim != 2 or context.shape[1] != len(FLOW_FEATURES)
            or target.shape != (len(context), len(FLOW_TARGETS))):
        raise ValueError("aligned physical context and target arrays required")
    transformed_context, transformed_target = context.copy(), target.copy()
    for axis, sign in enumerate(signs):
        transformed_context[:, FLOW_FEATURES.index(f'e{axis+1}_input_p')] *= sign
        transformed_target[:, axis] *= sign
    return transformed_context, transformed_target


def orbit_statistics(log_probabilities):
    logp = np.asarray(log_probabilities, dtype=float)
    if logp.ndim != 2 or logp.shape[1] != len(ACTIONS) or not np.isfinite(logp).all():
        raise ValueError("four finite orbit log densities required")
    maximum = logp.max(1)
    log_average = maximum + np.log(np.exp(logp-maximum[:, None]).mean(1))
    return {"identity_nll": float(-logp[:, 0].mean()),
            "orbit_average_nll": float(-log_average.mean()),
            "orbit_minus_identity_nll": float((logp[:, 0]-log_average).mean()),
            "action_log_density_delta": (logp-logp[:, :1]).mean(0),
            "action_mean_absolute_log_density_delta": np.abs(logp-logp[:, :1]).mean(0),
            "action_rms_log_density_delta": np.sqrt(np.square(logp-logp[:, :1]).mean(0))}


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--domain', type=Path, required=True)
    parser.add_argument('--flow', type=Path, action='append', required=True)
    parser.add_argument('--noise-catalogue', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--rows-per-case', type=int, default=4096)
    parser.add_argument('--seed', type=int, default=20260918)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('density/model audit must run under Slurm')
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.rows_per_case < 1 or len(set(args.flow)) != len(args.flow):
        raise ValueError('positive sample count and distinct flow paths required')
    noise = pd.read_csv(args.noise_catalogue)
    tile = noise.loc[noise['label'].astype(str) == '180.0_-0.5']
    if len(tile) != 1 or not np.array_equal(tile[['seeing_e1_r', 'seeing_e2_r']].to_numpy(), [[0., 0.]]):
        raise ValueError('audit hypothesis requires the recorded circular PSF')
    manifest_path = args.domain/'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    cases = sorted(manifest['split']['validation_cases'])
    if len(cases) < 2 or set(cases) & set(manifest['split']['train_cases']):
        raise ValueError('distinct held-out scenes required')
    models = [load_measurement_model(path, device=torch.device('cpu')) for path in args.flow]
    for model in models:
        model.model.eval()
        if (model.condition_preprocessor.feature_names != list(FLOW_FEATURES)
                or model.target_transform.target_names != list(FLOW_TARGETS)):
            raise ValueError('incompatible physical flow contract')
    records = {str(path): [] for path in args.flow}
    sources = {}
    for case in cases:
        path = args.domain/f'flow/g0/case{case:03d}.npz'
        if sha256(path) != manifest['flow']['g0_cases'][str(case)]['sha256']:
            raise ValueError('changed held-out g0 data')
        with np.load(path, allow_pickle=False) as stored:
            count = len(stored['input_index'])
            if not np.array_equal(stored['gamma'], np.zeros((count, 2))):
                raise ValueError('symmetry audit must use zero external shear')
            selected = np.random.default_rng(args.seed+1000003*case).choice(
                count, min(count, args.rows_per_case), replace=False)
            context, target = stored['context'][selected].astype(float), stored['target'][selected].astype(float)
        sources[str(case)] = {'sha256': manifest['flow']['g0_cases'][str(case)]['sha256'],
                              'available_rows': count, 'sample_rows': len(selected)}
        for flow, model in zip(args.flow, models):
            means = torch.as_tensor(model.target_transform.means, dtype=torch.float64)
            scales = torch.as_tensor(model.target_transform.scales, dtype=torch.float64)
            columns = []
            for action in ACTIONS:
                transformed_context, transformed_target = sign_action(context, target, action)
                conditions = model.context_tensor(pd.DataFrame(transformed_context, columns=FLOW_FEATURES))
                values = (torch.as_tensor(transformed_target)-means)/scales
                logp = model.log_prob_tensor(values, conditions, batch_size=4096)-scales.log().sum()
                columns.append(logp.numpy())
            record = {'case': case, 'rows': len(selected), **orbit_statistics(np.column_stack(columns))}
            records[str(flow)].append(record)
            print(f'SHAPE_SYMMETRY case={case} flow={flow} delta_nll={record["orbit_minus_identity_nll"]:+.8f}', flush=True)
    picks = np.random.default_rng(args.seed).integers(len(cases), size=(10000, len(cases)))
    summaries = {}
    for path, rows in records.items():
        summaries[path] = {}
        for field in orbit_statistics(np.zeros((1, len(ACTIONS)))):
            values = np.asarray([row[field] for row in rows])
            # The fixed sample size gives each independent held-out scene equal weight.
            bootstrap = values[picks].mean(1)
            summaries[path][field] = {'mean': values.mean(0), 'case_bootstrap_se': bootstrap.std(0, ddof=1)}
    write_json(args.output, {'cases': cases, 'actions': ACTIONS, 'rows_per_case': args.rows_per_case,
        'seed': args.seed, 'records': records, 'summary': summaries, 'data_sources': sources,
        'flow_sha256': {str(path): sha256(path) for path in args.flow},
        'manifest_sha256': sha256(manifest_path), 'script_sha256': sha256(__file__),
        'noise_catalogue_sha256': sha256(args.noise_catalogue),
        'limitations': ['circular PSF and scalar context motivate, but do not establish, conditional sign symmetry',
                        'no rotated images or measurement-pipeline symmetry verification',
                        'orbit average is a diagnostic normalized mixture, not a proposed deployment model',
                        'held-out g0 density only; no joint response, emulator, or independent acceptance claim',
                        'fixed per-scene row sample and equal-scene weighting; not a full-catalogue NLL gate']})
    print(f'SHAPE_SYMMETRY_COMPLETE {args.output}', flush=True)


if __name__ == '__main__':
    main()
