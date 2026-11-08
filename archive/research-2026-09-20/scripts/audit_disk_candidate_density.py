"""Evaluate the experimental transported density on observed ConstGold outputs.

By default compare transport against its own self-response flow. An explicit
reference flow instead compares two transported models on the same observations.
This is a density diagnostic, not a checkpoint-selection rule. Selection masses
need not agree when the flow changes, so restricted scores remain unnormalized.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from plots.constgold_property_bias import load_lookup
from sbsi.disk_response_transport import transported_physical_log_prob
from sbsi.measurement_model import load_measurement_model
from sbsi.output_conditioned_response import DiskResponseMoment, predict_scene_draw_responses
from scripts.evaluate_constgold_truth_parent import (
    CUTS, load_candidate_pairs, measured_leg, output_cuts, raw_root, shear_frames,
)
from scripts.train_fixed_g0_flow import sha256, write_json


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', type=int, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--flow', type=Path)
    parser.add_argument('--reference-flow', type=Path)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--shears', type=float, nargs='+', default=[0., .02, -.02],
                        help='Existing endpoint legs; omit zero to avoid historical g0 training measurements')
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('catalogue density scan requires Slurm')
    if args.output.exists():
        raise FileExistsError(args.output)
    cases = sorted(args.cases)
    shears = tuple(args.shears)
    if not shears or len(set(shears)) != len(shears) or not set(shears) <= {0., .02, -.02}:
        raise ValueError('distinct existing shear legs from 0,+/-0.02 required')
    if len(set(cases)) != len(cases) or len(cases) < 2 or not set(cases) <= set(range(40, 120)):
        raise ValueError('distinct full-parent cases40--119 required')
    data = Path('/project/ls-gruen/users/zekang.zhang')
    cache = data/'sbsi_caches'
    parent = cache/'constgold_truth_parent_20260917_v1'
    pair_root = cache/'output_conditioned_response_20260917_v1/pairs'
    raw = data/'lsst_sims_fs2_25876_constant'
    flow = args.flow or cache/'full_domain_true_mag26_guard_v1/flow/guard/selected.pt'
    model_root = data/'blendemu_runs/disk_response_moment_20260918_v1/model'
    manifest_path = parent/'parent_v2/manifest.json'
    manifest = json.loads(manifest_path.read_text())
    lookup = load_lookup(tuple(parent/f'rblend/cases{first}_{first+9}.feather'
                               for first in sorted({case//10*10 for case in cases})), tuple(cases))
    candidate = DiskResponseMoment(model_root, nthread=4, device=args.device)
    bundle = load_measurement_model(flow, device=torch.device(args.device))
    bundle.model.eval()
    means = torch.as_tensor(bundle.target_transform.means, dtype=torch.float64, device=args.device)
    scales = torch.as_tensor(bundle.target_transform.scales, dtype=torch.float64, device=args.device)
    reference_bundle = None
    if args.reference_flow:
        reference_bundle = load_measurement_model(args.reference_flow, device=torch.device(args.device))
        reference_bundle.model.eval()
        reference_means = torch.as_tensor(reference_bundle.target_transform.means, dtype=torch.float64, device=args.device)
        reference_scales = torch.as_tensor(reference_bundle.target_transform.scales, dtype=torch.float64, device=args.device)
    reports = []
    for case in cases:
        path = parent/f'parent_v2/case{case:03d}.npz'
        if sha256(path) != manifest['case_reports'][str(case)]['sha256']:
            raise ValueError('changed truth-parent artifact')
        with np.load(path) as stored:
            ids, context = stored['input_index'], stored['context'].astype(float)
        if not np.array_equal(ids, lookup[case]['input_index']):
            raise ValueError('response lookup and parent identities differ')
        pairs, _ = load_candidate_pairs(pair_root, case, path, ids, lookup[case]['R_blend'])
        for h in shears:
            if h == 0.:
                # ConstGold has no separate zero-shear image directory. Its
                # parent records the hash of the corresponding original g0
                # flow-domain catalogue; use those exact existing targets.
                zero_path = cache/f'full_domain_true_mag26_guard_v1/domain/flow/g0/case{case:03d}.npz'
                if sha256(zero_path) != manifest['case_reports'][str(case)]['g0_sha256']:
                    raise ValueError('zero-shear measurement differs from parent provenance')
                with np.load(zero_path) as stored:
                    zero_ids, zero_values = stored['input_index'], stored['target']
                if len(np.unique(zero_ids)) != len(zero_ids):
                    raise ValueError('duplicate zero-shear measurement identities')
                shared, pi, zi = np.intersect1d(ids, zero_ids, return_indices=True)
                observed = np.full((len(ids), 4), np.nan)
                observed[pi] = zero_values[zi]
                usable = np.zeros(len(ids), dtype=bool)
                usable[pi] = True
                if len(shared) != manifest['case_reports'][str(case)]['g0_usable']:
                    raise ValueError('zero-shear membership differs from frozen parent')
                receipt = {'parent_rows': len(ids), 'parent_usable': len(shared),
                           'zero_rows_outside_parent_dropped': len(zero_ids)-len(shared)}
                sources = [zero_path]
            else:
                observed, usable, receipt = measured_leg(raw, case, h, ids)
                catalogue = raw_root(raw, case, h)
                sources = [catalogue/'CrossMatch/tile180.0_-0.5_rot0_matched.feather',
                           catalogue/'Shapes/shape_catalogue_detect_position_all_tile180.0_-0.5.feather']
            if not usable.any():
                raise ValueError('empty observed usable population')
            values = torch.as_tensor(observed[usable], dtype=torch.float64, device=args.device)
            frame = shear_frames(context, h)[0].loc[usable]
            conditions = bundle.context_tensor(frame)
            kept_pairs = pairs.loc[pairs['index_input_p'].isin(ids[usable])]
            response, counts = predict_scene_draw_responses(
                candidate, kept_pairs, ids[usable], observed[usable, None, :])
            velocity = torch.as_tensor(response[:, 0], device=args.device)[:, None]*values.new_tensor([h, 0.])

            def base_log_prob(original):
                return bundle.log_prob_tensor((original-means)/scales, conditions, batch_size=8192)-scales.log().sum()

            original_lp = base_log_prob(values)
            transported_lp = transported_physical_log_prob(values, velocity, base_log_prob)
            if not bool(torch.isfinite(original_lp).all() and torch.isfinite(transported_lp).all()):
                raise ValueError('nonfinite observed physical density; no rows may be silently dropped')
            if h == 0.:
                torch.testing.assert_close(transported_lp, original_lp, rtol=0, atol=0)
            comparison_lp = original_lp
            if reference_bundle is not None:
                reference_conditions = reference_bundle.context_tensor(frame)

                def reference_log_prob(original):
                    return reference_bundle.log_prob_tensor(
                        (original-reference_means)/reference_scales, reference_conditions,
                        batch_size=8192)-reference_scales.log().sum()

                comparison_lp = transported_physical_log_prob(values, velocity, reference_log_prob)
                if not bool(torch.isfinite(comparison_lp).all()):
                    raise ValueError('nonfinite reference density; no rows may be silently dropped')
            original_nll, candidate_nll = (-original_lp.cpu().numpy(), -transported_lp.cpu().numpy())
            comparison_nll = -comparison_lp.cpu().numpy()
            cuts = output_cuts(observed[usable])
            entries = {}
            for index, cut in enumerate(CUTS):
                keep = cuts[:, index]
                difference = candidate_nll[keep]-comparison_nll[keep]
                if not len(difference):
                    raise ValueError('empty declared density subset')
                entries[cut] = {'rows': int(keep.sum()), 'self_flow_nll': float(original_nll[keep].mean()),
                                'transported_nll': float(candidate_nll[keep].mean()),
                                'comparison_nll': float(comparison_nll[keep].mean()),
                                'paired_delta_nll': float(difference.mean()),
                                'paired_object_se': float(difference.std(ddof=1)/np.sqrt(len(difference)))}
            reports.append({'case': case, 'shear': h, 'cuts': entries, 'measurement': receipt,
                            'prediction_counts': counts, 'pairs_for_usable': len(kept_pairs),
                            'pairs_dropped_with_unusable_primary': len(pairs)-len(kept_pairs),
                            'measurement_sources': {str(p): sha256(p) for p in sources}})
            print(f'DISK_DENSITY case={case} shear={h} cuts={entries}', flush=True)
    picks = np.random.default_rng(20260918).integers(len(cases), size=(2000, len(cases)))
    summary = {}
    for h in shears:
        for cut in CUTS:
            entries = [r['cuts'][cut] for r in reports if r['shear'] == h]
            count = np.array([r['rows'] for r in entries])
            fields = {}
            for field in ('self_flow_nll', 'transported_nll', 'comparison_nll', 'paired_delta_nll'):
                values = np.array([r[field] for r in entries])
                bootstrap = (values*count)[picks].sum(1)/count[picks].sum(1)
                fields[field] = {'mean': float(np.average(values, weights=count)),
                                 'case_bootstrap_se': float(bootstrap.std(ddof=1))}
            summary[f'{h}/{cut}'] = fields
    write_json(args.output, {'cases': cases, 'shears': shears, 'execution_device': args.device, 'reports': reports, 'summary': summary,
                            'model': candidate.metadata, 'flow': str(flow), 'flow_sha256': sha256(flow),
                            'reference_flow': str(args.reference_flow) if args.reference_flow else None,
                            'reference_flow_sha256': sha256(args.reference_flow) if args.reference_flow else None,
                            'comparison': 'same transport on reference flow' if args.reference_flow else 'candidate self-flow without transport',
                            'parent_manifest_sha256': sha256(manifest_path), 'script_sha256': sha256(__file__),
                            'density_measure': 'physical e1,e2,radius_pixels,flux; conditional on actual usability',
                            'limitations': ['no additive-emulator density baseline',
                                            *(['g0 targets are the original flow-domain measurements identified by the parent hash, not independent density validation'] if 0. in shears else []),
                                            'cut summaries use unnormalized restricted physical density; changed flows may have different selection masses',
                                            'no classifier comparison, new training, or calibration-based model selection']})
    print(f'DISK_DENSITY_COMPLETE {args.output}', flush=True)


if __name__ == '__main__':
    main()
