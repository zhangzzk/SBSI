"""Assemble the predeclared independent scene set without measured-g0 selection."""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from sbsi.fixed_g0_domain import FLOW_FEATURES
from scripts.build_constgold_fixed_g0_blend_lookup import CONDITIONS, prepare_truth
from scripts.prepare_independent_truth_parent import intrinsic_context, select_truth_parent
from scripts.train_fixed_g0_flow import sha256, write_json


def common_rendered_parent(primary, rendered_by_leg):
    """Drop and report truth parents absent from any rendered leg, before U cuts."""
    ids = primary['index'].to_numpy(np.int64)
    if len(np.unique(ids)) != len(ids) or not rendered_by_leg:
        raise ValueError('unique parents and nonempty rendered legs required')
    keep = np.ones(len(ids), dtype=bool)
    missing = {}
    for name, rendered in rendered_by_leg.items():
        rendered = np.asarray(rendered)
        if rendered.ndim != 1 or len(np.unique(rendered)) != len(rendered):
            raise ValueError('unique rendered identities required')
        present = np.isin(ids, rendered)
        missing[name] = ids[~present].tolist()
        keep &= present
    if not keep.any():
        raise ValueError('no common rendered truth parents')
    return primary.loc[keep].reset_index(drop=True), {
        'g0_rendered_parent_rows': len(ids), 'common_rendered_parent_rows': int(keep.sum()),
        'additional_unrendered_parent_dropped': int((~keep).sum()),
        'additional_unrendered_parent_ids': ids[~keep].tolist(), 'missing_parent_ids_by_leg': missing}


def validate_receipts(manifest, audit, measured, manifest_hash, audit_hash, *, small_smoke):
    if (manifest['status'] != 'render_commands_completed' or audit['status'] != 'passed'
            or measured['status'] != 'completed' or audit['manifest_sha256'] != manifest_hash
            or measured['render_audit_sha256'] != audit_hash
            or len(manifest['legs']) != len(audit['legs']) or len(manifest['legs']) != len(measured['legs'])
            or len({manifest['case'], audit['case'], measured['case']}) != 1):
        raise ValueError('completed matching scene/render/measurement receipts required')
    if small_smoke:
        if not 0 < manifest['area_deg2'] <= .01:
            raise ValueError('small operational scene required')
    elif (manifest.get('scope') != 'full_scene_development' or manifest['area_deg2'] != 1.
          or manifest.get('reference_seed') != 123
          or manifest.get('scene_count_rule') != 'exact original generated count after prior replay'):
        raise ValueError('full-area development with exact original-prior replay required')
    keys = [(leg['group'], leg['h']) for leg in manifest['legs']]
    if len(set(keys)) != len(keys) or keys[0] != ('g0', 0.) or manifest['legs'][0]['sky_shear'] != [0., 0.]:
        raise ValueError('distinct legs with explicit g0 first required')
    if not small_smoke:
        by_key = {(leg['group'], leg['h']): leg['sky_shear'] for leg in manifest['legs']}
        for axis in (1, 2):
            for h in (-.05, .05):
                expected = [h, 0.] if axis == 1 else [0., h]
                if by_key.get((f'g{axis}_h0.05', h)) != expected:
                    raise ValueError('both coherent axes at +/-0.05 required')
    for leg, rendered, fitted in zip(manifest['legs'], audit['legs'], measured['legs']):
        if any((item['group'], item['h']) != (leg['group'], leg['h']) for item in (rendered, fitted)):
            raise ValueError('leg receipt order mismatch')
        if fitted['sky_shear'] != leg['sky_shear']:
            raise ValueError('measurement shear identity mismatch')


def require_declared_scene(manifest, design):
    """Enforce the prospectively declared seeds and complete shear-leg set."""
    case = manifest['case']
    if case not in design['cases']:
        raise ValueError('undeclared scene')
    index = design['cases'].index(case)
    if (manifest['scene_seed'] != design['scene_seeds'][index]
            or manifest['noise_seed'] != design['noise_seeds'][index]
            or manifest['area_deg2'] != design['area_deg2']):
        raise ValueError('scene seed, noise or area differs from design')
    amplitudes = [design['primary_amplitude']]
    if case in design['additional_amplitude_cases']:
        amplitudes.append(design['additional_amplitude'])
    expected = {('g0', 0., (0., 0.))}
    for axis in (1, 2):
        for h in amplitudes:
            for sign in (-1, 1):
                vector = (sign*h, 0.) if axis == 1 else (0., sign*h)
                expected.add((f'g{axis}_h{h:g}', sign*h, vector))
    actual = {(leg['group'], leg['h'], tuple(leg['sky_shear'])) for leg in manifest['legs']}
    if actual != expected or len(actual) != len(manifest['legs']):
        raise ValueError('shear legs differ from predeclared design')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', action='append', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--design-protocol', type=Path, required=True)
    parser.add_argument('--operational-small-scene', action='store_true')
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('full-scene context preparation requires Slurm')
    if args.output_root.exists():
        raise FileExistsError(args.output_root)
    if not args.operational_small_scene and len(args.scene) < 2:
        raise ValueError('multiple full-area development scenes required')
    design = json.loads(args.design_protocol.read_text())
    if args.operational_small_scene or len(args.scene) != len(design['cases']):
        raise ValueError('all declared full-area scenes required')
    for path, expected in {**design['source_sha256'], **design['frozen_model_sha256']}.items():
        if sha256(path) != expected:
            raise ValueError('independent-validation design input changed: '+path)
    records, prepared, layout, scene_seeds, noise_seeds = {}, {}, {}, [], []
    for root in args.scene:
        manifest_path, audit_path, measurement_path = (root/name for name in ('manifest.json', 'render_validation.json', 'measurement_manifest.json'))
        manifest, audit, measured = (json.loads(path.read_text()) for path in (manifest_path, audit_path, measurement_path))
        validate_receipts(manifest, audit, measured, sha256(manifest_path), sha256(audit_path), small_smoke=args.operational_small_scene)
        require_declared_scene(manifest, design)
        case = manifest['case']
        if case in records:
            raise ValueError('duplicate scene identity')
        scene_seeds.append(manifest['scene_seed'])
        noise_seeds.append(manifest['noise_seed'])
        generated_path = Path(manifest['legs'][0]['catalogue'])
        if sha256(generated_path) != manifest['legs'][0]['catalogue_sha256']:
            raise ValueError('generated reference catalogue changed')
        generated = pd.read_feather(generated_path, columns=['index', 'r'])
        truth_paths, rendered_ids, sources = {}, {}, {}
        for leg, rendered, fitted in zip(manifest['legs'], audit['legs'], measured['legs']):
            name = f'{leg["group"]}/case{case}_{leg["h"]}'
            directory = root/name
            truth_path = directory/'real0/catalogues/input/gals_info_tile180.0_-0.5.feather'
            shape_path = directory/'real0/catalogues/Shapes/shape_catalogue_detect_position_all_tile180.0_-0.5.feather'
            cross_path = directory/'real0/catalogues/CrossMatch/tile180.0_-0.5_rot0_matched.feather'
            if shape_path.resolve() != Path(fitted['output']).resolve():
                raise ValueError('shape path differs from measurement receipt')
            for path, expected in ((truth_path, rendered['truth_sha256']), (shape_path, fitted['output_sha256']), (cross_path, fitted['cross_sha256'])):
                if sha256(path) != expected:
                    raise ValueError(f'changed scene product: {path}')
                sources[str(path)] = expected
            rendered_ids[name] = pd.read_feather(truth_path, columns=['index_input'])['index_input'].to_numpy(np.int64)
            truth_paths[leg['group']] = truth_path
            layout[name] = str(directory.resolve())
        truth = prepare_truth(truth_paths['g0'])
        primary, counts = select_truth_parent(generated, truth)
        primary, common = common_rendered_parent(primary, rendered_ids)
        context = intrinsic_context(primary, truth)
        ids = primary['index'].to_numpy(np.int64)
        prepared[case] = ids, context
        records[str(case)] = counts | common | {'truth_parent_rows': len(ids), 'scene_seed': manifest['scene_seed'],
            'noise_seed': manifest['noise_seed'], 'scene_manifest_sha256': sha256(manifest_path),
            'render_audit_sha256': sha256(audit_path), 'measurement_manifest_sha256': sha256(measurement_path),
            'generated_truth_sha256': sha256(generated_path), 'source_sha256': sources}
        print(f'INDEPENDENT_VALIDATION_PARENT case={case} parents={len(ids)} dropped={counts["unrendered_parent_dropped"]+common["additional_unrendered_parent_dropped"]}', flush=True)
    if sorted(prepared) != sorted(design['cases']):
        raise ValueError('missing declared scenes')
    if len(set(scene_seeds)) != len(scene_seeds) or len(set(noise_seeds)) != len(noise_seeds):
        raise ValueError('independent scene/noise seeds required across cases')
    parent = args.output_root/'parent'
    parent.mkdir(parents=True)
    for case, (ids, context) in prepared.items():
        output = parent/f'case{case:03d}.npz'
        np.savez(output, input_index=ids, context=context)
        records[str(case)]['sha256'] = sha256(output)
    for name, target in layout.items():
        link = args.output_root/'catalogues'/name
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(target, target_is_directory=True)
    groups = {}
    for name in layout:
        group, case_name = name.split('/')
        case = int(case_name.split('_')[0].removeprefix('case'))
        groups.setdefault(group, set()).add(case)
    write_json(parent/'manifest.json', {'format': 'fresh_joint_development_truth_parent_v1',
        'cases': sorted(prepared), 'case_reports': records,
        'population': 'generated secondary-role true-r<26 parent, common rendered identities across all supplied legs; no measured/g0 selection',
        'features': list(FLOW_FEATURES), 'conditions': CONDITIONS, 'cohorts': ['truth_parent'], 'g0_diagnostics_available': False,
        'operational_small_scene': args.operational_small_scene,
        'sources': {'script_sha256': sha256(__file__), 'design_protocol_sha256': sha256(args.design_protocol)},
        'limitations': ['prospective fixed-model scene validation; preparation alone is not calibration acceptance', 'g0 measurements never select truth-parent membership']})
    write_json(args.output_root/'preparation.json', {'status': 'completed', 'cases': sorted(prepared),
        'operational_small_scene': args.operational_small_scene, 'group_cases': {key: sorted(value) for key, value in groups.items()},
        'catalogue_symlinks': layout, 'parent_manifest_sha256': sha256(parent/'manifest.json'),
        'script_sha256': sha256(__file__), 'design_protocol_sha256': sha256(args.design_protocol)})


if __name__ == '__main__':
    main()
