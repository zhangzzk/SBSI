"""Bind prospective scene products, then evaluate the previously frozen candidate."""
import argparse
import json
import os
from pathlib import Path

from scripts.prepare_independent_joint_validation import require_declared_scene, validate_receipts
from scripts.train_full_parent_classifier import sha256, write_json
from scripts.validate_smooth_classifier import execute_evaluator


def check_pins(files):
    for name, expected in files.items():
        if sha256(Path(name)) != expected:
            raise ValueError('frozen independent-validation input changed: '+name)


def require_design_endpoints(endpoints, design):
    """Reject omitted axes, adaptive subsets and replacement shear amplitudes."""
    expected = [(axis, design['primary_amplitude'], design['cases']) for axis in (1, 2)]
    expected += [(axis, design['additional_amplitude'], design['additional_amplitude_cases']) for axis in (1, 2)]
    actual = [(e['axis'], e['h'], e['cases']) for e in endpoints]
    if actual != expected or len({e['name'] for e in endpoints}) != 4:
        raise ValueError('all four predeclared endpoints required in fixed order')
    for endpoint in endpoints:
        args = endpoint['arguments']
        if (args.count('--cases') != 1 or args.count('--output') != 1
                or args.count('--axis') != 1 or args.count('--h') != 1):
            raise ValueError('unique endpoint arguments required')
        start = args.index('--cases')+1
        end = next((i for i in range(start, len(args)) if args[i].startswith('--')), len(args))
        if ([int(v) for v in args[start:end]] != endpoint['cases']
                or int(args[args.index('--axis')+1]) != endpoint['axis']
                or float(args[args.index('--h')+1]) != endpoint['h']
                or args[args.index('--output')+1] != endpoint['output']):
            raise ValueError('evaluator arguments differ from declared endpoint')


def bind_prepared(protocol, design):
    root = Path(protocol['prepared_root'])
    preparation_path, parent_path = root/'preparation.json', root/'parent/manifest.json'
    geometry_path = root/'geometry/manifest.json'
    prep, parent, geometry = (json.loads(p.read_text()) for p in (preparation_path, parent_path, geometry_path))
    cases, design_hash = design['cases'], sha256(Path(protocol['design_protocol']))
    if (prep['status'] != 'completed' or prep['cases'] != cases or parent['cases'] != cases
            or geometry['cases'] != cases or geometry['with_corrected_labels']
            or prep['design_protocol_sha256'] != design_hash
            or parent['sources']['design_protocol_sha256'] != design_hash
            or prep['parent_manifest_sha256'] != sha256(parent_path)
            or prep['script_sha256'] != sha256(Path('scripts/prepare_independent_joint_validation.py'))):
        raise ValueError('complete matching independent preparation required')
    expected_groups = {'g0': cases, 'g1_h0.05': cases, 'g2_h0.05': cases,
                       'g1_h0.02': design['additional_amplitude_cases'],
                       'g2_h0.02': design['additional_amplitude_cases']}
    if prep['group_cases'] != expected_groups:
        raise ValueError('prepared shear groups differ from design')
    files = protocol['files']
    for path in (preparation_path, parent_path, geometry_path, root/'pair_lookup.feather'):
        files[str(path)] = sha256(path)
    check_pins(geometry['pins'])
    files.update(geometry['pins'])
    for case in cases:
        scene = Path(protocol['image_root'])/f'case{case}'
        manifest_path, audit_path, measured_path = (scene/name for name in
            ('manifest.json', 'render_validation.json', 'measurement_manifest.json'))
        manifest, audit, measured = (json.loads(p.read_text()) for p in (manifest_path, audit_path, measured_path))
        require_declared_scene(manifest, design)
        validate_receipts(manifest, audit, measured, sha256(manifest_path), sha256(audit_path), small_smoke=False)
        record = parent['case_reports'][str(case)]
        anchor, pair = root/f'parent/case{case:03d}.npz', root/f'pairs/case{case:03d}.feather'
        metadata_path, geometry_case = pair.with_suffix('.json'), root/f'geometry/cases/case{case:03d}.npz'
        metadata = json.loads(metadata_path.read_text())
        geom = geometry['case_records'][str(case)]
        if (record['scene_manifest_sha256'] != sha256(manifest_path)
                or record['render_audit_sha256'] != sha256(audit_path)
                or record['measurement_manifest_sha256'] != sha256(measured_path)
                or record['sha256'] != sha256(anchor) or metadata['anchor_sha256'] != record['sha256']
                or metadata['case'] != case or metadata['sha256'] != sha256(pair)
                or geom['anchor_sha256'] != record['sha256'] or geom['pair_sha256'] != metadata['sha256']
                or geom['pair_metadata_sha256'] != sha256(metadata_path)
                or geom['prepared_sha256'] != sha256(geometry_case)):
            raise ValueError('prepared case provenance mismatch: '+str(case))
        check_pins(record['source_sha256'])
        files.update(record['source_sha256'])
        for path in (anchor, pair, metadata_path, geometry_case, manifest_path, audit_path, measured_path):
            files[str(path)] = sha256(path)
        print('INDEPENDENT_CASE_BOUND', case, record['truth_parent_rows'], flush=True)


def require_result(result, endpoint, protocol, design):
    if (result['cases'] != endpoint['cases'] or result['h'] != endpoint['h']
            or result['shear_axis'] != endpoint['axis'] or result['seed'] != 7301
            or result['draws'] != 64 or result['cohorts'] != ['truth_parent']
            or result['operational_single_case'] or result['predictive_diagnostics'] is None):
        raise ValueError('result differs from frozen evaluation contract')
    inputs = result['inputs']
    args = endpoint['arguments']
    flow = args[args.index('--flow')+1]
    smooth = args[args.index('--classifier-smooth')+1]
    model = str(Path(args[args.index('--response-candidate')+1])/'model.json')
    if (inputs['flow_sha256'] != design['frozen_model_sha256'][flow]
            or inputs['classifier_arms']['modeled_smooth'] != [{'path': smooth, 'sha256': design['frozen_model_sha256'][smooth]}]
            or result['experimental_response']['metadata']['model_sha256'] != design['frozen_model_sha256'][model]
            or inputs['parent_manifest_sha256'] != protocol['files'][str(Path(protocol['prepared_root'])/'parent/manifest.json')]):
        raise ValueError('result did not use the original frozen candidate and bound parents')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--freeze-output', type=Path)
    parser.add_argument('--index', type=int)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Slurm required')
    protocol = json.loads(args.protocol.read_text())
    check_pins(protocol['files'])
    design = json.loads(Path(protocol['design_protocol']).read_text())
    check_pins(design['source_sha256'] | design['frozen_model_sha256'])
    require_design_endpoints(protocol['endpoints'], design)
    if args.freeze_output is not None:
        if args.index is not None or args.freeze_output.exists():
            raise ValueError('new frozen protocol required')
        bind_prepared(protocol, design)
        protocol['template_sha256'] = sha256(args.protocol)
        check_pins(protocol['files'])
        write_json(args.freeze_output, protocol)
        print('INDEPENDENT_JOINT_BOUND', len(protocol['files']), flush=True)
        return
    if args.index is None or 'template_sha256' not in protocol:
        raise ValueError('bound data protocol and endpoint index required')
    endpoint = protocol['endpoints'][args.index]
    if Path(endpoint['output']).exists() or Path(endpoint['receipt']).exists():
        raise FileExistsError('refusing to overwrite independent evidence')
    execute_evaluator(protocol, Path('scripts/evaluate_smooth_classifier_joint.py'), endpoint['arguments'])
    result = json.loads(Path(endpoint['output']).read_text())
    require_result(result, endpoint, protocol, design)
    check_pins(protocol['files'])
    write_json(Path(endpoint['receipt']), {'protocol_sha256': sha256(args.protocol),
        'result_sha256': sha256(Path(endpoint['output'])), 'endpoint': endpoint['name'],
        'original_candidate_retained': True, 'calibration_claim': False})
    primary = result['summary']['truth_parent/mag25.8_radius_gt0.60/modeled_smooth']
    print('INDEPENDENT_JOINT_COMPLETE', endpoint['name'], primary['m_percent'],
          primary['m_standard_error_percentage_points'], flush=True)


if __name__ == '__main__':
    main()
