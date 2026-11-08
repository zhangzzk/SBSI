"""Prepare truth-only contexts from the audited fresh rendering pilot."""
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('truth-context preparation requires Slurm')
    audit_path = args.pilot_root/'render_validation.json'
    audit = json.loads(audit_path.read_text())
    manifest_path = args.pilot_root/'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    measurement_path = args.pilot_root/'measurement_manifest.json'
    measured = json.loads(measurement_path.read_text())
    if (audit['status'] != 'passed' or measured['status'] != 'completed'
            or audit['manifest_sha256'] != sha256(manifest_path)
            or measured['render_audit_sha256'] != sha256(audit_path)
            or measured['case'] != manifest['case']):
        raise ValueError('fresh pilot preparation/measurement provenance mismatch')
    case = manifest['case']
    zero = manifest['legs'][0]
    if zero['sky_shear'] != [0., 0.] or zero['group'] != 'g0':
        raise ValueError('explicit zero-shear truth leg required')
    generated_path = Path(zero['catalogue'])
    truth_path = args.pilot_root/f'g0/case{case}_0.0/real0/catalogues/input/gals_info_tile180.0_-0.5.feather'
    if (sha256(generated_path) != zero['catalogue_sha256']
            or sha256(truth_path) != audit['legs'][0]['truth_sha256']):
        raise ValueError('pilot truth changed')
    generated = pd.read_feather(generated_path)
    truth = prepare_truth(truth_path)
    primary, counts = select_truth_parent(generated, truth)
    context = intrinsic_context(primary, truth)
    ids = primary['index'].to_numpy(np.int64)
    args.output_root.mkdir(parents=True, exist_ok=False)
    output = args.output_root/f'case{case:03d}.npz'
    np.savez(output, input_index=ids, context=context)
    write_json(args.output_root/'manifest.json', {
        'format': 'fresh_multiaxis_pilot_truth_parent_v1', 'cases': [case],
        'population': 'complete generated secondary-role truth parent; strict true r<26; rendered identities only',
        'case_reports': {str(case): {**counts, 'sha256': sha256(output)}},
        'features': list(FLOW_FEATURES), 'conditions': CONDITIONS, 'cohorts': ['truth_parent'],
        'g0_diagnostics_available': False,
        'sources': {'render_audit_sha256': sha256(audit_path),
                    'measurement_manifest_sha256': sha256(measurement_path),
                    'script_sha256': sha256(__file__), 'truth_sha256': sha256(truth_path)},
        'limitations': ['one small-area operational pilot, not calibration acceptance',
                        'g0 measurements exist but are deliberately not used to define parent membership or cohorts']})
    print(f'FRESH_PILOT_PARENT case={case} rows={len(ids)} dropped={counts["unrendered_parent_dropped"]}', flush=True)


if __name__ == '__main__':
    main()
