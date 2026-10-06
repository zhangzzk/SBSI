"""Prepare independent truth parents without zero-shear measurement anchors."""
import argparse
import os
from pathlib import Path
import json

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc

from sbsi.fixed_g0_domain import FLOW_FEATURES, circularized_radius
from sbsi.forward_catalogue import prepare_flow_inputs
from scripts.build_constgold_fixed_g0_blend_lookup import CONDITIONS, prepare_truth
from scripts.train_fixed_g0_flow import sha256, write_json


def select_truth_parent(generated, rendered):
    """Apply the generated structural role and strict magnitude cut before joining."""
    identity = generated['index'].to_numpy(np.int64)
    if not np.array_equal(np.sort(identity), np.arange(len(generated))):
        raise ValueError('generated identities must be unique and contiguous')
    magnitude = generated['r'].to_numpy(float)
    if not np.isfinite(magnitude).all() or rendered['index'].duplicated().any():
        raise ValueError('invalid generated magnitudes or duplicate rendered identities')
    wanted = np.sort(identity[(identity >= len(generated)//2) & (magnitude < 26.)])
    aligned = rendered.set_index('index', drop=False).reindex(wanted)
    present = aligned['index'].notna().to_numpy()
    selected = aligned.loc[present].reset_index(drop=True)
    selected['index'] = selected['index'].astype(np.int64)
    if selected.empty:
        raise ValueError('empty independent truth parent')
    return selected, {'generated_role_boundary': len(generated)//2,
                      'generated_truth_parent_rows': len(wanted),
                      'truth_parent_rows': len(selected),
                      'unrendered_parent_dropped': int((~present).sum()),
                      'unrendered_parent_ids': wanted[~present].tolist()}


def intrinsic_context(primary, scene):
    """Use intrinsic shapes and size; observed applied shear is not folded twice."""
    frame = prepare_flow_inputs(primary, scene, conditions=CONDITIONS)
    frame['e1_input_p'] = frame['e1_input_rot0_p']
    frame['e2_input_p'] = frame['e2_input_rot0_p']
    frame['circularized_Re_input_p'] = circularized_radius(frame)
    values = frame[list(FLOW_FEATURES)].to_numpy(np.float32)
    if not np.isfinite(values).all():
        raise ValueError('nonfinite independent truth context')
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-root', type=Path, required=True)
    parser.add_argument('--truth-audit', type=Path, required=True)
    parser.add_argument('--crowd-reference', type=Path, required=True)
    parser.add_argument('--cases', type=int, nargs='+', required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('truth crowding preparation requires Slurm')
    if args.output_root.exists():
        raise FileExistsError(args.output_root)
    audit = json.loads(args.truth_audit.read_text())
    if (len(set(args.cases)) != len(args.cases) or not args.cases
            or not set(args.cases) <= set(audit['independent_cases'])):
        raise ValueError('unique cases covered by independent-scene audit required')
    args.output_root.mkdir(parents=True)
    reports, arrays = {}, {}
    for case in sorted(args.cases):
        receipt = audit['case_reports'][str(case)]
        for path, digest in receipt['source_sha256'].items():
            if sha256(path) != digest:
                raise ValueError('truth changed since scene-independence audit')
        generated_path = args.raw_root/f'gals{case}_0.05.feather'
        path = args.raw_root/f'case{case}_0.05/real0/catalogues/input/gals_info_tile180.0_-0.5.feather'
        if str(path) not in receipt['source_sha256'] or str(generated_path) not in receipt['source_sha256']:
            raise ValueError('requested raw root differs from audited truth')
        generated = pd.read_feather(generated_path, columns=['index', 'r'])
        truth = prepare_truth(path)
        primary, counts = select_truth_parent(generated, truth)
        if (counts['generated_truth_parent_rows'] != receipt['true_mag26_target_parent']
                or counts['unrendered_parent_ids'] != receipt['unrendered_parent_ids']):
            raise ValueError('truth-parent membership differs from audit')
        context = intrinsic_context(primary, truth)
        ids = primary['index'].to_numpy(np.int64)
        arrays[case] = (ids, context)
        reports[str(case)] = counts | {'truth_sha256': sha256(path),
                                       'generated_truth_sha256': sha256(generated_path)}
        print(f'INDEPENDENT_PARENT_CONTEXT case={case} rows={len(ids)}', flush=True)
    # Reconstruct from raw truth first. The historical cache is an independent
    # numerical cross-check, never the sole source of new conditions.
    reference = {case: [] for case in args.cases}
    columns = ['case', 'input_index', *FLOW_FEATURES[-3:]]
    with ipc.open_file(args.crowd_reference) as reader:
        for bi in range(reader.num_record_batches):
            frame = pa.Table.from_batches([reader.get_batch(bi)]).select(columns).to_pandas()
            for case, group in frame.loc[frame['case'].isin(args.cases)].groupby('case'):
                reference[int(case)].append(group)
    for case, (ids, context) in arrays.items():
        if not reference[case]:
            raise ValueError('crowding reference lacks requested case')
        cached = pd.concat(reference[case]).set_index('input_index')
        if not cached.index.is_unique:
            raise ValueError('duplicate historical crowding identity')
        compared = cached.reindex(ids)[list(FLOW_FEATURES[-3:])].to_numpy(float)
        if not np.isfinite(compared).all():
            raise ValueError('missing/nonfinite historical crowding identity')
        np.testing.assert_allclose(context[:, -3:], compared, rtol=2e-6, atol=2e-6)
        output = args.output_root/f'case{case:03d}.npz'
        np.savez(output, input_index=ids, context=context)
        reports[str(case)].update(sha256=sha256(output),
                                 crowding_max_abs_difference=float(np.max(np.abs(context[:, -3:]-compared))),
                                 g0_diagnostics_available=False)
        print(f'INDEPENDENT_PARENT_PREPARED case={case} crowding_check=passed', flush=True)
    write_json(args.output_root/'manifest.json', {
        'format': 'independent_truth_parent_v1',
        'population': 'complete generated secondary-role truth parent; strict true r<26; rendered identities only; no measured or g0 selection',
        'features': list(FLOW_FEATURES), 'cases': sorted(args.cases), 'case_reports': reports,
        'cohorts': ['truth_parent'], 'g0_diagnostics_available': False,
        'sources': {'truth_audit_sha256': sha256(args.truth_audit),
                    'crowd_reference_sha256': sha256(args.crowd_reference),
                    'script_sha256': sha256(__file__),
                    'forward_catalogue_sha256': sha256(Path(__file__).resolve().parents[1]/'sbsi/forward_catalogue.py')},
        'conditions': CONDITIONS,
        'limitations': ['no observed g0 catalogue; no g0 masks or measurements are invented',
                        'measurement convention compatibility remains a separate gate',
                        'existing historical g1-only endpoint; no fresh multi-axis acceptance claim']})
    print(f'INDEPENDENT_PARENT_COMPLETE {args.output_root}', flush=True)


if __name__ == '__main__':
    main()
