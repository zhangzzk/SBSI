#!/usr/bin/env python
"""Screen the asinh experiment using training validation before external replay.

This does not promote a model. It only permits the predeclared ConstGold test
when full harmonized validation NLL improves by at least .005 nat per row and
the original per-leg validation guard does not worsen relative to bounded F8.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np
import torch


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument('--root', type=Path, required=True)
    a.add_argument('--reference', type=Path, required=True)
    a.add_argument('--forward-reference', type=Path, required=True)
    a.add_argument('--backtracking', action='store_true', help='Screen accepted backtracking refinement instead')
    a.add_argument('--evaluation-after', help='Slurm array prerequisite for the combined GPU ceiling')
    args = a.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('scheduler allocation required')
    root, reference = args.root, args.reference
    new_dir = root / ('refinement' if args.backtracking else 'joint/joint_probability_refinement_full')
    old_dir = reference / 'joint/joint_probability_refinement_full'
    new, old = [json.loads((p / 'completed.json').read_text()) for p in (new_dir, old_dir)]
    states = [torch.load(p / 'selected.pt', map_location='cpu', weights_only=False) for p in (new_dir, old_dir)]
    for key in ('means', 'scales'):
        np.testing.assert_array_equal(states[0]['target_transform'][key], states[1]['target_transform'][key])
    symmetry = json.loads((root / ('refinement_symmetry.json' if args.backtracking else 'final_symmetry.json')).read_text())
    new_flow = new_dir / 'selected.pt'
    if digest(new_flow) != new['selected_sha256'] or symmetry['status'] != 'passed':
        raise ValueError('checkpoint or symmetry audit mismatch')
    improvement = old['selected_full_validation_nll'] - new['selected_full_validation_nll']
    passed = bool(improvement >= .005 and new['selected_guard_loss'] <= old['selected_guard_loss'])
    rule = 'Full validation density improvement and no per-leg guard degradation; no ConstGold selection'
    if args.backtracking:
        passed = bool(new['best_step'] > 0 and new['eligible_for_joint_validation']
            and improvement >= -.001 and new['selected_guard_loss'] < old['selected_guard_loss']
            and np.all(np.abs(new['selected_selection_mass']['residual']) <=
                       np.abs(old['selected_selection_mass']['residual']) + 1e-12))
        rule = 'Accepted common-descent refinement with original full-density and soft/hard probability gates'
    screen = dict(passed=passed, nll_improvement_nats_per_row=improvement,
        minimum_improvement=-.001 if args.backtracking else .005, new_guard_loss=new['selected_guard_loss'],
        reference_guard_loss=old['selected_guard_loss'], production_accepted=False,
        rule=rule)
    (root / 'validation_screen.json').write_text(json.dumps(screen, indent=2) + '\n')
    print('VALIDATION_SCREEN', json.dumps(screen), flush=True)
    if not passed:
        return
    output = root / 'forward_validation'
    output.mkdir(exist_ok=False)
    for name in ('logs', 'cases'):
        (output / name).mkdir()
    for name in ('evaluate.py', 'job.sh', 'detect_resources.py'):
        shutil.copy2(args.forward_reference / name, output / name)
    p = json.loads((args.forward_reference / 'protocol.json').read_text())
    runtime = reference if args.backtracking else root
    p.update(name='Screened symmetric F8 predeclared external comparison', training=str(runtime),
             flow=str(new_flow), flow_sha256=digest(new_flow), validation_screen=screen)
    for file in (new_flow, root / 'validation_screen.json', output / 'evaluate.py',
                 output / 'job.sh', output / 'detect_resources.py'):
        p['hashes'][str(file)] = digest(file)
    (output / 'protocol.json').write_text(json.dumps(p, indent=2) + '\n')
    def submit(options, stage):
        return subprocess.check_output(['sbatch', '--parsable', *options, str(output / 'job.sh'),
            str(output), str(runtime), stage], text=True).strip().split(';')[0]
    dependency = [] if not args.evaluation_after else [f'--dependency=afterany:{args.evaluation_after}']
    evaluation = submit(['-p', 'inter', '--gres=gpu:h200nvl:1', '--array=0-7%1',
        '--job-name=signsym-screened-forward', '--output', str(output / 'logs/evaluate_%A_%a.out'), *dependency], 'evaluate')
    summary = submit(['-p', 'cluster', '--cpus-per-task=4', '--mem=8G', '--time=00:15:00',
        '--kill-on-invalid-dep=yes', f'--dependency=afterok:{evaluation}',
        '--job-name=signsym-asinh-summary', '--output', str(output / 'logs/summary_%j.out')], 'summary')
    (output / 'jobs.json').write_text(json.dumps(dict(evaluate=evaluation, summary=summary), indent=2) + '\n')
    print('EXTERNAL_COMPARISON_SUBMITTED', evaluation, summary, flush=True)


if __name__ == '__main__':
    main()
