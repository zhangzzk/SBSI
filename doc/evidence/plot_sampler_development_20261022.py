#!/usr/bin/env python
"""Plot pinned completed development evidence; never open held-out outcomes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


DEVELOPMENT_SHA = '09601af0a43c0b5801bc0467b090e588c745713a3c9fd6d86b58f2f9172d1574'
PRIOR_SHA = '9d59a54255a2c5f8c2248f663aa430a2e5e2ccf9f30f89c125ec9e4f7ec86598'
ARMS = ('small', 'large', 'broad')
IDS = [58165, 0, 1, 8]
MEASURES = ('raw_float32_sampled', 'precision_reweighted_float64')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bound_read(path, expected):
    if digest(path) != expected:
        raise ValueError(f'Pinned completed input differs: {path}')
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--development', type=Path, required=True)
    parser.add_argument('--prior', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise ValueError('Run this rendering task in a CPU scheduler allocation')
    development = bound_read(args.development, DEVELOPMENT_SHA)
    prior = bound_read(args.prior, PRIOR_SHA)
    if (development['development_checks_pass'] is not True
            or development['accepted_sampler'] is not False or prior['status'] != 'passed'
            or prior['original_populations'] != 24 or prior['original_target_particles'] != 73728):
        raise ValueError('Complete declared development evidence required')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    precision, sensitivity, allocated_minutes = {}, {}, {}
    for arm in ARMS:
        rows = development['groups'][arm]['observations']
        pr = prior['groups'][arm]['observations']
        if [r['observation'] for r in rows] != IDS or [r['observation'] for r in pr] != IDS:
            raise ValueError('Observation order differs')
        if len(development['groups'][arm]['seeds']) != 8 or not all(r['precision_gate'] for r in rows):
            raise ValueError('Every original population and precision gate required')
        precision[arm] = [float(np.max(np.asarray(r['score']['mcse']) /
            np.sqrt(r['posterior_score_variance_estimate']))) for r in rows]
        sensitivity[arm] = [max(abs(v) for m in MEASURES
            for v in r[m]['shift_over_analytic_mcse'][1:5]) for r in pr]
        allocated_minutes[arm] = development['timing']['scheduler_task_elapsed_seconds_by_arm'][arm] / 60.
    raw_agreement, controlled_agreement = [], []
    comparisons = development['comparisons']
    if set(comparisons) != {'small_minus_large', 'small_minus_broad', 'large_minus_broad'}:
        raise ValueError('All three pairwise comparisons required')
    for index, oid in enumerate(IDS):
        rows = [c['observations'][index] for c in comparisons.values()]
        if any(r['observation'] != oid for r in rows):
            raise ValueError('Pairwise observation order differs')
        raw_agreement.append(max(abs(v) for r in rows
            for v in r['raw']['score']['signed_standardized_difference']))
        controlled_agreement.append(max(abs(v) for r in rows for m in MEASURES
            for v in r['phase'][m]['corrected']['signed_standardized_difference']))
    if (not np.isfinite(list(precision.values()) + list(sensitivity.values())).all()
            or max(max(v) for v in precision.values()) > .05
            or max(raw_agreement + controlled_agreement) > 3.5):
        raise ValueError('Plot values contradict the recorded development gates')
    args.output.mkdir(parents=True, exist_ok=False)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
        'axes.titlesize': 13, 'axes.titleweight': 'bold', 'axes.spines.top': False,
        'axes.spines.right': False, 'svg.fonttype': 'none', 'savefig.facecolor': 'white'})
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.subplots_adjust(left=.09, right=.97, bottom=.14, top=.84, wspace=.27, hspace=.48)
    fig.suptitle('Development passes; independent acceptance pending', x=.09, y=.965,
                 ha='left', fontsize=20, fontweight='bold')
    fig.text(.09, .917, 'Corrected v10 · four fixed observations · eight independent populations per arm',
             fontsize=12, color='#444444')
    colors = ('#0072B2', '#D18A00', '#009E73')
    markers = ('o', '^', 's')
    labels = ('Fitted N256', 'Fitted N1024', 'Broad N1024')
    x = np.arange(4)
    a, b, c, d = axes.ravel()
    for arm, label, color, marker in zip(ARMS, labels, colors, markers):
        a.plot(x, precision[arm], marker=marker, color=color, label=label, lw=1.7)
        d.plot(x, sensitivity[arm], marker=marker, color=color, lw=1.7)
    a.axhline(.05, color='#555555', linestyle='--', lw=1)
    a.text(3.1, .051, 'gate', ha='right', fontsize=10, color='#555555')
    a.set(title='A  Raw-score precision', ylabel='Max MCSE / posterior score SD', ylim=(0, .06))
    a.legend(loc='upper left', bbox_to_anchor=(0., .8), fontsize=9, frameon=False)
    b.plot(x, raw_agreement, 'o-', color='#444444', label='Raw scores', lw=1.7)
    b.plot(x, controlled_agreement, 'D-', color='#9C4D9C', label='Controlled scores, F32 + F64', lw=1.7)
    b.axhline(3.5, color='#555555', linestyle='--', lw=1)
    b.text(3.1, 3.57, 'gate', ha='right', fontsize=10, color='#555555')
    b.set(title='B  Largest pairwise score contrast', ylabel='Absolute difference / combined MCSE', ylim=(0, 4.1))
    b.legend(loc='upper left', bbox_to_anchor=(0., .82), fontsize=9, frameon=False)
    bars = c.bar(np.arange(3), [allocated_minutes[k] for k in ARMS], color=colors,
                 width=.62, edgecolor='white')
    c.bar_label(bars, fmt='%.1f min', padding=4, fontsize=11)
    c.set_xticks(np.arange(3), labels)
    c.set(title='C  Inference allocation only', ylabel='Summed task elapsed time (minutes)', ylim=(0, 100))
    ratio = allocated_minutes['broad'] / allocated_minutes['small']
    c.text(.02, .93, f'Broad / small = {ratio:.2f}×', transform=c.transAxes, fontsize=11)
    c.text(.02, .83, 'Preparation + shared diagnostics excluded',
           transform=c.transAxes, fontsize=9, color='#555555')
    d.set_yscale('log')
    d.axhline(1., color='#555555', linestyle='--', lw=1)
    d.text(3.1, 1.13, '1× current MCSE (scale reference)', ha='right', fontsize=9, color='#555555')
    d.set(title='D  Numerical-versus-analytic prior sensitivity',
          ylabel='Max absolute score shift / original MCSE', ylim=(1e-5, 3.))
    for ax in (a, b, d):
        ax.set_xticks(x, [str(i) for i in IDS])
        ax.set_xlabel('Observation ID')
    for ax in axes.ravel():
        ax.grid(axis='y', alpha=.18)
        ax.set_axisbelow(True)
    fig.text(.09, .065, 'A–B: maxima over both score components; B also spans all three arm pairs. Region and normalizer gates also pass.', fontsize=9)
    fig.text(.09, .038, 'D: both numerical measures, raw + controlled scores, on represented states only. No global shear-bias or missing-mode bound.', fontsize=9)
    paths = []
    for extension in ('png', 'svg'):
        path = args.output / f'development_evidence.{extension}'
        fig.savefig(path, dpi=220)
        paths.append(path)
    plt.close(fig)
    bound_read(args.development, DEVELOPMENT_SHA)
    bound_read(args.prior, PRIOR_SHA)
    result = dict(schema='sampler-development-figure-v1', status='rendered',
        original_observations=IDS, original_populations_per_arm=8,
        development_checks_pass=True, accepted_sampler=False, heldout_outcomes_read=False,
        inputs={str(args.development): DEVELOPMENT_SHA, str(args.prior): PRIOR_SHA},
        script_sha256=digest(__file__), raw_precision=precision,
        max_raw_standardized_contrast=raw_agreement, max_controlled_standardized_contrast=controlled_agreement,
        inference_allocation_minutes=allocated_minutes, max_prior_shift_over_original_mcse=sensitivity,
        outputs={str(p): digest(p) for p in paths}, slurm_job_id=os.environ['SLURM_JOB_ID'],
        runtime=dict(python=sys.version.split()[0], numpy=np.__version__, matplotlib=matplotlib.__version__))
    (args.output/'plotted_data.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(status=result['status'], outputs=result['outputs']), indent=2), flush=True)


if __name__ == '__main__':
    main()
