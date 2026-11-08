"""Plot all completed classifier arms at the fixed primary measured cut."""
import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from scripts.train_full_parent_classifier import sha256, write_json

ARMS = ('modeled_usable', 'modeled_control', 'modeled_geometry', 'modeled_coordinate',
        'modeled_transported', 'modeled_smooth', 'actual_usable_flags')
LABELS = ('Historical (3 classifiers)', 'Matched control', 'Static annuli',
          'Transported, old weights', 'Transported, retrained', 'Smooth crowding', 'Observed U (oracle)')
ENDPOINTS = ('main', 'independent', 'fresh_g1', 'fresh_g2')
TITLES = ('Main: 80 scenes, g1, h=0.02', 'Independent: 40 scenes, g1, h=0.05',
          'Additional: 4 scenes, g1, h=0.02', 'Same 4 scenes, g2, h=0.02')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or args.output_root.exists():
        raise RuntimeError('Slurm and new output directory required')
    sources, data = {}, {}
    for endpoint in ENDPOINTS:
        path = args.root/f'smooth_joint_{endpoint}.json'
        receipt_path = args.root/f'smooth_joint_{endpoint}_comparison.json'
        receipt = json.loads(receipt_path.read_text())
        if receipt['result_sha256'] != sha256(path) or receipt['maximum_reference_sufficient_difference'] != 0:
            raise ValueError('verified complete reference replay required')
        result = json.loads(path.read_text())
        records = [result['summary']['truth_parent/mag25.8_radius_gt0.60/'+a] for a in ARMS]
        data[endpoint] = {'m_percent': [r['m_percent'] for r in records],
                          'case_se_pp': [r['m_standard_error_percentage_points'] for r in records]}
        sources.update({str(p): sha256(p) for p in (path, receipt_path)})
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42, 'savefig.facecolor': 'white'})
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    y = np.arange(len(ARMS))[::-1]
    for ax, endpoint, title in zip(axes.flat, ENDPOINTS, TITLES):
        d = data[endpoint]
        ax.axvspan(-.3, .3, color='#E5F1E8', zorder=0)
        ax.axvline(0, color='#84928C', linewidth=.8, zorder=1)
        ax.axhline(.5, color='#C8CDD2', linewidth=.6)
        for i, arm in enumerate(ARMS):
            color = '#0072B2' if arm == 'modeled_smooth' else '#8A6327' if arm == 'actual_usable_flags' else '#707A85'
            ax.errorbar(d['m_percent'][i], y[i], xerr=d['case_se_pp'][i],
                        fmt='D' if arm == 'actual_usable_flags' else 'o', color=color,
                        markersize=6 if arm == 'modeled_smooth' else 4.5,
                        elinewidth=1.4, capsize=3, zorder=3)
        ax.set_title(title, loc='left', fontsize=11, pad=12)
        ax.set_yticks(y, LABELS)
        ax.set_xlim(-2.4, 1.4)
        ax.set_ylim(-.6, 6.7)
        ax.grid(axis='x', color='#E1E4E8', linewidth=.6)
        ax.set_axisbelow(True)
        ax.text(.98, .98, f"Smooth: {d['m_percent'][5]:+.3f}%", transform=ax.transAxes,
                ha='right', va='top', color='#0072B2', fontsize=10)
    for ax in axes[1]:
        ax.set_xlabel('Joint multiplicative bias m (%)')
    fig.suptitle('One fixed flow and emulator; detection/usability classifiers compared',
                 fontsize=14, x=.03, ha='left', y=.985)
    fig.text(.03, .925, 'Measured FLUX_RADIUS > 0.6 arcsec; MAG_AUTO < 25.8. All declared scenes retained.', fontsize=11)
    fig.text(.03, .035, 'Error bars: ±1 case-bootstrap standard error. Shading: ±0.3% reference band, not a confidence test.\n'
             'Bottom panels share the same four scenes. Sampling and training repeats are not shown; this is development evidence.',
             fontsize=9, color='#4F5863')
    fig.subplots_adjust(left=.22, right=.98, top=.865, bottom=.13, hspace=.35, wspace=.15)
    args.output_root.mkdir(parents=True)
    for suffix in ('png', 'pdf'):
        fig.savefig(args.output_root/f'classifier_comparison.{suffix}', dpi=200)
    plt.close(fig)
    write_json(args.output_root/'manifest.json', {'sources': sources, 'values': data, 'arms': ARMS,
        'script_sha256': sha256(Path(__file__)), 'calibration_claim': False,
        'error_definition': 'one case-bootstrap SE; excludes training and finite-draw variation',
        'output_sha256': {p.name: sha256(p) for p in args.output_root.glob('classifier_comparison.*')}})
    print('SMOOTH_COMPARISON_FIGURE_COMPLETE', args.output_root, flush=True)


if __name__ == '__main__':
    main()
