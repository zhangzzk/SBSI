"""Review absolute predictive discrepancies of the frozen single-flow candidate."""
import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from scripts.train_full_parent_classifier import sha256, write_json

PRIMARY = 'mag25.8_radius_gt0.60'
ENDPOINTS = ('main', 'independent', 'fresh_g1', 'fresh_g2')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or args.output_root.exists():
        raise RuntimeError('Slurm and new output directory required')
    sources, records, means, variances, labels = {}, {}, [], [], []
    lines = ['# Absolute predictive diagnostics of the frozen candidate', '',
        'One flow, disk emulator, original smooth classifier. Predicted detection/usability weights are used.', '',
        'Primary selection: measured FLUX_RADIUS >0.6 arcsec and MAG_AUTO<25.8, generated secondary true-r<26 parent.', '',
        '| Endpoint / leg | Uncut 4D histogram TV (%) | Primary pass-fraction residual (pp) | Primary e1 variance residual (%) | Primary e2 variance residual (%) |',
        '|---|---:|---:|---:|---:|']
    for endpoint in ENDPOINTS:
        path = args.root/f'smooth_joint_{endpoint}.json'
        receipt_path = args.root/f'smooth_joint_{endpoint}_comparison.json'
        digest = sha256(path)
        result = json.loads(path.read_text())
        receipt = json.loads(receipt_path.read_text())
        if receipt['result_sha256'] != digest or receipt['maximum_reference_sufficient_difference'] != 0:
            raise ValueError('verified original-candidate replay required')
        sources.update({str(path): digest, str(receipt_path): sha256(receipt_path)})
        predictive = result['predictive_diagnostics']
        k = predictive['cuts'].index(PRIMARY)
        if predictive['metadata']['moment_coordinates'] != ['e1', 'e2', 'log_radius_pixels', 'MAG_AUTO']:
            raise ValueError('unexpected moment convention')
        endpoint_record = {'cases': result['cases'], 'h': result['h'], 'axis': result['shear_axis'],
                           'cuts': predictive['cuts'], 'legs': {}}
        for leg in ('plus', 'minus'):
            d = predictive['classifier_summary']['modeled_smooth'][leg]
            obs_var = np.diag(np.asarray(d['covariance']['observed'])[k])
            var_delta = np.diag(np.asarray(d['covariance']['generated_minus_observed'])[k])
            if not np.isfinite(obs_var).all() or (obs_var <= 0).any():
                raise ValueError('positive finite observed variances required')
            mean_scaled = 100*np.asarray(d['mean']['generated_minus_observed'])[k]/np.sqrt(obs_var)
            var_scaled = 100*var_delta/obs_var
            if not np.isfinite(mean_scaled).all() or not np.isfinite(var_scaled).all():
                raise ValueError('nonfinite diagnostic')
            hist = d['joint_histogram']
            for source in ('observed', 'generated'):
                np.testing.assert_allclose(np.sum(hist[source])+hist['invalid_photometry_bin_probability'][source], 1, atol=1e-10)
            # All original cut summaries and support diagnostics are retained.
            compact = {key: value for key, value in d.items() if key != 'joint_histogram'}
            compact['uncut_joint_histogram'] = {key: value for key, value in hist.items()
                                               if key not in ('observed', 'generated')}
            compact['primary_mean_residual_percent_observed_sd'] = mean_scaled
            compact['primary_variance_residual_percent_observed_variance'] = var_scaled
            endpoint_record['legs'][leg] = compact
            means.append(mean_scaled)
            variances.append(var_scaled)
            labels.append(f'{endpoint} / {leg}')
            pass_delta = 100*d['selection_fraction']['generated_minus_observed'][k]
            lines.append(f'| {endpoint} / {leg} | {100*hist["total_variation"]:.4f} | '
                         f'{pass_delta:+.4f} | {var_scaled[0]:+.4f} | {var_scaled[1]:+.4f} |')
        records[endpoint] = endpoint_record
        if sha256(path) != digest:
            raise ValueError('input changed during report')
        print('ABSOLUTE_DENSITY_ENDPOINT', endpoint, flush=True)
    lines += ['', 'Pass fractions are conditional on each leg’s usable population; these are not absolute detection-rate residuals.', '',
        'Moment summaries use the fixed primary measured cut, with log moments conditional on strictly positive radius and flux. '
        'The 4D histogram is uncut and includes its separate invalid-photometry bin. Its empirical total variation includes sampling noise; '
        'it is not an estimate of irreducible model error or an acceptance threshold.', '',
        'The figure expresses mean discrepancies as percentages of observed standard deviations, and variance discrepancies as percentages '
        'of observed variances. These are effect sizes, not significance or calibration tests. Raw residuals and their existing paired '
        'case-bootstrap standard errors, all cuts, covariance matrices and physical-support diagnostics are retained in evidence.json.', '',
        'Nonzero discrepancies remain. Relative selected-density gains and these coarse absolute checks do not establish density equivalence. '
        'No model, checkpoint, scene, leg or threshold is selected using this report.']
    plt.rcParams.update({'font.size': 10, 'pdf.fonttype': 42, 'savefig.facecolor': 'white'})
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.5))
    for ax, values, title in zip(axes, (means, variances),
            ('Mean residual (% of observed SD)', 'Variance residual (% of observed variance)')):
        values = np.asarray(values)
        bound = max(float(np.abs(values).max()), .01)
        ax.imshow(values, cmap='RdBu_r', vmin=-bound, vmax=bound, aspect='auto')
        for i in range(len(labels)):
            for j in range(4):
                ax.text(j, i, f'{values[i,j]:+.2f}', ha='center', va='center',
                        color='white' if abs(values[i,j]) > .6*bound else '#202020')
        ax.set_xticks(range(4), ['e1', 'e2', 'log radius', 'magnitude'])
        ax.set_yticks(range(len(labels)), labels)
        ax.set_title(title, fontsize=11, pad=12)
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.suptitle('Remaining predictive discrepancies: fixed single-flow candidate', fontsize=14, x=.03, ha='left')
    fig.text(.03, .91, 'Measured radius >0.6 arcsec; MAG_AUTO<25.8. Model minus observed; predicted detection/usability.', fontsize=10)
    fig.text(.03, .025, 'All four endpoints and both shear legs retained. Effect sizes, not significance tests.\n'
             'Small panels share four scenes; no density-equivalence claim or automatic acceptance threshold.', fontsize=9)
    fig.subplots_adjust(left=.16, right=.98, top=.83, bottom=.13, wspace=.52)
    args.output_root.mkdir(parents=True)
    for suffix in ('png', 'pdf'):
        fig.savefig(args.output_root/f'predictive_discrepancies.{suffix}', dpi=200)
    plt.close(fig)
    (args.output_root/'report.md').write_text('\n'.join(lines)+'\n')
    write_json(args.output_root/'evidence.json', {'sources': sources, 'endpoints': records,
        'script_sha256': sha256(Path(__file__)), 'primary_selection': PRIMARY,
        'usability_arm': 'modeled_smooth', 'calibration_claim': False, 'density_equivalence_claim': False,
        'artifacts': {p.name: sha256(p) for p in args.output_root.iterdir()}})
    print('ABSOLUTE_DENSITY_REPORT_COMPLETE', args.output_root, flush=True)


if __name__ == '__main__':
    main()
