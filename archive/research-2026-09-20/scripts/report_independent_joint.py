"""Report every predeclared independent endpoint without selecting a candidate."""
import argparse
import json
import os
from pathlib import Path

import numpy as np

from scripts.train_full_parent_classifier import sha256, write_json
from scripts.validate_independent_joint import check_pins, require_design_endpoints, require_result

PRIMARY = 'mag25.8_radius_gt0.60'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or args.output_root.exists():
        raise RuntimeError('Slurm and new report directory required')
    protocol = json.loads(args.protocol.read_text())
    check_pins(protocol['files'])
    design = json.loads(Path(protocol['design_protocol']).read_text())
    require_design_endpoints(protocol['endpoints'], design)
    sources, records, density_lines = {}, {}, []
    lines = ['# Prospective two-axis validation of the frozen candidate', '',
        'One flow, one smooth detection/usability classifier, original disk-response emulator; '
        'all models and all 40 scenes fixed before generation.', '',
        'Generated secondary true-r<26; measured FLUX_RADIUS>0.6 arcsec and MAG_AUTO<25.8 '
        'applied independently in each shear leg. Predicted detection/usability is the primary arm.', '',
        'Joint diagonal m (%) with case-bootstrap SE (percentage points) and percentile 95% interval. '
        'Both full-sample axes are primary; the first-four smaller-amplitude checks are auxiliary.', '',
        '| Endpoint | Scenes | Axis | h | m (%) | SE (pp) | 95% interval (%) |',
        '|---|---:|---:|---:|---:|---:|---|']
    protocol_hash = sha256(args.protocol)
    for endpoint in protocol['endpoints']:
        path, receipt_path = Path(endpoint['output']), Path(endpoint['receipt'])
        result, receipt = (json.loads(p.read_text()) for p in (path, receipt_path))
        if (receipt['protocol_sha256'] != protocol_hash or receipt['result_sha256'] != sha256(path)
                or receipt['endpoint'] != endpoint['name'] or not receipt['original_candidate_retained']):
            raise ValueError('verified independent result receipt required')
        require_result(result, endpoint, protocol, design)
        sources.update({str(p): sha256(p) for p in (path, receipt_path)})
        summary = result['summary']['truth_parent/'+PRIMARY+'/modeled_smooth']
        ci = summary['m_ci95_percent']
        lines.append(f'| {endpoint["name"]} | {len(result["cases"])} | {result["shear_axis"]} | '
            f'{result["h"]:g} | {summary["m_percent"]:+.6f} | '
            f'{summary["m_standard_error_percentage_points"]:.6f} | [{ci[0]:+.6f}, {ci[1]:+.6f}] |')
        predictive = result['predictive_diagnostics']
        if predictive['metadata']['moment_coordinates'] != ['e1', 'e2', 'log_radius_pixels', 'MAG_AUTO']:
            raise ValueError('unexpected density moment convention')
        k = predictive['cuts'].index(PRIMARY)
        density = {}
        for leg in ('plus', 'minus'):
            d = predictive['classifier_summary']['modeled_smooth'][leg]
            observed_var = np.diag(np.asarray(d['covariance']['observed'])[k])
            if not np.isfinite(observed_var).all() or (observed_var <= 0).any():
                raise ValueError('positive finite measured variances required')
            mean = 100*np.asarray(d['mean']['generated_minus_observed'])[k]/np.sqrt(observed_var)
            variance = 100*np.diag(np.asarray(d['covariance']['generated_minus_observed'])[k])/observed_var
            if not np.isfinite(mean).all() or not np.isfinite(variance).all():
                raise ValueError('finite predictive discrepancies required')
            hist = d['joint_histogram']
            density[leg] = {'diagnostics': d, 'primary_mean_residual_percent_observed_sd': mean,
                            'primary_variance_residual_percent_observed_variance': variance}
            density_lines.append(f'| {endpoint["name"]} / {leg} | {100*hist["total_variation"]:.4f} | '
                f'{100*d["selection_fraction"]["generated_minus_observed"][k]:+.4f} | '
                f'{variance[0]:+.4f} | {variance[1]:+.4f} |')
        records[endpoint['name']] = {'cases': result['cases'], 'axis': result['shear_axis'], 'h': result['h'],
            'all_response_summaries': result['summary'], 'measurement_counts': result['case_reports'],
            'density': density, 'result_path': str(path), 'receipt_path': str(receipt_path)}
        print('INDEPENDENT_REPORT_ENDPOINT', endpoint['name'], summary['m_percent'], flush=True)
    lines += ['', '## Predictive density discrepancies', '',
        '| Endpoint / leg | Uncut 4D histogram TV (%) | Primary pass fraction residual (pp) | e1 variance residual (%) | e2 variance residual (%) |',
        '|---|---:|---:|---:|---:|', *density_lines, '',
        'Pass fractions are conditional on the usable population, not absolute detection-rate residuals. '
        'Histogram TV is uncut, includes sampling noise and an invalid-photometry bin, and is not an acceptance threshold. '
        'Variance residuals use the fixed primary measured cut. All moments, covariances, support diagnostics, '
        'cuts and response arms remain in evidence.json and its linked source results.', '',
        '## Interpretation limits', '',
        'All scenes and endpoints are retained; no seed, checkpoint, amplitude or cut is selected from these outcomes. '
        'The original model is neither replaced by nor averaged with any training repeat. '
        'The two axes share scenes and are not independent samples; the auxiliary amplitudes reuse the first four scenes. '
        'Case-bootstrap intervals exclude training and finite-draw uncertainty. '
        'Cross-components remain secondary diagnostics. These results require scientific assessment against the approximate '
        '0.3% goal; this report makes no automatic calibration or density-equivalence claim.']
    check_pins(protocol['files'])
    check_pins(sources)
    args.output_root.mkdir(parents=True)
    (args.output_root/'report.md').write_text('\n'.join(lines)+'\n')
    write_json(args.output_root/'evidence.json', {'protocol_sha256': protocol_hash, 'sources': sources,
        'endpoints': records, 'script_sha256': sha256(Path(__file__)),
        'original_candidate_retained': True, 'calibration_claim': False, 'density_equivalence_claim': False})
    print('INDEPENDENT_REPORT_COMPLETE', args.output_root, flush=True)


if __name__ == '__main__':
    main()
