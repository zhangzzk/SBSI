"""Compare selected finite-parent densities using frozen actual-U likelihoods/masses.

For a uniformly chosen supplied parent i, conditional on its observed usable
flag and the measured cut S, q(i,y|S,U_actual) is proportional to U_i*q_i(y)*S(y).
The candidate-minus-reference mean NLL is therefore the raw observed-row NLL
change plus log(M_candidate/M_reference), with M=sum_i U_i P_q(S|i).
This is a pooled parent-and-output score, not density conditional on each i,
not predictive detection validation, and not a calibration estimator.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from scripts.train_full_parent_classifier import sha256, write_json


def score_change(count, raw_delta, reference_mass, candidate_mass, weights):
    count, raw_delta, reference_mass, candidate_mass = (
        np.asarray(x, float) for x in (count, raw_delta, reference_mass, candidate_mass))
    weights = np.asarray(weights, float)
    if (count.ndim != 1 or not len(count)
            or any(x.shape != count.shape for x in (raw_delta, reference_mass, candidate_mass))
            or weights.ndim != 2 or weights.shape[1] != len(count)
            or not all(np.isfinite(x).all() for x in (count, raw_delta, reference_mass, candidate_mass, weights))
            or any((x <= 0).any() for x in (count, reference_mass, candidate_mass))
            or (weights < 0).any() or (weights.sum(1) <= 0).any()):
        raise ValueError('finite aligned positive counts/masses and nonnegative case weights required')
    raw = weights@(count*raw_delta)/(weights@count)
    normalization = np.log((weights@candidate_mass)/(weights@reference_mass))
    return {'raw_delta_nll': raw, 'log_selection_mass_ratio': normalization,
            'normalized_delta_nll': raw+normalization}


def compare(reference, candidate, density):
    for field in ('cases', 'h', 'cuts', 'draws', 'seed', 'population', 'cohorts', 'case_reports'):
        if reference[field] != candidate[field]:
            raise ValueError('response population/execution mismatch: '+field)
    cases = candidate['cases']
    if (len(cases) < 2 or len(set(cases)) != len(cases) or density['cases'] != cases
            or density['shears'] != [candidate['h'], -candidate['h']]
            or density['flow_sha256'] != candidate['inputs']['flow_sha256']
            or density['reference_flow_sha256'] != reference['inputs']['flow_sha256']
            or density['parent_manifest_sha256'] != candidate['inputs']['parent_manifest_sha256']
            or reference['inputs']['parent_manifest_sha256'] != candidate['inputs']['parent_manifest_sha256']
            or density['model'] != candidate['experimental_response']['metadata']
            or density['model'] != reference['experimental_response']['metadata']):
        raise ValueError('density model/population contract mismatch')
    rows = {(r['case'], r['shear']): r for r in density['reports']}
    if len(rows) != len(density['reports']) or set(rows) != {(c, h) for c in cases for h in density['shears']}:
        raise ValueError('unique complete density endpoint records required')
    # Preserve the original density audit's bootstrap stream and replication count.
    picks = np.random.default_rng(20260918).integers(len(cases), size=(2000, len(cases)))
    weights = np.concatenate((np.ones((1, len(cases))),
                              np.stack([np.bincount(p, minlength=len(cases)) for p in picks])))
    result = {}
    for leg, h in (('plus', candidate['h']), ('minus', -candidate['h'])):
        for cut in candidate['cuts']:
            key = 'truth_parent/'+cut+'/actual_usable_flags'
            old = reference['per_case_sufficient'][key]
            new = candidate['per_case_sufficient'][key]
            for field in ('numerator', 'denominator'):
                np.testing.assert_array_equal(old['measured'][leg][field], new['measured'][leg][field])
            reports = [rows[c, h] for c in cases]
            for c, row in zip(cases, reports):
                if row['measurement'] != candidate['case_reports'][str(c)]['legs'][leg]:
                    raise ValueError('density/response measurement receipt mismatch')
            count = np.array([r['cuts'][cut]['rows'] for r in reports])
            np.testing.assert_array_equal(count, np.asarray(new['measured'][leg]['denominator'])[:, 0])
            raw = np.array([r['cuts'][cut]['paired_delta_nll'] for r in reports])
            old_mass = np.asarray(old['model'][leg]['denominator'])[:, 0]
            new_mass = np.asarray(new['model'][leg]['denominator'])[:, 0]
            changes = score_change(count, raw, old_mass, new_mass, weights)
            result[leg+'/'+cut] = {
                'observed_rows': int(count.sum()), 'reference_selected_mass': float(old_mass.sum()),
                'candidate_selected_mass': float(new_mass.sum()),
                'scores': {name: {'estimate': float(v[0]), 'case_bootstrap_se': float(v[1:].std(ddof=1)),
                                  'case_bootstrap_ci95': np.quantile(v[1:], [.025, .975])}
                           for name, v in changes.items()},
                'per_case': {'observed_count': count, 'raw_delta_nll': raw,
                             'reference_mass': old_mass, 'candidate_mass': new_mass}}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('reference', 'candidate', 'density', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    paths = (args.reference, args.candidate, args.density)
    inputs = [json.loads(p.read_text()) for p in paths]
    hashes = {str(p): sha256(p) for p in paths}
    result = compare(*inputs)
    if any(sha256(p) != hashes[str(p)] for p in paths):
        raise ValueError('cached inputs changed during normalization')
    write_json(args.output, {'status': 'completed', 'cases': inputs[1]['cases'], 'results': result,
        'sources': hashes, 'script_sha256': sha256(Path(__file__)), 'calibration_claim': False,
        'formula': 'delta_mean_NLL_selected = delta_mean_raw_NLL + log(sum_candidate_selected_mass/sum_reference_selected_mass)',
        'measure': 'pooled finite parent identity and physical e1,e2,radius_pixels,flux, conditional on actual U and measured cut',
        'bootstrap_seed': 20260918, 'bootstrap_replicates': 2000,
        'limitations': ['actual observed U is an oracle; no predicted detection claim',
                        'not the selected-output density conditional separately on each parent',
                        'selection masses use 64-draw frozen integration; bootstrap excludes finite-draw and training uncertainty',
                        'relative score against initial single flow does not establish absolute density adequacy',
                        'all cuts and both legs retained; no fitting or model selection']})
    for key, value in result.items():
        print('SELECTED_DENSITY', key, value['scores']['normalized_delta_nll'], flush=True)


if __name__ == '__main__':
    main()
