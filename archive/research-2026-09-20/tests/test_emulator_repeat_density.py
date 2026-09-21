from copy import deepcopy

import pytest

from scripts.audit_emulator_repeat_density import require_fixed_likelihood_parts


def test_emulator_density_rejects_changed_flow_or_selected_mass():
    reference = {'cases': [1], 'h': .02, 'draws': 64, 'seed': 7301, 'population': 'parent',
        'cohorts': ['truth_parent'], 'cuts': ['uncut'], 'usability': 'predicted', 'case_reports': {},
        'inputs': {'parent_manifest_sha256': 'parent', 'classifiers': ['classifier'], 'rblend': 'pairs',
                   'flow_sha256': 'flow'},
        'experimental_response': {'composition': 'mobius', 'output_conditions': 'radius_flux',
                                  'prediction_device': 'cuda', 'metadata': {'model_sha256': 'old'}},
        'per_case_sufficient': {'branch': {'model': {leg: {'denominator': [[10.]]}
                                                    for leg in ('plus', 'minus')}}}}
    candidate = deepcopy(reference)
    candidate['experimental_response']['metadata']['model_sha256'] = 'new'
    require_fixed_likelihood_parts(reference, candidate)
    changed = deepcopy(candidate)
    changed['inputs']['flow_sha256'] = 'other-flow'
    with pytest.raises(ValueError, match='changed flow'):
        require_fixed_likelihood_parts(reference, changed)
    changed = deepcopy(candidate)
    changed['per_case_sufficient']['branch']['model']['plus']['denominator'] = [[11.]]
    with pytest.raises(AssertionError):
        require_fixed_likelihood_parts(reference, changed)
