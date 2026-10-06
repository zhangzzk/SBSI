import copy

import pytest

from scripts.audit_flow_repeat_density import require_fixed_likelihood_parts


def test_flow_change_allowed_but_classifier_and_transport_changes_rejected():
    reference = {'inputs': {'flow': 'old', 'flow_sha256': 'oldhash', 'classifier_arms': {'smooth': 'fixed'},
                            'script_sha256': 'fixed'}, 'experimental_response': {'metadata': {'model': 'fixed'}}}
    candidate = copy.deepcopy(reference)
    candidate['inputs'].update(flow='new', flow_sha256='newhash')
    require_fixed_likelihood_parts(reference, candidate)
    candidate['inputs']['classifier_arms']['smooth'] = 'different'
    with pytest.raises(ValueError, match='only the flow'):
        require_fixed_likelihood_parts(reference, candidate)
    candidate = copy.deepcopy(reference)
    candidate['experimental_response']['metadata']['model'] = 'different'
    with pytest.raises(ValueError, match='only the flow'):
        require_fixed_likelihood_parts(reference, candidate)
