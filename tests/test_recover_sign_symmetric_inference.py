"""The recovery amendment must reject changed science or incomplete replay."""
from copy import deepcopy

import pytest

from _script_loader import load_script_module

module = load_script_module('recover_sign_symmetric_inference.py')


def fixture():
    old = dict(implementation_sha256={'sampler': 'old'}, invariant_work='cached',
               model_identity={'flow': 'fixed'}, steps=20480, observations=[2, 5], seed=123)
    new = dict(old, implementation_sha256={'sampler': 'new'}, invariant_work='full',
               nuisance_likelihood_cache=False)
    receipt = dict(status='completed_full_unit', observations=2,
        current_state_audit=dict(likelihood_calls_skipped=0, checked_states=100,
                                maximum_summary_change=0., maximum_log_kernel_change=0.))
    return old, new, receipt


def test_only_reviewed_cache_change_is_accepted():
    old, new, receipt = fixture()
    assert module.validate_repaired(old, new, receipt, old['model_identity']) == [
        'implementation_sha256', 'invariant_work', 'nuisance_likelihood_cache']


@pytest.mark.parametrize('key,value', [('seed', 124), ('observations', [5, 2]),
                                     ('steps', 10240), ('model_identity', {'flow': 'other'}),
                                     ('nuisance_likelihood_cache', True)])
def test_changed_protocol_is_rejected(key, value):
    old, new, receipt = fixture()
    new[key] = value
    with pytest.raises(ValueError):
        module.validate_repaired(old, new, receipt, old['model_identity'])


@pytest.mark.parametrize('key,value', [('checked_states', 0), ('likelihood_calls_skipped', 1),
                                     ('maximum_log_kernel_change', 1e-7),
                                     ('maximum_summary_change', float('nan'))])
def test_replay_must_match_completed_zero_error_evidence(key, value):
    old, new, receipt = fixture()
    receipt['current_state_audit'][key] = value
    with pytest.raises(ValueError):
        module.validate_repaired(old, new, receipt, old['model_identity'])


def test_incomplete_receipt_is_rejected():
    old, new, receipt = fixture()
    bad = deepcopy(receipt)
    bad['observations'] = 1
    with pytest.raises(ValueError):
        module.validate_repaired(old, new, bad, old['model_identity'])
