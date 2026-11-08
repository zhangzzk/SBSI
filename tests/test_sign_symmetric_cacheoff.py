"""Check that nuisance proposals use their own kernels and retain replay guards."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import textwrap

import numpy as np
import pytest


spec = importlib.util.spec_from_file_location('cacheoff',
    Path(__file__).resolve().parents[1]/'scripts/setup_sign_symmetric_cacheoff.py')
cacheoff = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cacheoff)


def run_branch(tmp_path, *, block=4, visits=0, replay_error=0.):
    calls, receipts = [], []
    def evaluate(scene, valid, **kwargs):
        calls.append((scene, valid.copy(), kwargs))
        return (np.array([2.+replay_error, 3.]) if scene == 'current'
                else np.array([4., -np.inf])), np.ones((2, 9))
    ns = dict(np=np, block=block, invariant_visits={4: visits},
        logprior=np.array([0., 0.]), newprior=np.array([0., -np.inf]),
        loglike=np.array([2., 3.]), summaries=np.ones((2, 9)),
        scene='current', newscene='proposal', evaluate_scene=evaluate,
        invariant_audit=dict(checked_states=0, maximum_summary_change=0.,
                             maximum_log_kernel_change=0.),
        protocol=dict(replay_tolerances=dict(summary_absolute=1e-10, log_kernel_absolute=1e-6)),
        args=SimpleNamespace(output=tmp_path), save=lambda *args: receipts.append(args))
    exec(textwrap.dedent(cacheoff.REPLACEMENT), ns)
    return ns, calls, receipts


@pytest.mark.parametrize('block,visits,expected_calls', [(4, 0, 2), (4, 1, 1), (1, 0, 1)])
def test_each_proposal_gets_its_own_likelihood(tmp_path, block, visits, expected_calls):
    ns, calls, receipts = run_branch(tmp_path, block=block, visits=visits)
    assert len(calls) == expected_calls
    assert calls[-1][0] == 'proposal'
    np.testing.assert_array_equal(calls[-1][1], [True, False])
    np.testing.assert_array_equal(ns['newlike'], [4., -np.inf])
    assert not receipts
    if expected_calls == 2:
        assert calls[0][0] == 'current'
        assert ns['invariant_audit']['checked_states'] == 2


def test_same_state_disagreement_remains_fatal(tmp_path):
    with pytest.raises(ValueError, match='retained state disagrees'):
        run_branch(tmp_path, replay_error=2e-6)


def test_transform_preserves_rng_and_surrounding_code():
    source = """prefix = 1
        invariant_work=('z/SED moves reuse the likelihood, environment moves the target prior; '
                        'every 1024th such move is replayed in full'),
    invariant_audit = dict(likelihood_calls_skipped=0,
                           checked_proposals=0, maximum_summary_change=0.)
        if block in invariant_visits:
            old_cache_branch()
        accepted = metropolis_accept(rng.uniform())
suffix = 2
"""
    result = cacheoff.transform(source)
    assert result.startswith('prefix = 1\n')
    assert result.endswith('        accepted = metropolis_accept(rng.uniform())\nsuffix = 2\n')
    assert result.count('rng.') == source.count('rng.')
    assert 'old_cache_branch' not in result
    assert 'nuisance_likelihood_cache=False' in result
    assert 'checked_proposals' not in result
