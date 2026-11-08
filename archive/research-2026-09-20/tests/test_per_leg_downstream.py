import copy

import pytest

from scripts import validate_per_leg_candidate as validation


def selection(step=1):
    mass = {'coordinates': ['soft_primary', 'hard_primary'], 'residual': [.003, .004]}
    record = {'smoke': False, 'best_step': step, 'eligible_for_joint_validation': step > 0,
              'selected_full_validation_nll': -4.768, 'baseline_full_validation_nll': -4.769,
              'selected_guard_loss': .2 if step else .4, 'baseline_guard_loss': .4,
              'selected_selection_mass': copy.deepcopy(mass), 'baseline_selection_mass': mass}
    return record, {'smoke': False, 'steps': 6, 'density_tolerance_nats_per_row': .001}


def test_gate_accepts_improvement_and_explicitly_skips_zero():
    assert validation.validate_selection(*selection())
    assert not validation.validate_selection(*selection(0))


@pytest.mark.parametrize('field,value', [('selected_full_validation_nll', -4.767),
                                         ('selected_guard_loss', .5),
                                         ('selected_guard_loss', float('nan')),
                                         ('eligible_for_joint_validation', False),
                                         ('smoke', True)])
def test_gate_rejects_failed_selection(field, value):
    record, protocol = selection()
    record[field] = value
    with pytest.raises(ValueError):
        validation.validate_selection(record, protocol)


def test_gate_checks_both_signs_of_probability_error():
    record, protocol = selection()
    record['selected_selection_mass']['residual'][1] = -.005
    with pytest.raises(ValueError, match='probability'):
        validation.validate_selection(record, protocol)


@pytest.mark.parametrize('endpoint,expected_cases,h', [('main', list(range(40, 120)), .02),
                                                     ('independent', list(range(200, 240)), .05)])
def test_existing_endpoint_keeps_same_model_and_extraction(monkeypatch, endpoint, expected_cases, h):
    calls, current = [], {}

    def run(module, *args):
        calls.append((module, list(args)))
        if module == 'scripts.evaluate_constgold_truth_parent':
            start, end = args.index('--cases')+1, args.index('--output')
            current.update(cases=list(args[start:end]), h=h, inputs={'flow_sha256': 'selected'},
                           predictive_diagnostics={})

    monkeypatch.setattr(validation, 'run', run)
    monkeypatch.setattr(validation, 'read', lambda path: current)
    monkeypatch.setattr(validation, 'digest', lambda path: 'selected')
    validation.existing(endpoint)
    evaluations = [args for module, args in calls if module == 'scripts.evaluate_constgold_truth_parent']
    assert len(evaluations) == 2
    for args, cases in zip(evaluations, (expected_cases[:2], expected_cases)):
        assert args[args.index('--cases')+1:args.index('--output')] == cases
        for flag, value in [('--flow', validation.CANDIDATE/'selected.pt'), ('--h', h),
                            ('--seed', 7301), ('--draws', 64), ('--cohorts', 'truth_parent'),
                            ('--response-composition', 'mobius')]:
            assert args[args.index(flag)+1] == value
    density_calls = [args for module, args in calls if module == 'scripts.audit_disk_candidate_density']
    assert len(density_calls) == (2 if endpoint == 'main' else 0)
    for args in density_calls:
        assert args[args.index('--shears')+1:args.index('--cases')] == [.02, -.02]
        assert args[args.index('--reference-flow')+1] == validation.INITIAL
