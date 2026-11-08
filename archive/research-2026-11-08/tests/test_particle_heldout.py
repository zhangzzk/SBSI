"""The frozen held-out evaluator must not accept missing or failed groups."""
import copy
import importlib.util
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location('heldout_eval', Path(__file__).resolve().parents[1]/
    'doc/evidence/run_particle_heldout_20261022.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def groups():
    return [dict(group=i, candidate_precision=True, reference_precision=True, comparison=True,
                 candidate_all_inclusive_seconds=2., reference_all_inclusive_seconds=5.) for i in range(5)]


def test_complete_accuracy_and_cost_are_both_required():
    rows = groups()
    assert module.aggregate_gate(rows)['accepted_sampler'] is True
    for flag in ('candidate_precision', 'reference_precision', 'comparison'):
        changed = copy.deepcopy(rows)
        changed[4][flag] = False
        result = module.aggregate_gate(changed)
        assert result['heldout_cost_gate'] is True
        assert result['accepted_sampler'] is False


def test_missing_duplicate_or_reordered_groups_rejected():
    rows = groups()
    for invalid in (rows[:-1], rows[:4]+[rows[0]], rows[::-1]):
        with pytest.raises(ValueError, match='all five'):
            module.aggregate_gate(invalid)


def test_speed_is_end_to_end_and_strict():
    rows = groups()
    for row in rows:
        row['candidate_all_inclusive_seconds'] = 5.
    assert module.aggregate_gate(rows)['heldout_accuracy_gate'] is True
    assert module.aggregate_gate(rows)['accepted_sampler'] is False
    rows[0]['candidate_all_inclusive_seconds'] = float('nan')
    with pytest.raises(ValueError, match='finite positive'):
        module.aggregate_gate(rows)
