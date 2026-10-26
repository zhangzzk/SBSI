import ast
from pathlib import Path

import pytest

from scripts.freeze_smooth_classifier_replica import validate_training_receipt


def test_replica_changes_only_training_seed():
    original = Path('scripts/train_smooth_classifier.py').read_text()
    replica = Path('scripts/train_smooth_classifier_replica.py').read_text()
    assert ast.dump(ast.parse(replica)) == ast.dump(ast.parse(original.replace('SEED = 20260913', 'SEED = 20260914')))


def test_replica_receipt_requires_declared_seed_and_probability_choice():
    protocol = {'geometry_coordinate_convention': 'state_major_axis', 'training_cases': list(range(40, 100)),
                'tune_cases': list(range(100, 120)), 'seed': 20260914, 'max_epochs': 100, 'patience': 15,
                'response_read': False, 'selection': 'minimum mean-case tune BCE'}
    history = {'history': [{'epoch': i, 'tune_mean_case_bce': .3-i*.001} for i in range(1, 101)],
               'best_epoch': 100, 'best_tune_mean_case_bce': .3-100*.001}
    completed = {'smoke': False, 'response_read': False, 'production_accepted': False,
                 'best_epoch': 100, 'best_tune_mean_case_bce': history['best_tune_mean_case_bce']}
    validate_training_receipt(protocol, history, completed)
    with pytest.raises(ValueError, match='training contract'):
        validate_training_receipt(protocol | {'seed': 20260913}, history, completed)
    with pytest.raises(ValueError, match='minimum-tune-loss'):
        validate_training_receipt(protocol, history, completed | {'best_epoch': 99})
