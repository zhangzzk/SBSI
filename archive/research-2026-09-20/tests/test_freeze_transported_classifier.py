from copy import deepcopy

import pytest

from scripts.freeze_transported_classifier import validate_training_receipt


def fixture():
    protocol = {'geometry_coordinate_convention': 'state_major_axis', 'training_cases': list(range(40, 100)),
        'tune_cases': list(range(100, 120)), 'seed': 20260913, 'max_epochs': 100, 'patience': 15,
        'response_read': False, 'selection': 'minimum mean-case tune BCE'}
    history = {'history': [{'epoch': i, 'tune_mean_case_bce': 1/i} for i in range(1, 101)],
               'best_epoch': 100, 'best_tune_mean_case_bce': .01}
    completed = {'smoke': False, 'response_read': False, 'production_accepted': False,
                 'best_epoch': 100, 'best_tune_mean_case_bce': .01}
    return protocol, history, completed


def test_complete_minimum_loss_receipt_and_early_stop():
    protocol, history, completed = fixture()
    validate_training_receipt(protocol, history, completed)
    history['history'] = [{'epoch': i, 'tune_mean_case_bce': .01 if i == 1 else .02} for i in range(1, 17)]
    history['best_epoch'] = completed['best_epoch'] = 1
    validate_training_receipt(protocol, history, completed)
    history['history'].pop()
    with pytest.raises(ValueError, match='stopping rule'):
        validate_training_receipt(protocol, history, completed)


def test_reject_alternate_checkpoint_or_response_selection():
    protocol, history, completed = fixture()
    changed = deepcopy(completed)
    changed['best_epoch'] = 99
    with pytest.raises(ValueError, match='minimum-tune-loss'):
        validate_training_receipt(protocol, history, changed)
    protocol['response_read'] = True
    with pytest.raises(ValueError, match='fixed coordinate'):
        validate_training_receipt(protocol, history, completed)
