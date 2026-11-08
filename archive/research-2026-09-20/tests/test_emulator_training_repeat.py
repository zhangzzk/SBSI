from copy import deepcopy

import pytest

from scripts.validate_emulator_training_repeat import require_training_repeat


def test_emulator_recipe_is_fixed_even_when_seed_changes():
    original = {'params': {'seed': 7, 'subsample': .76}, 'trees': 218,
                'prepared_manifest_sha256': 'data', 'script_sha256': 'old', 'model_sha256': 'old-model'}
    repeat = original | {'params': {'seed': 8, 'subsample': .76},
                         'script_sha256': 'new', 'model_sha256': 'new-model'}
    require_training_repeat(original, repeat, 8, 'new')
    for key, value in [('trees', 219), ('prepared_manifest_sha256', 'other-data'),
                       ('params', {'seed': 8, 'subsample': .8})]:
        changed = deepcopy(repeat)
        changed[key] = value
        with pytest.raises(ValueError, match='only its training seed'):
            require_training_repeat(original, changed, 8, 'new')
