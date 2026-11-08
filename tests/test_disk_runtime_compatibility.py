import copy

import pytest

from _script_loader import load_script_module


driver = load_script_module('run_disk_inference.py')


def test_cache_reuse_requires_exact_scientific_and_implementation_identity():
    current = {'models': {'flow': 'pinned'}, 'implementation_sha256': driver.implementation()}
    assert driver.check_prepared_identity(current, copy.deepcopy(current)) == 'identical'
    for key in ('scripts/run_disk_inference.py', 'sbsi/catalogue_disk_likelihood.py',
                'sbsi/measurement_model.py', 'sbsi/catalogue_disk_cache.py', 'sbsi/catalogue_sampling.py'):
        changed = copy.deepcopy(current)
        changed['implementation_sha256'][key] = 'older-or-modified-code'
        with pytest.raises(ValueError, match='rebuild caches'):
            driver.check_prepared_identity(changed, current)
    changed = copy.deepcopy(current)
    changed['models']['flow'] = 'another-model'
    with pytest.raises(ValueError, match='rebuild caches'):
        driver.check_prepared_identity(changed, current)
