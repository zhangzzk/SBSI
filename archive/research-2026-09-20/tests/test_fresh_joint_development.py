import copy

import numpy as np
import pandas as pd
import pytest

from scripts.prepare_fresh_joint_development import common_rendered_parent, validate_receipts


def test_unrendered_drops_reported_without_using_measured_validity():
    parent = pd.DataFrame({'index': [4, 6, 8], 'r': [24., 25., 23.]})
    selected, counts = common_rendered_parent(parent, {'zero': [0, 4, 6, 8], 'plus': [0, 4, 6], 'minus': [0, 4, 8]})
    np.testing.assert_array_equal(selected['index'], [4])
    assert counts['additional_unrendered_parent_ids'] == [6, 8]
    assert counts['missing_parent_ids_by_leg'] == {'zero': [], 'plus': [8], 'minus': [6]}
    with pytest.raises(ValueError, match='unique rendered'):
        common_rendered_parent(parent, {'zero': [4, 4]})


def receipts():
    leg = {'group': 'g0', 'h': 0., 'sky_shear': [0., 0.]}
    legs = [leg]+[{'group': f'g{axis}_h0.02', 'h': h, 'sky_shear': [h, 0.] if axis == 1 else [0., h]} for axis in (1, 2) for h in (-.02, .02)]
    m = {'status': 'render_commands_completed', 'case': 1, 'legs': legs, 'area_deg2': 1.,
         'scope': 'full_scene_development', 'reference_seed': 123,
         'scene_count_rule': 'exact original generated count after prior replay'}
    a = {'status': 'passed', 'case': 1, 'legs': legs, 'manifest_sha256': 'manifest'}
    fitted = {'status': 'completed', 'case': 1, 'legs': legs, 'render_audit_sha256': 'audit'}
    return copy.deepcopy((m, a, fitted))


def test_requires_matching_complete_full_area_provenance():
    m, a, fitted = receipts()
    validate_receipts(m, a, fitted, 'manifest', 'audit', small_smoke=False)
    m['reference_seed'] = 456
    with pytest.raises(ValueError, match='original-prior'):
        validate_receipts(m, a, fitted, 'manifest', 'audit', small_smoke=False)
    m, a, fitted = receipts()
    fitted['render_audit_sha256'] = 'changed'
    with pytest.raises(ValueError, match='receipts'):
        validate_receipts(m, a, fitted, 'manifest', 'audit', small_smoke=False)


def test_small_scene_requires_explicit_operational_scope():
    m, a, fitted = receipts()
    m['area_deg2'] = .01
    validate_receipts(m, a, fitted, 'manifest', 'audit', small_smoke=True)
    with pytest.raises(ValueError, match='full-area'):
        validate_receipts(m, a, fitted, 'manifest', 'audit', small_smoke=False)
