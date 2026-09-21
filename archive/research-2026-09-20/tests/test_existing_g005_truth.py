import numpy as np
import pandas as pd
import pytest

from scripts.audit_existing_g005_truth import generated_scene


def test_scene_fingerprint_ignores_row_order_and_shear_but_detects_new_positions(tmp_path):
    left, right = tmp_path/'left.feather', tmp_path/'right.feather'
    frame = pd.DataFrame({'index': [0, 1], 'RA': [180.1, 180.2], 'DEC': [.1, .2], 'g1': [.05, .05]})
    frame.to_feather(left)
    frame.assign(g1=-.05).iloc[::-1].reset_index(drop=True).to_feather(right)
    assert generated_scene(left) == generated_scene(right)
    frame.assign(RA=[180.2, 180.3]).to_feather(right)
    assert generated_scene(left) != generated_scene(right)


@pytest.mark.parametrize('column,values', [('index', [0, 0]), ('RA', [180., np.nan])])
def test_scene_fingerprint_rejects_ambiguous_truth(tmp_path, column, values):
    path = tmp_path/'scene.feather'
    frame = pd.DataFrame({'index': [0, 1], 'RA': [180.1, 180.2], 'DEC': [.1, .2]})
    frame[column] = values
    frame.to_feather(path)
    with pytest.raises(ValueError, match='invalid generated scene'):
        generated_scene(path)
