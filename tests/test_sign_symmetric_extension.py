"""Check immutable-prefix reuse and exact coverage for the 400k extension."""
import importlib.util
from pathlib import Path

import pandas as pd
import pytest

spec = importlib.util.spec_from_file_location('extension',
    Path(__file__).parents[1]/'scripts/extend_sign_symmetric_inference.py')
extension = importlib.util.module_from_spec(spec)
spec.loader.exec_module(extension)


@pytest.mark.parametrize('n,size', [(0, 640), (1, 640), (640, 640), (641, 640), (400000, 1024)])
def test_units_cover_rows_once_with_bounded_sizes(n, size):
    covered = []
    for line in extension.units('bright', 'ids.txt', n, size):
        label, file, first, last = line.split()
        first, last = int(first), int(last)
        assert label.startswith('bright_add') and file == 'ids.txt'
        assert 0 < last-first+1 <= size
        covered.extend(range(first, last+1))
    assert covered == list(range(n))


@pytest.fixture
def frames():
    obs = pd.DataFrame({'e1': [.1, .2, .3]})
    truth = pd.DataFrame({'source_case': [40, 40, 41], 'source_input_index': [1, 2, 3],
        'source_detection_id': [11, 12, 13], 'parent_sample_row': [0, 1, 100000]})
    return obs, truth, obs.iloc[:2].copy(), truth.iloc[:2].copy()


def test_accepts_exact_prefix(frames):
    extension.validate_prefix(*frames, 2)


def test_rejects_changed_measurement_prefix(frames):
    frames[0].loc[0, 'e1'] += 1.e-12
    with pytest.raises(AssertionError):
        extension.validate_prefix(*frames, 2)


@pytest.mark.parametrize('column', ['source_input_index', 'source_detection_id'])
def test_rejects_duplicate_catalogue_identity(frames, column):
    frames[1].loc[2, 'source_case'] = 40
    frames[1].loc[2, column] = frames[1].loc[0, column]
    with pytest.raises(ValueError, match='duplicate'):
        extension.validate_prefix(*frames, 2)


def test_rejects_extension_from_old_raw_prefix(frames):
    frames[1].loc[2, 'parent_sample_row'] = 99999
    with pytest.raises(ValueError, match='overlaps'):
        extension.validate_prefix(*frames, 2)


def test_accepts_later_extension_boundary(frames):
    frames[1].loc[2, 'parent_sample_row'] = 500000
    extension.validate_prefix(*frames, 2, previous_raw=500000)
    frames[1].loc[2, 'parent_sample_row'] = 499999
    with pytest.raises(ValueError, match='overlaps'):
        extension.validate_prefix(*frames, 2, previous_raw=500000)


def test_new_unit_names_do_not_collide_with_previous_extension():
    previous = extension.units('bright', 'old.txt', 1300, 640)
    new = extension.units('bright_ext523', 'new.txt', 1300, 640)
    assert set(x.split()[0] for x in previous).isdisjoint(x.split()[0] for x in new)


def test_disjoint_sample_replays_original_and_preserves_random_order(monkeypatch):
    import sys
    import numpy as np
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1]/'scripts'))
    from freeze_constgold_extension import extension_indices
    old, new = extension_indices(1000, 200, 42, 300, 20261028)
    np.testing.assert_array_equal(old, np.random.default_rng(42).choice(1000, 200, replace=False))
    assert len(new) == len(set(new)) == 300
    assert set(old).isdisjoint(new)
    np.testing.assert_array_equal(new, extension_indices(1000, 200, 42, 300, 20261028)[1])
    assert not np.array_equal(new, np.sort(new))
