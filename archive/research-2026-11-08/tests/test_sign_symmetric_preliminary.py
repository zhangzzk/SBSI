"""The preliminary evaluation must reject incomplete or mismatched selection artifacts."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location('preliminary',
    Path(__file__).parents[1]/'scripts/run_sign_symmetric_preliminary.py')
study = importlib.util.module_from_spec(spec)
spec.loader.exec_module(study)


@pytest.fixture
def prefix(tmp_path):
    (tmp_path/'normalization').mkdir()
    rule = dict(model_identity={'flow_sha256': 'frozen-test-flow'},
                source_plan_sha256='frozen-test-plan',
                shear_nodes=[[0, 0], [-.002, 0], [.002, 0], [0, -.002], [0, .002]])
    study.save(tmp_path/'rule.json', rule)
    for i in range(2):
        path = tmp_path/f'normalization/selection{i}.npz'
        values = np.full((study.CHUNKS, 5, 1, 2), .1+.1*i)
        values[:, :, :, 0] = .8  # Eight-draw diagnostic must not enter the estimate.
        np.savez(path, values=values, grid=rule['shear_nodes'])
        study.save(tmp_path/f'normalization/selection{i}.json', dict(
            model_identity=rule['model_identity'], source_plan_sha256=rule['source_plan_sha256'],
            chunks=study.CHUNKS, scenes_per_chunk=8192, flow_draws=16, replica=i,
            seed_base=2027207001+10000*i, data_sha256=study.digest(path)))
    return tmp_path


def test_exact_fixed_prefix_uses_full_draws(prefix):
    result = study.load_selection(prefix)
    assert result.shape == (1222, 5)
    assert np.all(result[:611] == .1) and np.all(result[611:] == .2)
    assert 2*study.CHUNKS*study.SCENES == 10010624


@pytest.mark.parametrize('key,value', [('chunks', 610), ('flow_draws', 8),
    ('model_identity', {'flow_sha256': 'wrong'}), ('seed_base', 123),
    ('source_plan_sha256', 'changed')])
def test_rejects_wrong_receipt(prefix, key, value):
    p = prefix/'normalization/selection0.json'
    receipt = json.loads(p.read_text()); receipt[key] = value; study.save(p, receipt)
    with pytest.raises(ValueError, match='receipt'):
        study.load_selection(prefix)


def test_rejects_corrupt_archive(prefix):
    p = prefix/'normalization/selection0.npz'
    p.write_bytes(p.read_bytes()+b'changed')
    with pytest.raises(ValueError, match='checksum'):
        study.load_selection(prefix)


@pytest.mark.parametrize('change', ['incomplete', 'nodes', 'nonfinite'])
def test_rejects_invalid_archive_even_with_matching_hash(prefix, change):
    p = prefix/'normalization/selection0.npz'
    with np.load(p) as data:
        values, grid = data['values'].copy(), data['grid'].copy()
    if change == 'incomplete':
        values = values[:-1]
    elif change == 'nodes':
        grid[0, 0] = .01
    else:
        values[0, 0, 0, 0] = np.nan
    np.savez(p, values=values, grid=grid)
    receipt_path = prefix/'normalization/selection0.json'
    receipt = json.loads(receipt_path.read_text()); receipt['data_sha256'] = study.digest(p)
    study.save(receipt_path, receipt)
    with pytest.raises(ValueError):
        study.load_selection(prefix)
