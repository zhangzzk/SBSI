import json

import pytest

from scripts.freeze_transported_classifier import freeze
from scripts.train_full_parent_classifier import sha256


@pytest.mark.parametrize('failure', [None, 'result', 'protocol'])
def test_deferred_reference_is_bound_only_to_its_pinned_stage(tmp_path, failure):
    train = tmp_path/'train'
    train.mkdir()
    protocol = {'geometry_coordinate_convention': 'state_major_axis', 'training_cases': list(range(40, 100)),
        'tune_cases': list(range(100, 120)), 'seed': 20260913, 'max_epochs': 100, 'patience': 15,
        'response_read': False, 'selection': 'minimum mean-case tune BCE', 'pins': {}}
    history = {'history': [{'epoch': i, 'tune_mean_case_bce': 1/i} for i in range(1, 101)],
               'best_epoch': 100, 'best_tune_mean_case_bce': .01}
    (train/'protocol.json').write_text(json.dumps(protocol))
    (train/'history.json').write_text(json.dumps(history))
    (train/'selected.pt').write_bytes(b'opaque checkpoint; deserialization belongs to model loader')
    completed = {'smoke': False, 'response_read': False, 'production_accepted': False,
                 'best_epoch': 100, 'best_tune_mean_case_bce': .01,
                 'protocol_sha256': sha256(train/'protocol.json'), 'selected_sha256': sha256(train/'selected.pt')}
    (train/'completed.json').write_text(json.dumps(completed))
    parent_protocol, result, receipt = [tmp_path/name for name in ('parent.json', 'result.json', 'receipt.json')]
    parent_protocol.write_text('{}')
    result.write_text('{"fixed_result": true}')
    hashes = {'result_sha256': sha256(result), 'protocol_sha256': sha256(parent_protocol)}
    if failure:
        hashes[failure+'_sha256'] = 'incorrect'
    receipt.write_text(json.dumps(hashes))
    template = tmp_path/'template.json'
    template.write_text(json.dumps({'training_root': str(train), 'files': {str(parent_protocol): sha256(parent_protocol)},
        'transported': str(train/'selected.pt'), 'deferred_references': [
            {'result': str(result), 'receipt': str(receipt), 'protocol': str(parent_protocol)}]}))
    output = tmp_path/'frozen.json'
    if failure:
        with pytest.raises(ValueError, match='reference-stage receipt'):
            freeze(template, output)
        assert not output.exists()
    else:
        freeze(template, output)
        frozen = json.loads(output.read_text())
        assert frozen['files'][str(result)] == sha256(result)
        assert frozen['files'][str(receipt)] == sha256(receipt)
        assert frozen['selected_sha256'] == sha256(train/'selected.pt')
