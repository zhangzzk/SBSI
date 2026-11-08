"""Bind a predeclared validation template to a completed fixed training run.

This verifies the existing minimum-tune-loss choice; it does not select a
checkpoint using downstream response or density results.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from scripts.train_full_parent_classifier import sha256, write_json


def validate_training_receipt(protocol, history, completed):
    if (protocol.get('geometry_coordinate_convention') != 'state_major_axis'
            or protocol.get('training_cases') != list(range(40, 100))
            or protocol.get('tune_cases') != list(range(100, 120))
            or protocol.get('seed') != 20260913 or protocol.get('max_epochs') != 100
            or protocol.get('patience') != 15 or protocol.get('response_read') is not False
            or protocol.get('selection') != 'minimum mean-case tune BCE'
            or completed.get('smoke') is not False or completed.get('response_read') is not False
            or completed.get('production_accepted') is not False):
        raise ValueError('completed fixed coordinate training contract required')
    rows = history['history']
    if not rows or [r['epoch'] for r in rows] != list(range(1, len(rows)+1)):
        raise ValueError('complete sequential training history required')
    scores = np.array([r['tune_mean_case_bce'] for r in rows], float)
    if not np.isfinite(scores).all():
        raise ValueError('finite tuning losses required')
    best = rows[int(np.argmin(scores))]
    if (history['best_epoch'] != best['epoch'] or completed['best_epoch'] != best['epoch']
            or history['best_tune_mean_case_bce'] != best['tune_mean_case_bce']
            or completed['best_tune_mean_case_bce'] != best['tune_mean_case_bce']):
        raise ValueError('checkpoint does not follow the frozen minimum-tune-loss rule')
    if len(rows) != 100 and len(rows)-best['epoch'] < 15:
        raise ValueError('training has not reached its declared stopping rule')
    if len(rows) > 100:
        raise ValueError('training exceeds declared epoch budget')


def freeze(template_path, output_path):
    template_path, output_path = Path(template_path), Path(output_path)
    if output_path.exists():
        raise FileExistsError(output_path)
    template = json.loads(template_path.read_text())
    for path, expected in template['files'].items():
        if sha256(Path(path)) != expected:
            raise ValueError('declared validation source changed: '+path)
    root = Path(template['training_root'])
    protocol, history, completed = (json.loads((root/name).read_text()) for name in ('protocol.json', 'history.json', 'completed.json'))
    validate_training_receipt(protocol, history, completed)
    if (completed['protocol_sha256'] != sha256(root/'protocol.json')
            or completed['selected_sha256'] != sha256(root/'selected.pt')):
        raise ValueError('completed training checkpoint/protocol receipt mismatch')
    for path, expected in protocol['pins'].items():
        if sha256(Path(path)) != expected:
            raise ValueError('training source/data changed: '+path)
    selected = str(root/'selected.pt')
    if 'transported' in template and template['transported'] != selected:
        raise ValueError('audit checkpoint differs from declared training run')
    for endpoint in template.get('endpoints', []):
        if 'arguments' in endpoint:
            arguments = endpoint['arguments']
            if arguments[arguments.index('--classifier-transported')+1] != selected:
                raise ValueError('joint checkpoint differs from declared training run')
    files = dict(template['files'])
    for reference in template.get('deferred_references', []):
        result, receipt, parent_protocol = (Path(reference[key]).resolve() for key in ('result', 'receipt', 'protocol'))
        if str(parent_protocol) not in files:
            raise ValueError('reference-stage protocol must be pinned before execution')
        recorded = json.loads(receipt.read_text())
        if (recorded['result_sha256'] != sha256(result)
                or recorded['protocol_sha256'] != files[str(parent_protocol)]):
            raise ValueError('completed reference-stage receipt mismatch')
        files[str(result)], files[str(receipt)] = sha256(result), sha256(receipt)
    for path in [template_path, *(root/name for name in ('protocol.json', 'history.json', 'completed.json', 'selected.pt'))]:
        files[str(path.resolve())] = sha256(path)
    write_json(output_path, template | {'files': files, 'training_receipt_verified': True,
                                       'selected_epoch': completed['best_epoch'], 'selected_sha256': completed['selected_sha256']})
    return output_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--template', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print('TRANSPORTED_CLASSIFIER_FROZEN', freeze(args.template, args.output), flush=True)


if __name__ == '__main__':
    main()
