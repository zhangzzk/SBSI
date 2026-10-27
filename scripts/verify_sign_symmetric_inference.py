#!/usr/bin/env python
"""Verify the frozen study, snapshot and model before starting any stage."""
import hashlib
import json
from pathlib import Path

ROOT = Path('@STUDY_ROOT@')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    frozen = json.loads((ROOT / 'frozen_inputs.json').read_text())
    for name, expected in frozen['files'].items():
        if digest(ROOT / name) != expected:
            raise ValueError(f'frozen study input changed: {name}')
    for name, expected in frozen['snapshot'].items():
        if digest(ROOT / 'source_snapshot' / name) != expected:
            raise ValueError(f'frozen source changed: {name}')
    identity = frozen['model_identity']
    if digest(identity['flow']) != identity['flow_sha256']:
        raise ValueError('symmetric checkpoint changed')
    print('SYMMETRIC_INFERENCE_INPUTS_VERIFIED', flush=True)


if __name__ == '__main__':
    main()
