"""Bounded device-equivalence diagnostic on fixed disk-emulator development rows."""
import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
import xgboost as xgb

from scripts.train_full_parent_classifier import sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or args.output.exists():
        raise RuntimeError('Slurm allocation and new output required')
    manifest_path = args.root/'prepared/manifest.json'
    manifest = json.loads(manifest_path.read_text())
    model_path = args.root/'model/model.json'
    metadata = json.loads((model_path.parent/'metadata.json').read_text())
    model_hash = sha256(model_path)
    if model_hash != metadata['model_sha256']:
        raise ValueError('model hash mismatch')
    model = xgb.Booster(model_file=model_path)
    records = []
    for part in manifest['parts'][:3]:
        path = Path(part['source_path'])
        if sha256(path) != part['sha256']:
            raise ValueError('prepared part hash mismatch')
        with np.load(path) as data:
            x = np.ascontiguousarray(data['x'])
        model.set_param({'device': 'cpu', 'nthread': 1})
        started = time.monotonic()
        cpu = model.inplace_predict(x)
        cpu_seconds = time.monotonic()-started
        model.set_param({'device': 'cuda', 'nthread': 1})
        started = time.monotonic()
        gpu = model.predict(xgb.DMatrix(x, feature_names=model.feature_names, nthread=1))
        gpu_seconds = time.monotonic()-started
        record = {'source': str(path), 'sha256': part['sha256'], 'rows': len(x),
                  'cpu_seconds': cpu_seconds, 'gpu_seconds': gpu_seconds,
                  'exactly_equal': bool(np.array_equal(cpu, gpu)),
                  'different_rows': int(np.count_nonzero(cpu != gpu)),
                  'maximum_raw_prediction_difference': float(np.max(np.abs(cpu-gpu)))}
        records.append(record)
        print('DISK_DEVICE_BENCHMARK', record, flush=True)
    if sha256(model_path) != model_hash:
        raise ValueError('model changed')
    write_json(args.output, {'records': records, 'model_sha256': model_hash,
        'manifest_sha256': sha256(manifest_path), 'source_sha256': sha256(Path(__file__)),
        'interpretation': 'finite development-row device check only; not calibration or full evaluator equivalence',
        'calibration_claim': False})


if __name__ == '__main__':
    main()
