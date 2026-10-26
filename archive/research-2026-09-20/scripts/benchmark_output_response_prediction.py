"""Check exact frozen-tree CPU/GPU prediction agreement and execution cost."""
import json
import os
from pathlib import Path
import time

import numpy as np
import xgboost as xgb


def main():
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("prediction benchmark requires Slurm")
    root = Path('/project/ls-gruen/users/zekang.zhang/blendemu_runs/output_conditioned_response_20260917_v1')
    manifest = json.loads((root/'prepared/manifest.json').read_text())
    part = manifest['parts'][0]
    with np.load(root/'prepared'/part['path']) as data:
        x = np.tile(data['x'], (4, 1))
    model = xgb.Booster(model_file=root/'models_cpu/output_conditioned/model.json')
    model.set_param({'device': 'cpu', 'nthread': 8})
    started = time.monotonic()
    cpu = model.inplace_predict(x)
    cpu_seconds = time.monotonic()-started
    print(f'PREDICT_CPU rows={len(x)} elapsed={cpu_seconds}', flush=True)
    started = time.monotonic()
    matrix = xgb.DMatrix(x, feature_names=model.feature_names, nthread=8)
    matrix_seconds = time.monotonic()-started
    model.set_param({'device': 'cuda', 'nthread': 8})
    started = time.monotonic()
    gpu = model.predict(matrix)
    gpu_seconds = time.monotonic()-started
    np.testing.assert_allclose(gpu, cpu, rtol=1e-6, atol=1e-7)
    warm_seconds = []
    for _ in range(3):
        started = time.monotonic()
        fresh = xgb.DMatrix(x.copy(), feature_names=model.feature_names, nthread=8)
        warm = model.predict(fresh)
        warm_seconds.append(time.monotonic()-started)
        np.testing.assert_allclose(warm, cpu, rtol=1e-6, atol=1e-7)
    result = {'rows': len(x), 'cpu_seconds': cpu_seconds, 'matrix_seconds': matrix_seconds,
              'gpu_seconds': gpu_seconds, 'gpu_fresh_matrix_warm_seconds': warm_seconds,
              'maximum_prediction_difference': float(np.max(np.abs(gpu-cpu)))}
    output = Path('/project/ls-gruen/users/zekang.zhang/sbsi_caches/output_conditioned_response_20260917_v1/prediction_benchmark.json')
    output.write_text(json.dumps(result, indent=2)+'\n')
    print('PREDICTION_BENCHMARK_COMPLETE', result, flush=True)


if __name__ == '__main__':
    main()
