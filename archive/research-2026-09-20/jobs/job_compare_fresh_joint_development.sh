#!/usr/bin/env bash
#SBATCH --job-name=sbsi_fresh_compare
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/fresh_fullscene_joint_20260918_v1/logs/%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi_caches/fresh_fullscene_joint_20260918_v1/logs/%x_%j.err
set -euo pipefail
python=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
root=/project/ls-gruen/users/zekang.zhang/sbsi_caches/fresh_fullscene_joint_20260918_v1/full
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /home/z/Zekang.Zhang/SBSI
mkdir -p "$root/comparisons"
for axis in 1 2; do
  for pair in initial:control initial:augmented control:augmented; do
    baseline=${pair%:*}
    candidate=${pair#*:}
    for kind in response density; do
      if [[ $kind == response ]]; then module=scripts.compare_truth_parent_response; else module=scripts.compare_joint_predictive_results; fi
      "$python" -u -m "$module" \
        --baseline "$root/results/${baseline}_g${axis}_h0.02.json" \
        --candidate "$root/results/${candidate}_g${axis}_h0.02.json" \
        --output "$root/comparisons/${baseline}_vs_${candidate}_g${axis}_${kind}.json"
    done
  done
done
"$python" - "$root" <<'PY'
from pathlib import Path
import json, sys
import numpy as np
from scripts.joint_response_axes import summarize_axis_branch, write_json
from scripts.train_fixed_g0_flow import sha256
root = Path(sys.argv[1])
output = root/'comparisons/case20010_amplitude_diagnostic.json'
if output.exists():
    raise FileExistsError(output)
results = {}
for arm in ('initial', 'control', 'augmented'):
    for axis in (1, 2):
        paths = [root/f'results/{arm}_g{axis}_h{h}.json' for h in ('0.02', '0.04')]
        low, high = [json.loads(p.read_text()) for p in paths]
        if (low['h'] != .02 or high['h'] != .04 or high['cases'] != [20010]
                or low['shear_axis'] != axis or high['shear_axis'] != axis
                or low['draws'] != high['draws'] or low['seed'] != high['seed']
                or low['inputs'] != high['inputs'] or low['population'] != high['population']
                or low['cohorts'] != high['cohorts'] or low['cuts'] != high['cuts']
                or high['operational_single_case'] is not True):
            raise ValueError('matching model/input/axis/latent amplitude pilot required')
        for field in ('metadata', 'composition', 'prediction_device', 'output_conditions'):
            if low['experimental_response'][field] != high['experimental_response'][field]:
                raise ValueError('response emulator or composition differs between amplitudes')
        if low['per_case_sufficient'].keys() != high['summary'].keys() or low['usability'] != high['usability']:
            raise ValueError('response branch definitions differ between amplitudes')
        index = low['cases'].index(20010)
        branches = {}
        for branch, raw in low['per_case_sufficient'].items():
            stats = {source: {leg: {field: np.asarray(value)[index:index+1] for field, value in entry.items()}
                              for leg, entry in by_leg.items()} for source, by_leg in raw.items()}
            point = summarize_axis_branch(stats, None, .02, axis)
            smaller = {field: np.asarray(point[field])[0] for field in ('measured_response', 'model_response', 'm_percent')}
            larger = {field: high['summary'][branch][field] for field in smaller}
            branches[branch] = {'h0.02': smaller, 'h0.04': larger,
                'h0.04_minus_h0.02': {field: np.asarray(larger[field])-smaller[field] for field in smaller},
                'case_uncertainty': None}
        results[f'{arm}/g{axis}'] = {'source_sha256': [sha256(p) for p in paths], 'branches': branches}
write_json(output, {'case': 20010, 'results': results,
    'limitations': ['one scene, no case-based confidence interval or acceptance claim',
                    'paired amplitudes retain identical model, parent, optimizer/noise and latent streams',
                    'finite-step difference is descriptive; no favorable-step selection or fitted response correction']})
print('FRESH_AMPLITUDE_DIAGNOSTIC_COMPLETE', output, flush=True)
PY
