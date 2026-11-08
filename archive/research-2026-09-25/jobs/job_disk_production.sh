#!/bin/bash
#SBATCH --job-name=sbsi-v36-500k
#SBATCH --partition=inter,cip
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --array=0-19%4
#SBATCH --output=logs/v36_production_%A_%a.out
set -euo pipefail
cd /home/z/Zekang.Zhang/SBSI
export SBSI_CACHE_DIR=/project/ls-gruen/users/zekang.zhang/sbsi_caches
export BLENDEMU_RUNS_DIR=/project/ls-gruen/users/zekang.zhang/blendemu_runs
export PYTHONPATH=/home/z/Zekang.Zhang/SBSI${PYTHONPATH:+:$PYTHONPATH}
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
PYTHON=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
RUN_ROOT=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/inference_constgold_v36_uncut_size060_n500k_m16384_20260920_v1
GATE_ROOT="$RUN_ROOT/fullprior_lru_pilot_16606079"
"$PYTHON" - "$GATE_ROOT" <<'PY'
import json
import sys
from pathlib import Path
from sbsi.disk_inference_store import file_hash
root = Path(sys.argv[1])
gate = json.loads((root / 'replay_validation.json').read_text())
report = json.loads((root / 'result.json').read_text())
assert gate['status'] == 'bitwise_identical'
assert file_hash(root / 'result.json') == gate['replay_result_sha256']
assert report['pipeline_config']['prior']['n_atoms'] == 24000000
assert report['observation_partition']['n_total'] == 500000
assert report['config']['adaptive_draw_ladder'][-1] == 16384
assert report['config']['compile_flow'] is True
for name, expected in report['implementation_sha256'].items():
    assert file_hash(name) == expected, f'implementation changed since validated pilot: {name}'
print('PRODUCTION_GATE_PASSED', flush=True)
PY
"$PYTHON" /home/z/Zekang.Zhang/.agents/skills/get-available-resources/scripts/detect_resources.py \
  --output "logs/v36_production_resources_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.json"
printf -v PART '%02d' "$SLURM_ARRAY_TASK_ID"
START=$((SLURM_ARRAY_TASK_ID * 25000))
"$PYTHON" scripts/run_disk_inference.py run --subset "$RUN_ROOT/prior_subset24m/manifest.json" \
  --input "$RUN_ROOT/input" --prepared "$RUN_ROOT/disk_assembled_v1" \
  --start "$START" --count 25000 --object-chunk 16 --compile-flow \
  --output "$RUN_ROOT/production_lru_v1/part_$PART"
