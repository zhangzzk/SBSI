#!/usr/bin/env bash
#SBATCH --job-name=sbsi_disk_nll
#SBATCH --partition=inter,cip
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
set -euo pipefail
unset PYTHONPATH
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
: "${DISK_NLL_TRIAL_ROOT:?Set the source-frozen, installed disk trial root}"
cd "$DISK_NLL_TRIAL_ROOT/source"
audit_python="$DISK_NLL_TRIAL_ROOT/audit_env/bin/python"
train_python="$DISK_NLL_TRIAL_ROOT/train_env/bin/python"
"$audit_python" "$DISK_NLL_TRIAL_ROOT/verify_sources.py"
"$audit_python" /home/z/Zekang.Zhang/.agents/skills/get-available-resources/scripts/detect_resources.py \
  --output "$DISK_NLL_TRIAL_ROOT/resources.json"
# Completion artifacts, never a scheduler poll. The owner-facing hourly
# review must accept recovered data and the test gate before submitting.
"$audit_python" - "$DISK_NLL_TRIAL_ROOT" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
manifest = json.loads((root/'manifest.json').read_text())
gate = json.loads((root/'gate_result.json').read_text())
if not gate['passed'] or gate['manifest_sha256'] != __import__('hashlib').sha256((root/'manifest.json').read_bytes()).hexdigest():
    raise RuntimeError('matching successful full-suite gate required')
if not (pathlib.Path(manifest['raw_root'])/'results.json').is_file():
    raise RuntimeError('completed original float64 data recovery required')
PY
"$train_python" -u -m scripts.train_flow_disk_nll \
  --prepared /project/ls-gruen/users/zekang.zhang/sbsi_caches/plain_complete_flow_response_re037_v1/grid_covariance_normalized_v1/prepared \
  --raw-root /project/ls-gruen/users/zekang.zhang/sbsi_caches/plain_complete_flow_response_re037_v1/grid_covariance_disk_data_v1/output \
  --output "$DISK_NLL_TRIAL_ROOT/output" \
  --epochs 3 --batch-size 8192 --learning-rates 0.00001 0.0001 0.0007 \
  --probe-steps 64 --max-grad-norm 5 --seed 501
"$audit_python" "$DISK_NLL_TRIAL_ROOT/verify_sources.py"
