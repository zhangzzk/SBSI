#!/usr/bin/env bash
# Two controlled weights share the same prepared pairs, covariance, and parent.
#SBATCH --job-name=sbsi_gridcov
#SBATCH --partition=inter,cip
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
set -euo pipefail
: "${GRID_TRIAL_ROOT:?Set the external trial root}"
: "${GRID_WEIGHT:?Set 1 or 10}"
grid_loss_reduction=${GRID_LOSS_REDUCTION:-sum}
case "$GRID_WEIGHT" in 1|10) ;; *) echo "Expected GRID_WEIGHT=1 or 10" >&2; exit 2 ;; esac
repo=/home/z/Zekang.Zhang/SBSI
base=/project/ls-gruen/users/zekang.zhang/sbsi_caches/plain_complete_flow_response_re037_v1
python=$base/distribution_response_v1/train_env/bin/python
audit_python=$base/distribution_response_v1/audit_env/bin/python
unset PYTHONPATH
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$repo"
"$audit_python" "$GRID_TRIAL_ROOT/verify_sources.py"
"$audit_python" /home/z/Zekang.Zhang/.agents/skills/get-available-resources/scripts/detect_resources.py \
  --output "$GRID_TRIAL_ROOT/lambda$GRID_WEIGHT/resources.json"
"$python" -u -m scripts.train_flow_grid_response train \
  --root "$GRID_TRIAL_ROOT/prepared" \
  --output "$GRID_TRIAL_ROOT/lambda$GRID_WEIGHT/measurement_flow_gridcov_s501.pt" \
  --weight "$GRID_WEIGHT" --grid-loss-reduction "$grid_loss_reduction" \
  --epochs 200 --patience 15 --batch-size 4096 \
  --cells-per-step 4 --pairs-per-cell 128 --draws 16 --validation-draws 16 \
  --validation-batch-size 1024 --validation-seed 7301 --seed 501 --lr 1e-5
"$audit_python" "$GRID_TRIAL_ROOT/verify_sources.py"
