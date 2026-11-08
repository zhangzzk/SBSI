#!/usr/bin/env bash
# Development-only numerical-support audit for the direct selected-event response.

#SBATCH --job-name=sbsi_sel_support
#SBATCH --partition=cip
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --array=0-19%10
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/minimal_selected_response_support_v3/logs/%A_%a_support.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi_caches/minimal_selected_response_support_v3/logs/%A_%a_support.err

set -euo pipefail

repo=/home/z/Zekang.Zhang/SBSI
root=/project/ls-gruen/users/zekang.zhang/sbsi_caches/minimal_selected_response_support_v3
selected=/project/ls-gruen/users/zekang.zhang/sbsi_caches/minimal_selected_event_head_v2
python=/project/ls-gruen/users/zekang.zhang/sbsi_caches/plain_complete_flow_response_re037_v1/physical_circularized_grid_free_shape_radius_lambda10_v2/constgold50_v1/env/bin/python
source_root=/project/ls-gruen/users/zekang.zhang/sbsi_caches/plain_complete_flow_response_re037_v1/physical_circularized_grid_free_shape_radius_lambda10_v2/constgold50_v1/source

[[ -s "$selected/development_result.json" ]] || { echo "missing selected-event gate" >&2; exit 2; }
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export PYTHONPATH="$source_root:$repo"
cd /tmp
"$python" -u "$repo/scripts/audit_minimal_selected_response_support.py" case \
  --output-root "$root" \
  --index "$SLURM_ARRAY_TASK_ID" \
  --batch-size 2048
