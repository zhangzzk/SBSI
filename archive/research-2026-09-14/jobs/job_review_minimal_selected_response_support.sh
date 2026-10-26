#!/usr/bin/env bash
# Reduce the response-blind support audit and freeze the smallest passing budget.

#SBATCH --job-name=sbsi_sel_support_review
#SBATCH --partition=inter
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:05:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/minimal_selected_response_support_v3/logs/%j_support_review.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi_caches/minimal_selected_response_support_v3/logs/%j_support_review.err

set -euo pipefail

repo=/home/z/Zekang.Zhang/SBSI
root=/project/ls-gruen/users/zekang.zhang/sbsi_caches/minimal_selected_response_support_v3
python=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
source_root=/project/ls-gruen/users/zekang.zhang/sbsi_caches/plain_complete_flow_response_re037_v1/physical_circularized_grid_free_shape_radius_lambda10_v2/constgold50_v1/source

export PYTHONPATH="$source_root:$repo"
cd /tmp
"$python" -u "$repo/scripts/audit_minimal_selected_response_support.py" review \
  --output-root "$root"
