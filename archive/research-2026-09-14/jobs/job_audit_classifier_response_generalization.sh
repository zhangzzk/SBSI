#!/usr/bin/env bash
# Audit current classifier response error across existing train/tune/screen cases.

#SBATCH --job-name=sbsi_clf_resp_audit
#SBATCH --partition=inter
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/classifier_response_generalization_v1/logs/%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi_caches/classifier_response_generalization_v1/logs/%x_%j.err

set -euo pipefail

repo=/home/z/Zekang.Zhang/SBSI
root=/project/ls-gruen/users/zekang.zhang/sbsi_caches/classifier_response_generalization_v1
source_root=/project/ls-gruen/users/zekang.zhang/sbsi_caches/plain_complete_flow_response_re037_v1/physical_circularized_grid_free_shape_radius_lambda10_v2/constgold50_v1/source
python=/project/ls-gruen/users/zekang.zhang/sbsi_caches/plain_complete_flow_response_re037_v1/physical_circularized_grid_free_shape_radius_lambda10_v2/constgold50_v1/env/bin/python
script=$repo/archive/research-2026-09-14/scripts/audit_classifier_response_generalization.py
output=$root/results.json

mkdir -p "$root/logs"
[[ ! -e "$output" ]] || { echo "refusing to overwrite $output" >&2; exit 2; }
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export PYTHONPATH="$source_root:$repo${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH=/project/ls-gruen/users/zekang.zhang/envs/sims1/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
cd "$repo"
"$python" -u "$script" run --output "$output" --device cpu
sha256sum "$output"
