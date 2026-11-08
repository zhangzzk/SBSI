#!/usr/bin/env bash
# Reduce the frozen direct-selected existing-catalogue response.

#SBATCH --job-name=sbsi_sel_g005_review
#SBATCH --partition=inter
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:05:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/minimal_selected_existing_response_v1/logs/%j_response_review.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi_caches/minimal_selected_existing_response_v1/logs/%j_response_review.err

set -euo pipefail

repo=/home/z/Zekang.Zhang/SBSI
root=/project/ls-gruen/users/zekang.zhang/sbsi_caches/minimal_selected_existing_response_v1
python=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python

export PYTHONPATH="$repo"
cd /tmp
"$python" -u "$repo/scripts/summarize_minimal_selected_existing_response.py" review \
  --output-root "$root"

sha256sum "$root/response/result.json" "$root/response/decision.json"
