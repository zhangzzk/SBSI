#!/usr/bin/env bash
#SBATCH --job-name=sbsi_nbrsel_audit
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:20:00
#SBATCH --array=0-9%4
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/neighbour_selection_audit_20260917_v1/logs/%x_%A_%a.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi_caches/neighbour_selection_audit_20260917_v1/logs/%x_%A_%a.err
set -euo pipefail
root=/project/ls-gruen/users/zekang.zhang/sbsi_caches/neighbour_selection_audit_20260917_v1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /home/z/Zekang.Zhang/SBSI
first=$((400 + 50 * SLURM_ARRAY_TASK_ID))
/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python -u -m scripts.audit_neighbour_selection_response compute \
  --case-start "$first" --case-stop "$((first + 50))" --output "$root/parts/block${SLURM_ARRAY_TASK_ID}.json"
