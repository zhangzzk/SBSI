#!/usr/bin/env bash
#SBATCH --job-name=sbsi_nbrsel_summary
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:05:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/neighbour_selection_audit_20260917_v1/logs/%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi_caches/neighbour_selection_audit_20260917_v1/logs/%x_%j.err
set -euo pipefail
root=/project/ls-gruen/users/zekang.zhang/sbsi_caches/neighbour_selection_audit_20260917_v1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /home/z/Zekang.Zhang/SBSI
/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python -u -m scripts.audit_neighbour_selection_response summarize \
  --parts "$root/parts" --output "$root/result.json"
