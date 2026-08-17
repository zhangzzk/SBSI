#!/bin/bash
#SBATCH --job-name=hs_paircand
#SBATCH --time=01:00:00
#SBATCH --mem=40G
#SBATCH --cpus-per-task=4
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --array=0-2
#SBATCH --output=/home/z/Zekang.Zhang/logs/hs_paircand_%A_%a.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/hs_paircand_%A_%a.err
set -euo pipefail
ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
PY=/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python
CAT=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876/response_catalogue_train.feather
CAL=results/halfshear_pair_vector_calibration_v22_c0-39.json
TAGS=(lsst_r_extnbr_v22 lsst_r_extnbr_v22_rpowunwtg0 lsst_r_extnbr_v22_rpowposa0065)
TOKENS=(baseline gamma0 positive0065)
TAG=${TAGS[$SLURM_ARRAY_TASK_ID]}
TOKEN=${TOKENS[$SLURM_ARRAY_TASK_ID]}
export LD_LIBRARY_PATH=/project/ls-gruen/users/zekang.zhang/envs/sims1/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=/home/z/Zekang.Zhang/blendemu:$ROOT:${PYTHONPATH:-}
cd "$ROOT"
"$PY" -u scripts/audit_halfshear_pair_calibration_bins.py \
  --catalogue "$CAT" --calibration "$CAL" --shear 0.2 \
  --case-min 0 --n-cases 40 --development-max 19 \
  --model-tag "$TAG" --bin-model-tag lsst_r_extnbr_v22 \
  --output "results/halfshear_pair_bins_${TOKEN}_in_v22_bins_c0-39.json"
