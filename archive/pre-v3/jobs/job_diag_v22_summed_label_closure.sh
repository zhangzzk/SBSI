#!/bin/bash
#SBATCH --job-name=v22_sumlabel
#SBATCH --time=03:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=16
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/v22_sumlabel_%j.out
set -euo pipefail

ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
PY=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
CAT=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876/response_catalogue_train.feather
OUT=${OUT:-results/v22_summed_label_closure_c0-39.json}
EXTRA_ARGS=()
if [ "${V21_DOMAIN:-0}" = 1 ]; then
  EXTRA_ARGS+=(--v21-domain)
fi
export PYTHONPATH="$ROOT:/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
cd "$ROOT"

[ ! -e "$OUT" ] || { echo "REFUSING existing $OUT"; exit 1; }
"$PY" -u scripts/diag_v22_summed_label_closure.py \
  --catalogue "$CAT" --tag lsst_r_extnbr_v22 --heldout-min 40 \
  --shear 0.2 "${EXTRA_ARGS[@]}" --output "$OUT"

echo V22_SUMMED_LABEL_CLOSURE_JOB_DONE; date
