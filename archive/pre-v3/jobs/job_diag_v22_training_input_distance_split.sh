#!/bin/bash
#SBATCH --job-name=v22trid23
#SBATCH --time=01:00:00
#SBATCH --mem=40G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/v22trid23_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/v22trid23_%j.err
set -euo pipefail

ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
PY=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
OUT=$ROOT/results/v22_summed_label_input_distance_split_c40-199.json
test ! -e "$OUT" || { echo "REFUSING existing $OUT"; exit 1; }
export PYTHONPATH="$ROOT:/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
cd "$ROOT"
"$PY" -u scripts/diag_v22_summed_label_closure.py \
  --catalogue /project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876/response_catalogue_train.feather \
  --tag lsst_r_extnbr_v22 --case-min 40 --case-max 199 --shear 0.2 \
  --v21-domain --split-distance 2 3 --split-coordinate input --output "$OUT"
echo V22_TRAINING_INPUT_DISTANCE_SPLIT_JOB_DONE
