#!/bin/bash
#SBATCH --job-name=v22calcurve
#SBATCH --time=00:45:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/%x_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/%x_%j.err
set -euo pipefail

ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
ENV=/project/ls-gruen/users/zekang.zhang/envs/py31
CAT=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876/response_catalogue_train.feather
REFERENCE=$ROOT/results/v22_halfshear_rblend_vs_secondary_size_training_c40-199.json
PREFIX=$ROOT/results/v22_halfshear_rblend_emulator_label_calibration_training_c40-199
export PATH="$ENV/bin:$PATH"
export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT:/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
for input in "$CAT" "$REFERENCE"; do
  test -s "$input" || { echo "MISSING $input"; exit 1; }
done
for suffix in json md csv png pdf; do
  test ! -e "$PREFIX.$suffix"
done
cd "$ROOT"

python -u scripts/plot_v22_emulator_label_calibration.py \
  --catalogue "$CAT" \
  --reference "$REFERENCE" \
  --tag lsst_r_extnbr_v22 \
  --case-min 40 --case-max 199 --shear 0.2 --n-bins 20 \
  --output-prefix "$PREFIX"
echo V22_EMULATOR_LABEL_CALIBRATION_JOB_DONE
date
