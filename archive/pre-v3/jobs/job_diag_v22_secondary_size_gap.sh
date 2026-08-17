#!/bin/bash
#SBATCH --job-name=v22secsize
#SBATCH --time=01:30:00
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
ANCHOR=$ROOT/results/anchorblend_g002_bias_features_v22_c400-899.feather
REFERENCE=$ROOT/results/v22_summed_label_closure_c40-199.json
TRAIN_PREFIX=$ROOT/results/v22_halfshear_rblend_vs_secondary_size_training_c40-199
ANCHOR_PREFIX=$ROOT/results/anchorblend_dominant_secondary_size_cut0p4_c400-899
export PATH="$ENV/bin:$PATH"
export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT:/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
for input in "$CAT" "$ANCHOR" "$REFERENCE"; do
  test -s "$input" || { echo "MISSING $input"; exit 1; }
done
cd "$ROOT"

python -u scripts/diag_v22_secondary_size_gap.py \
  --catalogue "$CAT" \
  --anchor-features "$ANCHOR" \
  --reference-closure "$REFERENCE" \
  --tag lsst_r_extnbr_v22 \
  --case-min 40 --case-max 199 --shear 0.2 --size-cut 0.4 \
  --training-output-prefix "$TRAIN_PREFIX" \
  --anchor-output-prefix "$ANCHOR_PREFIX"
echo V22_SECONDARY_SIZE_GAP_JOB_DONE
date
