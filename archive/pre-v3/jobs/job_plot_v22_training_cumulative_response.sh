#!/bin/bash
#SBATCH --job-name=v22trcum
#SBATCH --time=01:30:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/%x_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/%x_%j.err
set -euo pipefail

ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
PY=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
CAT=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876/response_catalogue_train.feather
REFERENCE=$ROOT/results/v22_summed_label_closure_c40-199.json
OUT=$ROOT/results/v22_training_cumulative_response_mag_fluxratio_c40-199
export PYTHONPATH="$ROOT:/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

for input in "$CAT" "$REFERENCE"; do
  test -s "$input" || { echo "MISSING $input"; exit 1; }
done
for suffix in csv json md pdf png; do
  test ! -e "$OUT.$suffix" || {
    echo "REFUSING existing $OUT.$suffix"; exit 1; }
done

cd "$ROOT"
"$PY" -u scripts/plot_v22_training_cumulative_response.py \
  --catalogue "$CAT" \
  --reference-closure "$REFERENCE" \
  --tag lsst_r_extnbr_v22 \
  --case-min 40 --case-max 199 --shear 0.2 \
  --score-chunk 1000000 \
  --output-prefix "$OUT"
echo V22_TRAINING_CUMULATIVE_RESPONSE_DONE
