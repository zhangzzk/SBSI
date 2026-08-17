#!/bin/bash
#SBATCH --job-name=v22_g005_sum
#SBATCH --time=03:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=16
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/v22_g005_sum_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/v22_g005_sum_%j.err
set -euo pipefail

ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
PY=/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python
CAT=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876/response_catalogue_g005_train.feather
BASE=results/v22_g005_summed_baseline_c0-39.json
CAND=results/v22_g005_summed_candidate_c0-39.json
OUT=results/v22_g005_summed_comparison_c0-39.json
export LD_LIBRARY_PATH="/project/ls-gruen/users/zekang.zhang/envs/sims1/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT:/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
cd "$ROOT"

for path in "$BASE" "$CAND" "$OUT"; do
  [ ! -e "$path" ] || { echo "REFUSING existing $path"; exit 1; }
done
"$PY" -u scripts/diag_v22_summed_label_closure.py \
  --catalogue "$CAT" --tag lsst_r_extnbr_v22 --heldout-min 40 \
  --shear 0.05 --output "$BASE"
"$PY" -u scripts/diag_v22_summed_label_closure.py \
  --catalogue "$CAT" --tag lsst_r_extnbr_v22_g005 --heldout-min 40 \
  --shear 0.05 --output "$CAND"
"$PY" -u scripts/compare_v22_g005_closure.py \
  --baseline "$BASE" --candidate "$CAND" --output "$OUT"
echo V22_G005_SUMMED_JOB_DONE
date
