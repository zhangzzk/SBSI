#!/bin/bash
#SBATCH --job-name=v22_sodist
#SBATCH --time=01:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/v22_sodist_%j.out
set -euo pipefail

ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
PY=/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python
export LD_LIBRARY_PATH="/project/ls-gruen/users/zekang.zhang/envs/sims1/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT:/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
cd "$ROOT"

OUT=results/v22_sameobject_distance_decomposition_c40-139_s16.json
[ ! -e "$OUT" ] || { echo "REFUSING overwrite $OUT"; exit 1; }

"$PY" -u scripts/diag_v22_sameobject_distance_decomposition.py \
  --constgold-features results/v22_constgold_gap_features_c40-139.feather \
  --half-selfresp /project/ls-gruen/users/zekang.zhang/sbsi_caches/hs_selfresp_v22_c40-199/halfshear_selfresp_v22_c40-199.feather \
  --seeds 501 502 503 505 506 507 508 509 510 511 512 513 514 515 516 517 \
  --case-min 40 --case-max 139 --development-max 89 \
  --output "$OUT"

echo V22_SAMEOBJECT_DISTANCE_DONE; date
