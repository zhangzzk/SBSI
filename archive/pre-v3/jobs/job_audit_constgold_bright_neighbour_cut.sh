#!/bin/bash
#SBATCH --job-name=cg_brightaudit
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/cg_brightaudit_%j.out
set -euo pipefail

ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
PY=/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python
export LD_LIBRARY_PATH="/project/ls-gruen/users/zekang.zhang/envs/sims1/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT:/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
cd "$ROOT"

BASE=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant
OUT=results/constgold_bright_neighbour_cut_audit_c40-49.json
[ ! -e "$OUT" ] || { echo "REFUSING overwrite $OUT"; exit 1; }

"$PY" -u scripts/audit_constgold_bright_neighbour_cut.py \
  --base "$BASE" \
  --catalogue "$BASE/constant_response_catalogue_train.feather" \
  --cases 40 41 42 43 44 45 46 47 48 49 \
  --output "$OUT"

echo CONSTGOLD_BRIGHT_AUDIT_DONE; date
