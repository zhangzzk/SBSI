#!/bin/bash
#SBATCH --job-name=v27rho_cg
#SBATCH --array=0-9
#SBATCH --time=12:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=10
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/v27rho_cg_%A_%a.%N.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/v27rho_cg_%A_%a.%N.err
set -euo pipefail
ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
PY=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
C=/project/ls-gruen/users/zekang.zhang/sbsi_caches/v27_scene
TASK=${SLURM_ARRAY_TASK_ID:?array task required}
START=$((40 + 10 * TASK)); END=$((START + 9))
OUT=$C/parts/scene_features_const_r2_c${START}-${END}.feather
mkdir -p "$C/parts"
[ ! -e "$OUT" ] || { echo "REFUSING to overwrite $OUT"; exit 1; }
CASES=(); for ((c=START;c<=END;c++)); do CASES+=("$c"); done
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$ROOT"
"$PY" -u scripts/compute_scene_features_v27.py \
  --manifest "$C/scene_anchor_manifest_const_c40-139.feather" \
  --base /project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant \
  --sign 0.02 --cases "${CASES[@]}" --n-jobs 10 --max-stamp-size 768 --output "$OUT"
echo V27_SCENE_FEATURES_CONST_DONE cases=$START-$END
