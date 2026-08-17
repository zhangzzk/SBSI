#!/bin/bash
#SBATCH --job-name=hsv30ns
#SBATCH --array=0-1%2
#SBATCH --time=03:00:00
#SBATCH --mem=40G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a40:1
#SBATCH --partition=inter
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/hsv30ns_%A_%a.out
set -euo pipefail

ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
PY=/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python
export LD_LIBRARY_PATH="/project/ls-gruen/users/zekang.zhang/envs/sims1/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT:/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}"
unset PYTORCH_CUDA_ALLOC_CONF
cd "$ROOT"

TASK=${SLURM_ARRAY_TASK_ID:?array task required}
ARMS=(v30_m10_c16 v30_m12_c16)
ARM=${ARMS[$TASK]:?unknown task}
SEED=501
CACHE=/project/ls-gruen/users/zekang.zhang/sbsi_caches/v30_nosize_grid
CK=$CACHE/measurement_flow_g0_ngmix_${ARM}_s${SEED}_swaavg.pt
BASE=/project/ls-gruen/users/zekang.zhang/sbsi_caches/v27_scene/v28_rblend/halfshear_rblend_base_c40-199_v22domain.feather
OUTDIR=$CACHE/halfshear_scores
OUT=$OUTDIR/${ARM}_s${SEED}.feather
mkdir -p "$OUTDIR"
for file in "$CK" "$BASE"; do [ -f "$file" ] || { echo "MISSING $file"; exit 1; }; done
[ ! -e "$OUT" ] || { echo "REFUSING overwrite $OUT"; exit 1; }

echo "### V3.0 NO-SIZE HALF-SHEAR arm=$ARM seed=$SEED ###"; date
"$PY" -u scripts/dump_halfshear_selfresp.py \
  --ckpt "$CK" --min-case 40 --max-case 199 \
  --base-cache "$BASE" --n-samples 32 --batch-size 16384 --blind --out "$OUT"
echo "V30_NOSIZE_HALFSHEAR_DONE arm=$ARM seed=$SEED"; date
