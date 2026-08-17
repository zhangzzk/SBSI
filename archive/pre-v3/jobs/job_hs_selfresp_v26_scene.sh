#!/bin/bash
#SBATCH --job-name=hsv26scene
#SBATCH --time=03:00:00
#SBATCH --mem=36G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a40-16gb:1
#SBATCH --partition=cip
#SBATCH --constraint=x86-64-v3
#SBATCH --array=0-3
#SBATCH --output=/home/z/Zekang.Zhang/logs/hsv26scene_%A_%a.out
set -euo pipefail

PY=/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python
export LD_LIBRARY_PATH="/project/ls-gruen/users/zekang.zhang/envs/sims1/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}"
unset PYTORCH_CUDA_ALLOC_CONF
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot

SEEDS=(501 502 503 505)
SD=${SEEDS[${SLURM_ARRAY_TASK_ID:?array task required}]}
CK=/project/ls-gruen/users/zekang.zhang/sbsi_caches/ablation/measurement_flow_g0_ngmix_ablate_s2c_lt500_v26_scene_target_s${SD}_swaavg.pt
CACHE=/project/ls-gruen/users/zekang.zhang/sbsi_caches/hs_selfresp_v26_scene_c40-199/base_c40-199.feather
OUTDIR=/project/ls-gruen/users/zekang.zhang/sbsi_caches/hs_selfresp_v26_scene_target_c40-199
OUT=$OUTDIR/parts/part_s${SD}.feather
for f in "$CK" "$CACHE"; do [ -f "$f" ] || { echo "MISSING $f"; exit 1; }; done
mkdir -p "$OUTDIR/parts"
[ ! -e "$OUT" ] || { echo "REFUSING to overwrite $OUT"; exit 1; }

"$PY" -u scripts/dump_halfshear_selfresp.py \
  --ckpt "$CK" --min-case 40 --max-case 199 --base-cache "$CACHE" \
  --n-samples 32 --batch-size 16384 --blind --out "$OUT"
echo HS_V26_SCENE_TASK_DONE; date
