#!/bin/bash
#SBATCH --job-name=hsv27
#SBATCH --array=0-3
#SBATCH --time=03:00:00
#SBATCH --mem=40G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a40-16gb:1
#SBATCH --partition=cip
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/hsv27_%A_%a.out
set -euo pipefail
ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
PY=/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python
export LD_LIBRARY_PATH="/project/ls-gruen/users/zekang.zhang/envs/sims1/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT:/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}"
unset PYTORCH_CUDA_ALLOC_CONF
cd "$ROOT"
TASK=${SLURM_ARRAY_TASK_ID:?array task required}
ARMS=(sixshell sixshell purity purity); SEEDS=(501 502 501 502)
ARM=${ARMS[$TASK]}; SEED=${SEEDS[$TASK]}
C=/project/ls-gruen/users/zekang.zhang/sbsi_caches/v27_scene
CK=/project/ls-gruen/users/zekang.zhang/sbsi_caches/ablation/measurement_flow_g0_ngmix_ablate_s2c_lt500_v27_${ARM}_s${SEED}_swaavg.pt
OUTDIR=$C/halfshear_scores
OUT=$OUTDIR/${ARM}_s${SEED}.feather
mkdir -p "$OUTDIR"
for f in "$CK" "$C/halfshear_scenev27_base_c40-199_v22domain.feather"; do [ -f "$f" ] || { echo "MISSING $f"; exit 1; }; done
[ ! -e "$OUT" ] || { echo "REFUSING overwrite $OUT"; exit 1; }
"$PY" -u scripts/dump_halfshear_selfresp.py \
  --ckpt "$CK" --min-case 40 --max-case 199 \
  --base-cache "$C/halfshear_scenev27_base_c40-199_v22domain.feather" \
  --n-samples 32 --batch-size 16384 --blind --out "$OUT"
echo V27_HALFSHEAR_DONE arm=$ARM seed=$SEED; date
