#!/bin/bash
#SBATCH --job-name=hsv30ns4
#SBATCH --array=0-7%8
#SBATCH --time=03:00:00
#SBATCH --mem=40G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:v100:1
#SBATCH --partition=inter
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/hsv30ns4_%A_%a.out
set -euo pipefail

ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
PY=/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python
export LD_LIBRARY_PATH="/project/ls-gruen/users/zekang.zhang/envs/sims1/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT:/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}"
unset PYTORCH_CUDA_ALLOC_CONF
cd "$ROOT"

TASK=${SLURM_ARRAY_TASK_ID:?array task required}
ARMS=(
  v22_control v22_control
  v30_m10_c16 v30_m10_c16 v30_m10_c16
  v30_m12_c16 v30_m12_c16 v30_m12_c16
)
SEEDS=(503 505 502 503 505 502 503 505)
ARM=${ARMS[$TASK]:?unknown arm for task $TASK}
SEED=${SEEDS[$TASK]:?unknown seed for task $TASK}

C=/project/ls-gruen/users/zekang.zhang/sbsi_caches/v30_nosize_grid
V29=/project/ls-gruen/users/zekang.zhang/sbsi_caches/v29_conditional
if [ "$ARM" = v22_control ]; then
  CK=/project/ls-gruen/users/zekang.zhang/sbsi_caches/ablation/measurement_flow_g0_ngmix_ablate_s2c_lt500_v22_s${SEED}_swaavg.pt
  OUTDIR=$V29/halfshear_scores
else
  CK=$C/measurement_flow_g0_ngmix_${ARM}_s${SEED}_swaavg.pt
  OUTDIR=$C/halfshear_scores
fi
BASE=/project/ls-gruen/users/zekang.zhang/sbsi_caches/v27_scene/v28_rblend/halfshear_rblend_base_c40-199_v22domain.feather
OUT=$OUTDIR/${ARM}_s${SEED}.feather
mkdir -p "$OUTDIR"
for file in "$CK" "$BASE"; do [ -f "$file" ] || { echo "MISSING $file"; exit 1; }; done
[ ! -e "$OUT" ] || { echo "REFUSING overwrite $OUT"; exit 1; }

echo "### V3.0 FOUR-SEED HALF-SHEAR arm=$ARM seed=$SEED ###"; date
"$PY" -u scripts/dump_halfshear_selfresp.py \
  --ckpt "$CK" --min-case 40 --max-case 199 \
  --base-cache "$BASE" --n-samples 32 --batch-size 16384 --blind --out "$OUT"
echo "V30_NOSIZE_MORESEED_HALFSHEAR_DONE arm=$ARM seed=$SEED"; date
