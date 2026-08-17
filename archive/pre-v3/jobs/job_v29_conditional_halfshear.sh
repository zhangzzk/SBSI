#!/bin/bash
#SBATCH --job-name=hsv29cg
#SBATCH --array=0-11%6
#SBATCH --time=03:00:00
#SBATCH --mem=40G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a40-16gb:1
#SBATCH --partition=cip
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/hsv29cg_%A_%a.out
set -euo pipefail

# Score the five V2.9 arms and the existing V2.2 control on exactly the same half-shear rows.
ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
PY=/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python
export LD_LIBRARY_PATH="/project/ls-gruen/users/zekang.zhang/envs/sims1/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT:/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}"
unset PYTORCH_CUDA_ALLOC_CONF
cd "$ROOT"

TASK=${SLURM_ARRAY_TASK_ID:?array task required}
SEEDS=(501 502)
ARMS=(v22_control rb5_all rbc8_s6_cap4m rbc8_s6_all rbc8_s4_cap4m rbc8_s4_all)
ARM_INDEX=$((TASK / ${#SEEDS[@]}))
SEED_INDEX=$((TASK % ${#SEEDS[@]}))
(( ARM_INDEX < ${#ARMS[@]} )) || { echo "NO ARM for task $TASK"; exit 1; }
ARM=${ARMS[$ARM_INDEX]}
SEED=${SEEDS[$SEED_INDEX]}

if [ "$ARM" = v22_control ]; then
  CK=/project/ls-gruen/users/zekang.zhang/sbsi_caches/ablation/measurement_flow_g0_ngmix_ablate_s2c_lt500_v22_s${SEED}_swaavg.pt
else
  CK=/project/ls-gruen/users/zekang.zhang/sbsi_caches/v29_conditional/measurement_flow_g0_ngmix_${ARM}_s${SEED}_swaavg.pt
fi
C=/project/ls-gruen/users/zekang.zhang/sbsi_caches/v29_conditional
BASE=/project/ls-gruen/users/zekang.zhang/sbsi_caches/v27_scene/v28_rblend/halfshear_rblend_base_c40-199_v22domain.feather
OUTDIR=$C/halfshear_scores
OUT=$OUTDIR/${ARM}_s${SEED}.feather
mkdir -p "$OUTDIR"
for f in "$CK" "$BASE"; do [ -f "$f" ] || { echo "MISSING $f"; exit 1; }; done
[ ! -e "$OUT" ] || { echo "REFUSING overwrite $OUT"; exit 1; }

echo "### V2.9 HALF-SHEAR arm=$ARM seed=$SEED ###"; date
"$PY" -u scripts/dump_halfshear_selfresp.py \
  --ckpt "$CK" --min-case 40 --max-case 199 \
  --base-cache "$BASE" --n-samples 32 --batch-size 16384 --blind --out "$OUT"
echo "V29_CONDITIONAL_HALFSHEAR_DONE arm=$ARM seed=$SEED"; date
