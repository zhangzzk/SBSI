#!/bin/bash
#SBATCH --job-name=hsv24tp
#SBATCH --time=03:00:00
#SBATCH --mem=36G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a40-16gb:1
#SBATCH --partition=cip
#SBATCH --constraint=x86-64-v3
#SBATCH --array=0-3
#SBATCH --output=/home/z/Zekang.Zhang/logs/hsv24tp_%A_%a.out
set -euo pipefail

PY=/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python
export LD_LIBRARY_PATH="/project/ls-gruen/users/zekang.zhang/envs/sims1/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}"
unset PYTORCH_CUDA_ALLOC_CONF
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot

SEEDS=(501 502 503 505)
SD=${SEEDS[${SLURM_ARRAY_TASK_ID:?array task required}]}
CACHE=/project/ls-gruen/users/zekang.zhang/sbsi_caches/hs_selfresp_v23_thirdplus_c40-199/base_c40-199.feather
OUTDIR=/project/ls-gruen/users/zekang.zhang/sbsi_caches/hs_selfresp_v24_thirdplus_target_c40-199
CK=/project/ls-gruen/users/zekang.zhang/sbsi_caches/ablation/measurement_flow_g0_ngmix_ablate_s2c_lt500_v24_thirdplus_target_s${SD}_swaavg.pt
OUT=$OUTDIR/parts/part_s${SD}.feather
for f in "$CACHE" "$CK"; do [ -f "$f" ] || { echo "MISSING $f"; exit 1; }; done
mkdir -p "$OUTDIR/parts"
[ ! -e "$OUT" ] || { echo "REFUSING to overwrite $OUT"; exit 1; }

echo "### V2.4 THIRD-PLUS TARGET HALF-SHEAR seed=$SD job=$SLURM_JOB_ID ###"; nvidia-smi -L; date
"$PY" -u scripts/dump_halfshear_selfresp.py \
  --ckpt "$CK" --min-case 40 --max-case 199 --base-cache "$CACHE" \
  --n-samples 32 --batch-size 16384 --blind --out "$OUT"
echo HS_V24_THIRDPLUS_PART_DONE; date
