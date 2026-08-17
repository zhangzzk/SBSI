#!/bin/bash
#SBATCH --job-name=v22q3c_flow
#SBATCH --time=03:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=cip
#SBATCH --exclude=met-cl-vis01,met-cl-vis02
#SBATCH --output=/home/z/Zekang.Zhang/logs/v22q3c_flow_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/v22q3c_flow_%j.err

set -euo pipefail
ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
PY=/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python
CACHE=/project/ls-gruen/users/zekang.zhang/sbsi_caches
IN=$CACHE/derisk/v22_constgold_q3_decomp_c40-139_truth.feather
OUT=$CACHE/derisk/v22_constgold_q3_decomp_c40-139_flow.feather
SEEDS=(501 502 503 505 506 507 508 509 510 511 512 513 514 515 516 517)
CKPTS=()
for seed in "${SEEDS[@]}"; do
  ck="$CACHE/ablation/measurement_flow_g0_ngmix_ablate_s2c_lt500_v22_s${seed}_swaavg.pt"
  [ -f "$ck" ] || { echo "MISSING $ck"; exit 1; }
  CKPTS+=("$ck")
done
export LD_LIBRARY_PATH="/project/ls-gruen/users/zekang.zhang/envs/sims1/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT:/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}"
cd "$ROOT"
"$PY" -u scripts/score_v22_matched_decomposition_flow.py \
  --input "$IN" --ckpt "${CKPTS[@]}" --g 0.02 --n-samples 64 \
  --batch-size 16384 --flow-seed 12345 --output "$OUT"
echo V22_CONSTGOLD_Q3_DECOMP_C100_FLOW_DONE
