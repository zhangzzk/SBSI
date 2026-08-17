#!/bin/bash
#SBATCH --job-name=v22rk_flow
#SBATCH --time=02:00:00
#SBATCH --mem=36G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a40-16gb:1
#SBATCH --partition=cip
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/v22rk_flow_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/v22rk_flow_%j.err

set -euo pipefail
ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
PY=/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python
CACHE=/project/ls-gruen/users/zekang.zhang/sbsi_caches/ablation
IN=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v22_matched_decomp_rankpilot_truth.feather
OUT=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v22_matched_decomp_rankpilot_flow.feather
SEEDS=(501 502 503 505 506 507 508 509 510 511 512 513 514 515 516 517)
CKPTS=()
for seed in "${SEEDS[@]}"; do
  ck="$CACHE/measurement_flow_g0_ngmix_ablate_s2c_lt500_v22_s${seed}_swaavg.pt"
  [ -f "$ck" ] || { echo "MISSING $ck"; exit 1; }
  CKPTS+=("$ck")
done
export LD_LIBRARY_PATH="/project/ls-gruen/users/zekang.zhang/envs/sims1/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT:/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}"
cd "$ROOT"
"$PY" -u scripts/score_v22_matched_decomposition_flow.py \
  --input "$IN" --ckpt "${CKPTS[@]}" --g 0.05 --n-samples 64 \
  --batch-size 16384 --flow-seed 12345 --output "$OUT"
echo V22_MATCHED_RANKPILOT_FLOW_DONE
