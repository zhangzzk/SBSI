#!/bin/bash
#SBATCH --job-name=train160
#SBATCH --time=03:00:00
#SBATCH --mem=160G
#SBATCH --cpus-per-task=8
#SBATCH --constraint=x86-64-v3
#SBATCH --partition=cluster
#SBATCH --output=/home/z/Zekang.Zhang/logs/train160_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/train160_%j.err
# HONEST self+blend acceptance, MAX training data (WORKLOG cont.72). Definitive data-scaling test:
# train R_flow(smooth)+R_blend on 160 OOS cases {0-39} U {80-199}, eval constgold 40-79 (40 cases).
# train100 already tightened the faint-decile tail (10/12 realistic cuts <3%); this checks whether the
# residual keeps shrinking with more data (coverage-limited, keep going) or plateaus (floor reached).
# The isolated cut barely moved train40->train100 (-6.8->-6.3) -> watch if 160 helps it (structural?).
# OOS by case, constgold validation-only, estimator definition unchanged.
set -e
source /home/z/Zekang.Zhang/.bashrc
module load python/3.11-2023.09 2>/dev/null || true
conda activate /project/ls-gruen/users/zekang.zhang/envs/sims1
export PYTHONPATH="${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}:${BLENDEMU_ROOT:-/home/z/Zekang.Zhang/blendemu}:$PYTHONPATH"
export OMP_NUM_THREADS=8
cd "${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}"

CACHE=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk
RF=$CACHE/rflow_smooth_train160_c40-79.npz
RB=$CACHE/rblend_scene_train160_c40-79.npz

echo "=== HARVEST smooth R_flow + scene R_blend, train {0-39,80-199}=160 cases, eval 40-79 ==="
python -u scripts/blend_scene_closure_test.py --train-cases 0-39,80-199 --eval-cases 40-79 --r-max 10 \
    --flow-model smooth --harvest-flow "$RF" --harvest-blend "$RB"

echo "=== ACCEPTANCE HARNESS with 160-case-trained smooth R_flow + R_blend ==="
python -u scripts/eval_selection_robustness.py \
    --rflow-override "$RF" --rblend-override "$RB" --tag honest_train160 --cases-from-override
