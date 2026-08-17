#!/bin/bash
#SBATCH --job-name=tr100ev79
#SBATCH --time=03:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --constraint=x86-64-v3
#SBATCH --partition=cluster
#SBATCH --output=/home/z/Zekang.Zhang/logs/tr100ev79_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/tr100ev79_%j.err
# DECONFOUND the data-scaling test (WORKLOG cont.72). train160 (160 cases, eval 40-79) looked WORSE than
# train100 (100 cases, eval 40-139) -- but the eval set differed. This runs train100's EXACT training set
# {0-39,140-199}=100 cases on the SAME eval 40-79 as train160, so train100 vs train160 is a clean
# 100-vs-160 comparison at fixed eval. If train100@4079 beats train160@4079 -> more data (160) HURTS
# (plateau/overfit -> framework floor ~100 cases); if train160 wins -> more data still helps (recommend
# more sims). OOS by case (train {0-39,140-199} disjoint from eval 40-79), constgold validation-only.
set -e
source /home/z/Zekang.Zhang/.bashrc
module load python/3.11-2023.09 2>/dev/null || true
conda activate /project/ls-gruen/users/zekang.zhang/envs/sims1
export PYTHONPATH="${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}:${BLENDEMU_ROOT:-/home/z/Zekang.Zhang/blendemu}:$PYTHONPATH"
export OMP_NUM_THREADS=8
cd "${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}"

CACHE=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk
RF=$CACHE/rflow_smooth_tr100_ev4079.npz
RB=$CACHE/rblend_scene_tr100_ev4079.npz

echo "=== HARVEST smooth R_flow + scene R_blend, train {0-39,140-199}=100 cases, eval 40-79 ==="
python -u scripts/blend_scene_closure_test.py --train-cases 0-39,140-199 --eval-cases 40-79 --r-max 10 \
    --flow-model smooth --harvest-flow "$RF" --harvest-blend "$RB"

echo "=== ACCEPTANCE HARNESS (eval 40-79, restricted) ==="
python -u scripts/eval_selection_robustness.py \
    --rflow-override "$RF" --rblend-override "$RB" --tag honest_tr100ev79 --cases-from-override
