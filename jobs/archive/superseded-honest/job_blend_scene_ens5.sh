#!/bin/bash
#SBATCH --job-name=ens5
#SBATCH --time=04:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --constraint=x86-64-v3
#SBATCH --partition=cluster
#SBATCH --output=/home/z/Zekang.Zhang/logs/ens5_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/ens5_%j.err
# HONEST self+blend acceptance, BAGGED ENSEMBLE (WORKLOG cont.71). n=5 bootstrap-bagged HistGBR members
# for BOTH R_flow(smooth) and R_blend, predictions averaged. NO baseline-sub, so this isolates the pure
# ensemble effect vs honest_flowsmooth (single, no bsub). Tests whether the faint-decile residuals
# (q5/q9/q10 -4..-9%) are single-model-fit noise (which a case-bootstrap z cannot see) or true bias.
# a-priori, sim-only, constgold validation-only. Train 0-39, eval 40-139.
set -e
source /home/z/Zekang.Zhang/.bashrc
module load python/3.11-2023.09 2>/dev/null || true
conda activate /project/ls-gruen/users/zekang.zhang/envs/sims1
export PYTHONPATH="${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}:${BLENDEMU_ROOT:-/home/z/Zekang.Zhang/blendemu}:$PYTHONPATH"
export OMP_NUM_THREADS=8
cd "${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}"

CACHE=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk
RF=$CACHE/rflow_smooth_ens5_c40-139.npz
RB=$CACHE/rblend_scene_ens5_c40-139.npz

echo "=== HARVEST bagged-5 smooth R_flow + bagged-5 scene R_blend (no bsub), train 0-39, eval 40-139 ==="
python -u scripts/blend_scene_closure_test.py --train-cases 0-39 --eval-cases 40-139 --r-max 10 \
    --flow-model smooth --n-ensemble 5 --harvest-flow "$RF" --harvest-blend "$RB"

echo "=== ACCEPTANCE HARNESS with bagged-5 R_flow + bagged-5 R_blend ==="
python -u scripts/eval_selection_robustness.py \
    --rflow-override "$RF" --rblend-override "$RB" --tag honest_ens5
