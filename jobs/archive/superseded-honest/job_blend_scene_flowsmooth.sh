#!/bin/bash
#SBATCH --job-name=flowsmooth
#SBATCH --time=03:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --constraint=x86-64-v3
#SBATCH --partition=cluster
#SBATCH --output=/home/z/Zekang.Zhang/logs/flowsmooth_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/flowsmooth_%j.err
# HONEST self+blend acceptance, SMOOTH R_flow variant (WORKLOG cont.69). Same as
# job_blend_scene_harvest.sh but R_flow = HistGBR(flux,size) on the self sim instead of the
# 6x10 flux-quantile grid, to test whether flux-grid resolution drives the magnitude-decile
# residuals in the realistic (trueprop) family. R_blend scene-sum unchanged. Constgold
# validation-only; nothing is fit to it. Train 0-39, eval 40-139.
set -e
source /home/z/Zekang.Zhang/.bashrc
module load python/3.11-2023.09 2>/dev/null || true
conda activate /project/ls-gruen/users/zekang.zhang/envs/sims1
export PYTHONPATH="${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}:${BLENDEMU_ROOT:-/home/z/Zekang.Zhang/blendemu}:$PYTHONPATH"
export OMP_NUM_THREADS=8
cd "${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}"

CACHE=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk
RF=$CACHE/rflow_smooth_c40-139.npz
RB=$CACHE/rblend_scenesim_c40-139.npz   # reuse the identical scene-sum R_blend from cont.68

echo "=== HARVEST smooth R_flow (+ reuse scene R_blend), train 0-39, eval 40-139 ==="
python -u scripts/blend_scene_closure_test.py --train-cases 0-39 --eval-cases 40-139 --r-max 10 \
    --flow-model smooth --harvest-flow "$RF"

echo "=== ACCEPTANCE HARNESS with smooth R_flow + honest scene R_blend ==="
python -u scripts/eval_selection_robustness.py \
    --rflow-override "$RF" --rblend-override "$RB" --tag honest_flowsmooth
