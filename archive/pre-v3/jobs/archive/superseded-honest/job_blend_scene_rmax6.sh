#!/bin/bash
#SBATCH --job-name=rmax6
#SBATCH --time=03:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --constraint=x86-64-v3
#SBATCH --partition=cluster
#SBATCH --output=/home/z/Zekang.Zhang/logs/rmax6_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/rmax6_%j.err
# HONEST self+blend acceptance, smooth R_flow + 6" scene aperture (WORKLOG cont.72). The blend-sim
# leakage curve (diag_floor.py) falls to <0.004 beyond 6"; a 6" aperture keeps the real 0-6" leakage
# but drops the many spurious far neighbours whose per-pair floor SUMS into faint/isolated over-fill.
# Cleaner than the flat baseline-sub (which also shaved real near leakage). a-priori (leakage-curve
# calibrated, NOT |m|-tuned), single model, constgold validation-only. Train 0-39, eval 40-139.
set -e
source /home/z/Zekang.Zhang/.bashrc
module load python/3.11-2023.09 2>/dev/null || true
conda activate /project/ls-gruen/users/zekang.zhang/envs/sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
export OMP_NUM_THREADS=8
cd /home/z/Zekang.Zhang/SBSI

CACHE=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk
RF=$CACHE/rflow_smooth_rmax6_c40-139.npz
RB=$CACHE/rblend_scene_rmax6_c40-139.npz

echo "=== HARVEST smooth R_flow + 6\" scene R_blend, train 0-39, eval 40-139 ==="
python -u scripts/blend_scene_closure_test.py --train-cases 0-39 --eval-cases 40-139 --r-max 6 \
    --flow-model smooth --harvest-flow "$RF" --harvest-blend "$RB"

echo "=== ACCEPTANCE HARNESS with smooth R_flow + 6\" R_blend ==="
python -u scripts/eval_selection_robustness.py \
    --rflow-override "$RF" --rblend-override "$RB" --tag honest_rmax6
