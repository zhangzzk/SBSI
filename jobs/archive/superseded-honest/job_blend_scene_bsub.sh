#!/bin/bash
#SBATCH --job-name=bsub
#SBATCH --time=03:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --constraint=x86-64-v3
#SBATCH --partition=cluster
#SBATCH --output=/home/z/Zekang.Zhang/logs/bsub_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/bsub_%j.err
# HONEST self+blend acceptance, SMOOTH R_flow + far-pair-baseline-subtracted R_blend (WORKLOG cont.70).
# Enforces the physical BC that shear leakage ->0 at the aperture edge by subtracting the blend-sim
# far-pair (8-10") prediction floor from every per-pair leakage (clip>=0). Targets the faint/isolated
# R_blend OVER-prediction that survives smooth R_flow. a-priori, blend-sim-only, constgold validation-only.
set -e
source /home/z/Zekang.Zhang/.bashrc
module load python/3.11-2023.09 2>/dev/null || true
conda activate /project/ls-gruen/users/zekang.zhang/envs/sims1
export PYTHONPATH="${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}:${BLENDEMU_ROOT:-/home/z/Zekang.Zhang/blendemu}:$PYTHONPATH"
export OMP_NUM_THREADS=8
cd "${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}"

CACHE=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk
RF=$CACHE/rflow_smooth_c40-139.npz          # identical to cont.69 smooth flow (re-harvested, deterministic)
RB=$CACHE/rblend_bsub_c40-139.npz

echo "=== HARVEST smooth R_flow + baseline-subtracted scene R_blend, train 0-39, eval 40-139 ==="
python -u scripts/blend_scene_closure_test.py --train-cases 0-39 --eval-cases 40-139 --r-max 10 \
    --flow-model smooth --blend-baseline-sub --harvest-flow "$RF" --harvest-blend "$RB"

echo "=== ACCEPTANCE HARNESS with smooth R_flow + baseline-subtracted R_blend ==="
python -u scripts/eval_selection_robustness.py \
    --rflow-override "$RF" --rblend-override "$RB" --tag honest_smooth_bsub
