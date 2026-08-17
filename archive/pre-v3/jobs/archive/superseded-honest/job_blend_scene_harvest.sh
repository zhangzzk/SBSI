#!/bin/bash
#SBATCH --job-name=blendscene
#SBATCH --time=03:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --constraint=x86-64-v3
#SBATCH --partition=cluster
#SBATCH --output=/home/z/Zekang.Zhang/logs/blendscene_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/blendscene_%j.err
# HONEST self+blend acceptance (WORKLOG cont.68). Trains R_flow (shape self, self sim) and a
# per-pair blend model (blend sim) on OOS cases 0-39, harvests per-object R_flow + scene-summed
# R_blend over constgold cases 40-139, then runs the certified acceptance harness with those
# overrides (dump r_sim untouched). Constgold is validation-only; nothing is fit to it.
set -e
source /home/z/Zekang.Zhang/.bashrc
module load python/3.11-2023.09 2>/dev/null || true
conda activate /project/ls-gruen/users/zekang.zhang/envs/sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
export OMP_NUM_THREADS=8
cd /home/z/Zekang.Zhang/SBSI

CACHE=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk
RF=$CACHE/rflow_selfsim_c40-139.npz
RB=$CACHE/rblend_scenesim_c40-139.npz

echo "=== HARVEST honest R_flow + scene-summed R_blend (train 0-39, eval 40-139) ==="
python -u scripts/blend_scene_closure_test.py --train-cases 0-39 --eval-cases 40-139 --r-max 10 \
    --harvest-flow "$RF" --harvest-blend "$RB"

echo "=== ACCEPTANCE HARNESS with honest overrides ==="
python -u scripts/eval_selection_robustness.py \
    --rflow-override "$RF" --rblend-override "$RB" --tag honest_selfblend
