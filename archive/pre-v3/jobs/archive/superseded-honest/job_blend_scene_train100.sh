#!/bin/bash
#SBATCH --job-name=train100
#SBATCH --time=03:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --constraint=x86-64-v3
#SBATCH --partition=cluster
#SBATCH --output=/home/z/Zekang.Zhang/logs/train100_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/train100_%j.err
# HONEST self+blend acceptance, 2.5x TRAINING DATA (owner-suggested; WORKLOG cont.72). Self and blend
# sims both span cases 0-199 (cases0_99 + cases100_199 shards). Train R_flow(smooth)+R_blend on the 100
# cases DISJOINT from the eval set: {0-39} U {140-199}. Eval constgold stays 40-139 (100 cases) so this
# is directly comparable to honest_flowsmooth (which trained on 40 cases, 0-39). Tests whether more
# training data (variance + faint/rare-config coverage) shrinks the faint-decile residuals. OOS by case,
# constgold validation-only, estimator definition unchanged.
set -e
source /home/z/Zekang.Zhang/.bashrc
module load python/3.11-2023.09 2>/dev/null || true
conda activate /project/ls-gruen/users/zekang.zhang/envs/sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
export OMP_NUM_THREADS=8
cd /home/z/Zekang.Zhang/SBSI

CACHE=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk
RF=$CACHE/rflow_smooth_train100_c40-139.npz
RB=$CACHE/rblend_scene_train100_c40-139.npz

echo "=== HARVEST smooth R_flow + scene R_blend, train {0-39,140-199}=100 cases, eval 40-139 ==="
python -u scripts/blend_scene_closure_test.py --train-cases 0-39,140-199 --eval-cases 40-139 --r-max 10 \
    --flow-model smooth --harvest-flow "$RF" --harvest-blend "$RB"

echo "=== ACCEPTANCE HARNESS with 100-case-trained smooth R_flow + R_blend ==="
python -u scripts/eval_selection_robustness.py \
    --rflow-override "$RF" --rblend-override "$RB" --tag honest_train100
