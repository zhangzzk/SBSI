#!/bin/bash
#SBATCH --job-name=blflow
#SBATCH --time=03:00:00
#SBATCH --mem=150G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=inter
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/blflow_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/blflow_%j.err

# Gold-V3 FLOW #2 -- the blending-response flow, a firewalled drop-in for BlendEMU.
# Trains on the half-shear pair set only; constgold is never opened.
#
# SEEDS: flow #2's output is an e-response (R_blend enters `m` as a shape response), so
# AGENTS.md's e-response standard binds -- any `m` quoted from this model needs the full 16
# seeds. One seed first to establish the model works at all; scale after.
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
# The A40-16Q vGPU slices on `cip` lack the CUDA virtual-memory APIs expandable_segments needs.
if [ "${NO_EXPANDABLE_SEGMENTS:-0}" = "1" ]; then
  unset PYTORCH_CUDA_ALLOC_CONF
else
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
fi
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot

SEED=${SEED:-501}
PAIRSET=${PAIRSET:-/project/ls-gruen/users/zekang.zhang/sbsi_caches/blendflow/blend_pairset_ap7.feather}
D=/project/ls-gruen/users/zekang.zhang/sbsi_caches/blendflow
TAG=${TAG:-v1}
OUT=$D/blendflow_${TAG}_s${SEED}.pt

echo "### BLEND FLOW #2  tag=$TAG seed=$SEED job=$SLURM_JOB_ID ###"; nvidia-smi -L; date
python -u scripts/train_blend_flow.py \
  --pairset "$PAIRSET" --output "$OUT" \
  --epochs ${EPOCHS:-40} --batch-size ${BS:-65536} --lr ${LR:-1e-3} \
  --mean-hidden ${MH:-256} --hidden-dim ${HD:-128} --n-layers ${NL:-3} --n-flows ${NF:-6} \
  --delta ${DELTA:-0.02} --nll-weight ${NW:-1.0} --response-weight ${RW:-10.0} \
  --val-case-frac ${VCF:-0.2} --patience ${PAT:-10} --seed "$SEED" \
  || { echo "FAILED seed=$SEED"; exit 1; }

echo "--- scoring against BlendEMU on the SAME held-out rows ---"
python -u scripts/eval_blend_flow.py --pairset "$PAIRSET" --checkpoint "$OUT" \
  --save-npz "$D/eval_${TAG}_s${SEED}.npz" || { echo "EVAL FAILED"; exit 1; }
echo "BLFLOW_DONE tag=$TAG seed=$SEED"; date
