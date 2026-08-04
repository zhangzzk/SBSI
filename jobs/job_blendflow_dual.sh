#!/bin/bash
#SBATCH --job-name=bfdual
#SBATCH --time=08:00:00
#SBATCH --mem=180G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=inter
#SBATCH --constraint=x86-64-v3
#SBATCH --array=0-3
#SBATCH --output=/home/z/Zekang.Zhang/logs/bfdual_%A_%a.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/bfdual_%A_%a.err

# STEP 2 OF MERGING FLOWS #1 AND #2 (WORKLOG 2026-08-02i).
#
# THE QUESTION. Flows #1 and #2 model the same object and differ mainly in which galaxy's shape the
# shear map acts on. Flow #2 already conditions on the primary's oriented intrinsic shape, so the
# self-response channel exists in it today and has simply never been supervised. Supervise it, and
# ask whether ONE network can carry a ~0.86 self response and a ~0.14 blend response at once.
#
# WHY THAT IS NOT OBVIOUS. The NLL-only run already measured what happens when the two channels
# compete without supervision: maximum likelihood spent its capacity on the primary's own shape
# (correlation ~0.8 with the target) and under-fitted the neighbour's (~0.035), leaving the blend
# response 70-85% too low and nearly flat. Supervising both is precisely an attempt to stop the
# strong channel from crowding out the weak one. It may not work, and the failure mode to watch for
# is the blend side degrading while the self side looks fine.
#
# THE SWEEP. Both response terms are normalised by their own label variance, so both sit near 1.0 and
# the blend side already runs at weight 1000. Symmetric would be 1000; the sweep brackets it 3x
# either way. Task 0 is the CONTROL -- self weight 0, i.e. the round-2 blend-only model retrained on
# THIS pair set, so the comparison is not confounded by the pair set changing underneath it.
#
# There is no self-side control here, and that is not an oversight: an unsupervised model's R_self is
# not architecturally comparable (its residual flow is not blinded to the primary's shape), and
# flow #1's own R_self is not comparable either -- different inputs, different outputs, and it sees
# isolated primaries that this pair set does not contain at all. The self numbers stand on the
# held-out label, not on a baseline.
#
# EVERYTHING ELSE IS THE ROUND-2 WINNER, held fixed: response weight 1000, strata weight 100 on all
# three marginal grids, --select-on strata. Building on the best known blend configuration is what
# makes any blend-side degradation attributable to the self channel rather than to a co-varying
# hyperparameter.
#
# WHAT DECIDES IT. The blend side is decided by `eval_blend_flow.py` -- flow and emulator scored on
# the SAME held-out rows -- exactly as in rounds 1 and 2. The self side is decided by the per-PRIMARY
# tables the trainer prints (magnitude, size, and crowding), read against the label's own sem. Read
# both. A model that fixes one by breaking the other has not merged anything.
#
# FIREWALL: half-shear only. constgold is not opened by any step here.
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
if [ "${NO_EXPANDABLE_SEGMENTS:-0}" = "1" ]; then
  unset PYTORCH_CUDA_ALLOC_CONF
else
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
fi
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot

CACHE=/project/ls-gruen/users/zekang.zhang/sbsi_caches/blendflow
PAIRSET=$CACHE/blend_pairset_ap7_dual.feather
SEED=501

SW_LIST=(0.0 300.0 1000.0 3000.0)
SW=${SW_LIST[$SLURM_ARRAY_TASK_ID]}
TAG="dual_sw${SW}"
OUT=$CACHE/blendflow_${TAG}_s${SEED}.pt

if [ ! -f "$PAIRSET" ]; then
  echo "MISSING $PAIRSET -- run jobs/job_blend_pairset_dual.sh first"; exit 1
fi

echo "self-response weight = $SW   -> $OUT"
python -u scripts/train_blend_flow.py \
  --pairset "$PAIRSET" --output "$OUT" \
  --epochs 40 --batch-size 65536 --lr 1e-3 \
  --mean-hidden 256 --hidden-dim 128 --n-layers 3 --n-flows 6 \
  --delta 0.02 --nll-weight 1.0 --response-weight 1000.0 \
  --strata-weight 100.0 --select-on strata \
  --self-response-weight "$SW" \
  --val-case-frac 0.2 --patience 10 --seed "$SEED" \
  || { echo "TRAIN FAILED sw=$SW"; exit 1; }

echo "--- the ruler: flow vs BlendEMU on the SAME held-out rows (decides the BLEND side) ---"
python -u scripts/eval_blend_flow.py --pairset "$PAIRSET" --checkpoint "$OUT" \
  --save-npz "$CACHE/eval_${TAG}_s${SEED}.npz" || { echo "EVAL FAILED sw=$SW"; exit 1; }

echo "BFDUAL_DONE sw=$SW"; date
