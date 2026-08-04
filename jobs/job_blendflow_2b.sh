#!/bin/bash
#SBATCH --job-name=bf2b
#SBATCH --time=08:00:00
#SBATCH --mem=180G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=inter
#SBATCH --constraint=x86-64-v3
#SBATCH --array=0-3
#SBATCH --output=/home/z/Zekang.Zhang/logs/bf2b_%A_%a.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/bf2b_%A_%a.err

# STEP 2b OF MERGING FLOWS #1 AND #2 (WORKLOG 2026-08-02j) -- fixing the TWO defects step 2 measured.
#
# STEP 2 SAID: the two channels coexist globally (self response -0.60%, best-ever blend separation
# chi2/dof 0.1), but the self response is far too FLAT in every conditional, and on crowding it has
# the trend BACKWARDS -- truth falls 15% from k=1 to k=6-9, the model rises 1%. Sweeping the self
# weight 300/1000/3000 moved the magnitude chi2/dof 1338/791/510 and left it hugely flat, so this is
# not an optimisation shortfall.
#
# TWO DEFECTS, TWO DIFFERENT FIXES -- and this array is built to tell them apart rather than to throw
# both in and hope:
#
#   --crowding            adds nbr_flux_near/far/max and log_k, computed from the pair set's own
#                         neighbour rows with the SAME shells, zero point and noise normalisation as
#                         `scripts/build_crowding_lookup.py`, so they are the quantity flow #1
#                         conditions on and not a lookalike. This is aimed at CROWDING, which no
#                         anchor can fix: anchoring on k would force the marginal to match while the
#                         network has no crowding input to condition on.
#   --self-strata-weight  anchors the per-cell SELF mean over primary mag x size x k, with anchors
#                         formed over PRIMARIES (every self grid keys on primary properties, so a
#                         primary's k rows share a cell and the collapse is exact). This is aimed at
#                         the FLATNESS on the primary's own axes -- the same symptom, and the same
#                         remedy, that fixed the blend channel in rounds 1 and 2.
#
# THE 2x2. Task 0 = crowding only. Task 1 = anchors only. Task 2 = both. The "neither" control is
# step 2's `blendflow_dual_sw1000.0_s501.pt`, already run on this same pair set, so no GPU time is
# spent re-deriving it. Task 3 repeats "both" at a stronger anchor weight.
#
# WEIGHT SCALE, sized from measured values rather than guessed. Every other term contributes ~1000 to
# the loss (response 1.0 x 1000; blend strata ~12 x 100). The self strata term is a chi2/dof against
# well-determined per-primary anchors and starts in the hundreds, so parity is around 3, not 100. The
# trainer now PRINTS this term (`train sstr`) even at weight 0, so the next sweep can be sized from
# the run rather than from arithmetic. (A previous sweep guessed 0.1/1/10 and would have spent all
# three tasks on the flat end.)
#
# EVERYTHING ELSE IS HELD AT STEP 2's BEST: response 1000, blend strata 100 on three grids,
# self response 1000, --select-on strata. Only the two new levers move.
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

CROWD_LIST=(1 0 1 1)          # --crowding on/off
SSW_LIST=(0.0 3.0 3.0 10.0)   # --self-strata-weight
NAME_LIST=(crowdonly anchoronly both bothstrong)

CROWD=${CROWD_LIST[$SLURM_ARRAY_TASK_ID]}
SSW=${SSW_LIST[$SLURM_ARRAY_TASK_ID]}
TAG="2b_${NAME_LIST[$SLURM_ARRAY_TASK_ID]}"
OUT=$CACHE/blendflow_${TAG}_s${SEED}.pt

CROWD_FLAG=""
[ "$CROWD" = "1" ] && CROWD_FLAG="--crowding"

if [ ! -f "$PAIRSET" ]; then
  echo "MISSING $PAIRSET -- run jobs/job_blend_pairset_dual.sh first"; exit 1
fi

echo "variant=$TAG  crowding=$CROWD  self-strata-weight=$SSW  -> $OUT"
python -u scripts/train_blend_flow.py \
  --pairset "$PAIRSET" --output "$OUT" \
  --epochs 40 --batch-size 65536 --lr 1e-3 \
  --mean-hidden 256 --hidden-dim 128 --n-layers 3 --n-flows 6 \
  --delta 0.02 --nll-weight 1.0 --response-weight 1000.0 \
  --strata-weight 100.0 --select-on strata \
  --self-response-weight 1000.0 --self-strata-weight "$SSW" $CROWD_FLAG \
  --val-case-frac 0.2 --patience 10 --seed "$SEED" \
  || { echo "TRAIN FAILED $TAG"; exit 1; }

echo "--- the ruler: flow vs BlendEMU on the SAME held-out rows (decides the BLEND side) ---"
python -u scripts/eval_blend_flow.py --pairset "$PAIRSET" --checkpoint "$OUT" \
  --save-npz "$CACHE/eval_${TAG}_s${SEED}.npz" || { echo "EVAL FAILED $TAG"; exit 1; }

echo "BF2B_DONE $TAG"; date
