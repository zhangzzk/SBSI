#!/bin/bash
#SBATCH --job-name=bfens
#SBATCH --time=08:00:00
#SBATCH --mem=180G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=inter
#SBATCH --constraint=x86-64-v3
#SBATCH --array=0-15
#SBATCH --output=/home/z/Zekang.Zhang/logs/bfens_%A_%a.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/bfens_%A_%a.err

# 16-SEED ENSEMBLE OF FLOW #2, round-2 configuration (WORKLOG 2026-08-02m).
#
# WHY NOW. Every flow-#2 number quoted so far comes from ONE checkpoint, so its error bar has been a
# lower bound by construction (`job_blendflow_matched.sh` says exactly that). Two things then became
# unavoidable:
#   * The accidental replicate (2026-08-02j) showed same-seed reruns differ by 0.28 pt of summed bias
#     and ~0.2 in chi2/dof from CUDA nondeterminism alone. Seed scatter can only be larger.
#   * Two tuning attempts in a row returned nulls (2026-08-02k crowding, 2026-08-02l size grid), and
#     in both the per-bin tables showed the flow already sitting within ~1-2 sigma of the label. There
#     is nothing left to tune that is resolved by a single run; what is missing is the error bar.
#
# AGENTS.md's seed convention is about flow-#1 checkpoints, but its reasoning binds here: anything
# reporting `m` needs 16 seeds, and a bias on a shear response is exactly that. The 16 are
# 501 502 503 505..517 -- **504 does not exist**, and that gap is deliberate, not a typo.
#
# CONFIGURATION IS FROZEN at the round-2 winner: response 1000, strata 100, the THREE grids
# (sep x nbrmag / primag / prisize), select-on strata, 13 features. No crowding block (2026-08-02k
# found nothing for it to fix on this channel) and no fourth grid (2026-08-02l: -0.17 +- 0.12, a
# null, and it degraded separation through grid dilution). This array measures the model we have; it
# is not another search.
#
# Seeds 501/502/503 duplicate the g3 arm of 15482995 under a different tag. That is deliberate: three
# independent repeats of the same seed give a direct check that the ensemble spread reported here is
# consistent with the nondeterminism floor already measured, rather than assumed to be.
#
# Each task also scores its checkpoint against BlendEMU on the same held-out rows and saves the
# per-row responses, so `scripts/agg_blendflow_ensemble.py` can form BOTH statistics that matter and
# keep them distinct: the spread of the per-seed metrics (how much a single run can be trusted), and
# the metrics of the seed-AVERAGED response (what an ensemble prediction would actually deliver).
# Those are different numbers and conflating them would overstate the ensemble.
#
# FIREWALL: half-shear only. constgold is not opened here.
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

SEEDS=(501 502 503 505 506 507 508 509 510 511 512 513 514 515 516 517)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
TAG="ens"
OUT=$CACHE/blendflow_${TAG}_s${SEED}.pt

if [ ! -f "$PAIRSET" ]; then
  echo "MISSING $PAIRSET -- run jobs/job_blend_pairset_dual.sh first"; exit 1
fi

echo "ensemble member seed=$SEED  -> $OUT"
python -u scripts/train_blend_flow.py \
  --pairset "$PAIRSET" --output "$OUT" \
  --epochs 40 --batch-size 65536 --lr 1e-3 \
  --mean-hidden 256 --hidden-dim 128 --n-layers 3 --n-flows 6 \
  --delta 0.02 --nll-weight 1.0 --response-weight 1000.0 \
  --strata-weight 100.0 --select-on strata \
  --val-case-frac 0.2 --patience 10 --seed "$SEED" \
  || { echo "TRAIN FAILED s$SEED"; exit 1; }

echo "--- the ruler: flow vs BlendEMU on the SAME held-out rows ---"
python -u scripts/eval_blend_flow.py --pairset "$PAIRSET" --checkpoint "$OUT" \
  --save-npz "$CACHE/eval_${TAG}_s${SEED}.npz" || { echo "EVAL FAILED s$SEED"; exit 1; }

echo "BFENS_DONE seed=$SEED"; date
