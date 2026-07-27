#!/bin/bash
#SBATCH --job-name=v2_acc
#SBATCH --time=01:30:00
#SBATCH --mem=200G
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:a40:1
#SBATCH --constraint=x86-64-v3
#SBATCH --partition=inter
#SBATCH --output=/home/z/Zekang.Zhang/logs/v2_acc_%j.out

# Re-run the ACCEPTANCE-POPULATION bias check with the REAL V2 model.
#
# The earlier version of this check (WORKLOG cont.161) was run with the certified V1 flow and,
# in a retracted follow-up, with an ABLATION rung mislabelled V2.  Real V2 = the joint forward
# model `forward_ens_lr250_swa8_seed42{1..6}_joint.pt` (metadata.true_cut = (0.3, 26.0),
# primary_only_shear = True), paired with the 7"-gated emulator the V2 chain uses.
#
# Two loads, identical in every other respect:
#   ACC     true cut as loaded  -> the GOALS.md deliverable population on its own
#   RELAX   cut relaxed         -> ACCEPTED and REJECTED bands on IDENTICAL rows, which is the
#                                  only way to see whether a global number closes or merely
#                                  cancels between the two (the V1 finding being re-tested)
# Both at bridge=1.0, matching the previous V2 run; the npz carries per-seed R_flow and mag, so
# the ngmix->constgold bridge is applied afterwards from the dump instead of burning a second
# pass over the catalogue for a scalar that multiplies in at the end.
set -e
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
export OMP_NUM_THREADS=12 MKL_NUM_THREADS=12
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/inference-5b

CK=/project/ls-gruen/users/zekang.zhang/sbsi_caches/forward_proto/forward_ens_lr250_swa8_seed42*_joint.pt
NN=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/nn_dist_c0-39.feather
BLEND=results/blend_lookup_extnbrho_d7_c0-39.feather
OUTD=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk

echo "### V2_ACC job=$SLURM_JOB_ID node=$SLURMD_NODENAME ###"; nvidia-smi -L; date

echo; echo "########## 1/2  ACCEPTANCE population (true cut Re>0.3 & mag<26) ##########"
python -B -u scripts/eval_constgold_closure.py --ckpt-glob "$CK" --bridge 1.0 \
  --nn-lookup "$NN" --iso-radius 7.0 --blend-lookup "$BLEND" \
  --max-case 39 --true-re-min 0.3 --true-mag-max 26 \
  --output "$OUTD/v2acc_accept.npz"

echo; echo "########## 2/2  RELAXED load -> ACCEPTED vs REJECTED on identical rows ##########"
python -B -u scripts/eval_constgold_closure.py --ckpt-glob "$CK" --bridge 1.0 \
  --nn-lookup "$NN" --iso-radius 7.0 --blend-lookup "$BLEND" \
  --max-case 39 --true-re-min 0.0 --true-mag-max 99 \
  --output "$OUTD/v2acc_relaxed.npz"

echo "### V2_ACC_DONE job=$SLURM_JOB_ID ###"; date
