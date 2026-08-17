#!/bin/bash
#SBATCH --job-name=tune_cip
#SBATCH --array=0-3
#SBATCH --time=04:00:00
#SBATCH --mem=30G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a40-16gb:1
#SBATCH --partition=cip
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/tune_cip_%A_%a.out

# Same parallel+pruned search as job_tune_indom_parallel.sh, but as an ARRAY on `cip` vGPU slices
# instead of one job holding a whole a40 on `inter`.
#
# WHY THE REROUTE. `inter` has 17 a40s and every one is allocated. Cancelling the serial job gave up
# its slot and another job took it immediately, so the one-node plan sat in PENDING (Priority) with
# no visible start. Meanwhile `cip` had 7 IDLE a40-16gb vGPU nodes. Waiting in a queue is the exact
# opposite of the acceleration that was asked for.
#
# THE PARALLEL DESIGN DOES NOT CARE WHERE THE WORKERS LIVE. They coordinate only through the shared
# SQLite study, so 4 workers on 4 nodes behave like 4 workers on 1 node -- each stops when the STUDY
# reaches the target trial count.
#
# SIZING, from measurements rather than guesses:
#   --mem=30G          : a worker's resident set was measured at ~18 GB. 30G is headroom AND it
#                        deliberately excludes the 26 GB h01g02n* nodes, steering the array onto the
#                        41 GB nodes (h01g04n3, h01g05n[1-3]) -- exactly 4 idle ones for 4 tasks.
#   --gres=a40-16gb:1  : a worker needs ~5.4 GB of GPU memory, so a 16 GB slice is ample. One worker
#                        per node here, versus 4 sharing one a40 before.
#   --cpus-per-task=8  : the real bottleneck is the per-round R2 computed in Python on CPU, so CPUs
#                        matter more than GPU. 8 per worker beats the 4 each they would have had
#                        sharing a 16-CPU node.
# The vGPU caveat in CLAUDE.md (NO_EXPANDABLE_SEGMENTS for A40-16Q) is a PyTorch allocator issue;
# XGBoost does not use the CUDA VMM APIs, so it does not apply. Nothing PyTorch runs here.
#
# FIREWALL unchanged: HELDOUT_MIN_CASE=40 keeps constgold (cases 0-39) out of training AND out of the
# Optuna validation split, so no hyperparameter is selected on constgold m.
#
# NOTE ON CONCURRENCY: the study is SQLite on NFS with 4 writers now on 4 DIFFERENT nodes. blendemu's
# storage already sets a 300 s busy timeout and pool_pre_ping for exactly this. The DB was backed up
# before any of this started; if it does get wedged, restore the backup and drop to K=2.
set -o pipefail
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/blendemu:/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:$PYTHONPATH"
export CONFIG_PATH=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot/configs/fs2_lsst_r_extnbr_indom_tuned.yaml
export HELDOUT_MIN_CASE=40
export XGB_DEVICE=${XGB_DEVICE:-cuda}
export XGB_NJOBS=${XGB_NJOBS:-8}
export OMP_NUM_THREADS=$XGB_NJOBS
unset WEIGHT_CLOSE
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot

TARGET=${TARGET:-100}
echo "### TUNE CIP worker=$SLURM_ARRAY_TASK_ID  job=$SLURM_JOB_ID  target=$TARGET  njobs=$XGB_NJOBS ###"
nvidia-smi -L; hostname; date

# Stagger by task id: 4 workers loading 18 GB each and touching the same SQLite file at the same
# instant is the one thing most likely to wedge NFS locking.
sleep $(( SLURM_ARRAY_TASK_ID * 45 ))

python -u scripts/tune_emulator_worker.py \
    --worker-id "$SLURM_ARRAY_TASK_ID" --target-trials "$TARGET" --n-jobs "$XGB_NJOBS" \
    2>&1 | grep -v "module command" \
  || { echo "TUNE_CIP_FAILED"; exit 1; }
echo "TUNE_CIP_DONE"; date
