#!/bin/bash
#SBATCH --job-name=cg_nd_a
#SBATCH --time=01:00:00
#SBATCH --mem=200G
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --partition=inter
#SBATCH --constraint=x86-64-v3
#SBATCH --array=0-15
#SBATCH --output=/home/z/Zekang.Zhang/logs/cg_nd_a_%A_%a.out

# Near-domain constgold selection table, FANNED OUT ONE SEED PER TASK.
#
# The sequential 16-seed job is ~2h20 (setup ~8 min + ~8 min per seed at the full 11.67M-row
# population). Seeds are independent given the prepared population, so each task rebuilds the
# population and scores exactly ONE checkpoint: wall clock becomes setup + one seed, ~16 min.
# The population build is duplicated 16x -- that is the price, and it is CPU only.
#
# Each task writes a JSON dump carrying its accumulators, the sim side, and a population
# fingerprint. `scripts/merge_neardomain_seeds.py` refuses to combine dumps whose fingerprints
# disagree or whose checkpoints repeat, so a drifted or double-counted task cannot pass silently.
#
# Run the merge AFTER the array completes:
#   sbatch --dependency=afterok:<ARRAY_JOBID> jobs/job_constgold_neardomain_merge.sh
set -o pipefail
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
# Fail loudly if the env did not take. Without `conda activate sims1` this falls through to the
# system python 3.11 torch, whose allocator does not know `expandable_segments` -- the first run of
# this script died 16x with "Unrecognized CachingAllocator option" after ~90s each. A one-line check
# turns that into an immediate, readable failure instead of a torch traceback.
python -c "import torch,sys; sys.exit(0 if hasattr(torch.cuda,'is_available') and torch.__version__>='2' else 1)" \
  || { echo "WRONG PYTHON/TORCH -- conda activate sims1 did not take"; exit 1; }
D=/project/ls-gruen/users/zekang.zhang/sbsi_caches/ablation
# DUMPDIR and EXTRA are overridable so a variant cut list (e.g. --complements) can be run without
# mixing its dumps into the canonical set. The merge reads a whole directory, and dumps built from
# DIFFERENT cut lists must never share one -- the fingerprint guards the population, not the cuts.
DUMPDIR=${DUMPDIR:-results/nd_seeds}
EXTRA=${EXTRA:-}
mkdir -p $DUMPDIR
echo "dumpdir=$DUMPDIR extra='$EXTRA'"

# 16 e-response seeds; 504 does not exist. Index by SLURM_ARRAY_TASK_ID.
SEEDS=(501 502 503 505 506 507 508 509 510 511 512 513 514 515 516 517)
SD=${SEEDS[$SLURM_ARRAY_TASK_ID]}
if [ -z "$SD" ]; then echo "no seed for array index $SLURM_ARRAY_TASK_ID"; exit 1; fi
CK=$D/measurement_flow_g0_ngmix_ablate_s2c_lt500_dom6x6_s${SD}_swaavg.pt

echo "### CONSTGOLD NEAR-DOMAIN ARRAY job=$SLURM_ARRAY_JOB_ID task=$SLURM_ARRAY_TASK_ID seed=$SD ###"
nvidia-smi -L; date
python -u scripts/eval_selection_constgold_neardomain.py \
  --ckpt $CK --n-samples 32 --batch-size 16384 --max-rows 0 \
  --dump-per-seed $DUMPDIR/s${SD}.json $EXTRA \
  2>&1 | grep -v --line-buffered "module command" || { echo CG_ND_A_FAILED; exit 1; }
echo CG_ND_A_TASK_DONE; date
