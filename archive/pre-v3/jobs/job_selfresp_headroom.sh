#!/bin/bash
#SBATCH --job-name=sr_head
#SBATCH --time=01:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --constraint=x86-64-v3
#SBATCH --partition=cluster
#SBATCH --output=/home/z/Zekang.Zhang/logs/sr_head_%j.out
set -o pipefail

# Is flow #1's small/faint SELF-response failure RECOVERABLE or an information limit?
# Fig 5 shows the defect cleanly (no blend term); this asks whether a regressor on the same
# features and the same truth beats the flow there. It also re-tests 2026-08-03j's attribution of
# the constgold small-size floor to R_blend, which rested on four models that shared one label.
# Result (job 15494037): flow -3.42 +- 1.10% at small size and -3.63 +- 1.34% at faint, where the
# pin-free 3-feature fit sits at zero -- the headroom is RECOVERABLE.
# FIREWALL: half-shear self-response only, constgold not read.
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
echo "### SELFRESP HEADROOM job=$SLURM_JOB_ID ###"; date
python -u scripts/diag_selfresp_headroom.py 2>&1 | grep -v --line-buffered "module command" || exit 1
echo; echo "SR_HEAD_DONE"; date
