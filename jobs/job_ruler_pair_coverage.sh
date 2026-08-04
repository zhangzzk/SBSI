#!/bin/bash
#SBATCH --job-name=rbpaircov
#SBATCH --time=02:00:00
#SBATCH --mem=120G
#SBATCH --cpus-per-task=8
#SBATCH --partition=inter
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/rbpaircov_%j.out

# How many of the pairs the deployed R_blend sums over does the per-pair ruler actually see?
# The radial coverage of the 7" annotation is ~98% (jobs/job_rblend_coverage.sh), but the ap7
# catalogue annotates only 4.16 neighbours per primary inside 7" while the input field holds more.
# This counts the true neighbours from the input positions and compares.
# FIREWALL: half-shear input fields + half-shear ruler dump. No constgold. Nothing trained.
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot

echo "### RULER PAIR COVERAGE  job=$SLURM_JOB_ID ###"; date
python -u scripts/eval_ruler_pair_coverage.py \
  --cases ${CASES:-0 1 2} --sign 0.05 --radius 7.0 \
  --ruler-npz ${RULER_NPZ:-/project/ls-gruen/users/zekang.zhang/sbsi_caches/ablation/eval/rblend_gap_measured_ap7.npz}
echo "RBPAIRCOV_DONE"; date
