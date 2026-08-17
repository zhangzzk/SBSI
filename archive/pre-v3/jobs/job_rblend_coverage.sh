#!/bin/bash
#SBATCH --job-name=rbcover
#SBATCH --time=03:00:00
#SBATCH --mem=120G
#SBATCH --cpus-per-task=8
#SBATCH --partition=inter
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/rbcover_%j.out

# What FRACTION of the deployed R_blend does the per-pair ruler actually certify?
#
# The ruler can only see pairs the catalogue annotates -- the ap7 legs stop at 7.0" -- while the
# R_blend that enters `m` is summed by build_blend_lookup out to the emulator's trained r_max=10".
# This measures the emulator's own summed prediction at r_max = 1/3/5/7/10", all inside the trained
# [0,10] distance cut (so nothing here is extrapolation), and reports coverage(7)/coverage(10).
#
# FIREWALL: half-shear input fields only; constgold is never opened. Model-only, nothing trained.
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
export XGB_DEVICE=cpu
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
D=/project/ls-gruen/users/zekang.zhang/sbsi_caches/rblend_verdict; mkdir -p $D

echo "### RBLEND RULER COVERAGE  job=$SLURM_JOB_ID ###"; date
python -u scripts/eval_rblend_ruler_coverage.py \
  --cases ${CASES:-0 1 2} --sign 0.05 --k 60 --radii 1 3 5 7 10 \
  --tags lsst_r_extnbr_indom_tuned lsst_r_extnbr_ho \
  --ruler-npz ${RULER_NPZ:-/project/ls-gruen/users/zekang.zhang/sbsi_caches/ablation/eval/rblend_gap_measured_ap7.npz} \
  --output $D/rblend_ruler_coverage.csv
echo "RBCOVER_DONE"; date
