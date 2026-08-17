#!/bin/bash
#SBATCH --job-name=fig2f2
#SBATCH --time=3:00:00
#SBATCH --mem=250G
#SBATCH --cpus-per-task=12
#SBATCH --partition=inter
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/fig2f2_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/fig2f2_%j.err

# Remake figure 2 with flow #2 as the blend term, alongside the fiducial BlendEMU curve.
#
# WHY A JOB AND NOT THE LOGIN NODE: it reads 16 per-object dumps (16 x 26.9M rows) plus a 9.3 GB
# lookup and stacks 16 seed columns over ~11.7M rows. That is not "negligible work".
#
# WHAT TO CHECK IN THE LOG BEFORE TRUSTING THE PNG:
#   1. "flow #2 R_blend matched" should read ~99.97%. The script REFUSES below 95%.
#   2. The two aggregate m values must reproduce WORKLOG 2026-08-03g (-0.126 / -0.206). They are
#      computed here from scratch on the intersected rows, so a mismatch means the population moved.
#   3. The per-panel rms lines at the end are the quantitative version of the figure -- the primary
#      flux panel is the one to read, since that is where the per-bin ruler says flow #2 is 4.3x
#      worse than BlendEMU.
#
# FIREWALL: constgold is read to SCORE only; nothing here tunes or selects a model.
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot

date
python -u plotting/plot_fig2_flow2.py --out-dir figures || exit 1
echo "FIG2F2_DONE"; date
