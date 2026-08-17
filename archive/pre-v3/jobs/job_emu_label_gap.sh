#!/bin/bash
#SBATCH --job-name=emulabel
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=12
#SBATCH --partition=cluster
#SBATCH --output=/home/z/Zekang.Zhang/logs/emulabel_%j.out

# Does the emulator FIT its own training labels at <1", or are the labels themselves the problem?
# Three model-side fixes have failed (domain restriction, close-pair weighting, pair angle); none
# asked whether the emulator's labels agree with the half-shear ruler in the first place. See the
# docstring of scripts/eval_emu_label_gap.py for the two candidate mismatches read off
# blendemu/response.py: the ratio_max=5 / 3" bright-neighbour rejection applied to the labels, and
# the Dg=0.2 label step vs the ruler's Dg=0.05.
# Streams the 27 GB response catalogue batch-by-batch, so memory is small; the cost is I/O + predict.
# FIREWALL: reads the emulator's own training catalogue, never constgold.
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
export XGB_DEVICE=cpu
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
D=/project/ls-gruen/users/zekang.zhang/sbsi_caches/ablation/eval; mkdir -p $D
echo "### EMU vs OWN LABELS job=$SLURM_JOB_ID ###"; date
# CAT=<corrected catalogue> rebins the labels by INPUT-frame separation, the same x-axis the
# ruler is binned in. The first run binned labels in the DETECTED frame and compared them to a
# ruler binned in the input frame, so the two tables were not on the same axis -- see WORKLOG
# 2026-07-28n.
python -u scripts/eval_emu_label_gap.py \
    --tag ${TAG:-lsst_r_extnbr_ho} --stride ${STRIDE:-1} \
    --cat ${CAT:-/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876/response_catalogue_train.feather} \
    --true-re-min 0.3 --true-mag-max 26.0 \
    --output "$D/${OUT:-emu_label_gap}.npz" 2>&1 | grep -v "module command"
echo EMULABEL_DONE; date
