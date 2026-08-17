#!/bin/bash
#SBATCH --job-name=contrcut
#SBATCH --time=03:00:00
#SBATCH --mem=250G
#SBATCH --cpus-per-task=12
#SBATCH --partition=inter
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/contrcut_%j.out

# Apply blendemu's OWN bright-neighbour rejection (ratio_max=5 within 3", from
# blendemu/response.py::retrieve_response) to the half-shear ruler and see whether the -41.5%
# close-pair deficit collapses. If it does, the deficit is a POPULATION mismatch between the
# emulator's labels and the evaluation sample -- which no retraining on those labels could fix.
# Needs the whole detection field per case (KDTree over every detected object) -> large memory.
# FIREWALL: half-shear legs only; constgold is never read.
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
export XGB_DEVICE=cpu
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
D=/project/ls-gruen/users/zekang.zhang/sbsi_caches/ablation/eval; mkdir -p $D
echo "### CONTRAST CUT job=$SLURM_JOB_ID ###"; date
python -u scripts/eval_contrast_cut.py \
    --true-re-min 0.3 --true-mag-max 26.0 --tag ${TAG:-lsst_r_extnbr_ho} \
    --output "$D/contrast_cut.npz" 2>&1 | grep -v "module command"
echo CONTRCUT_DONE; date
