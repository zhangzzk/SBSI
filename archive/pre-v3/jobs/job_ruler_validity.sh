#!/bin/bash
#SBATCH --job-name=rulerval
#SBATCH --time=02:00:00
#SBATCH --mem=250G
#SBATCH --cpus-per-task=12
#SBATCH --partition=inter
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/rulerval_%j.out

# Clear (or convict) the RULER at <1", where 31% of the close-pair deficit survives the distance fix.
# The 45-degree null test has only ever been quoted GLOBALLY; a bias confined to the 17% of pairs
# below 1" would be diluted ~6x in it. This runs the null test per separation bin, and checks whether
# the deficit depends on cross-match quality (deblending inconsistency between legs would inflate the
# measured shape difference at exactly these separations).
# FIREWALL: half-shear legs only; constgold is never read.
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
export XGB_DEVICE=cpu
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
D=/project/ls-gruen/users/zekang.zhang/sbsi_caches/ablation/eval; mkdir -p $D
echo "### RULER VALIDITY job=$SLURM_JOB_ID ###"; date
python -u scripts/eval_ruler_validity.py --tag ${TAG:-lsst_r_extnbr_indist} \
    --true-re-min 0.3 --true-mag-max 26.0 \
    --output "$D/ruler_validity.npz" 2>&1 | grep -v "module command"
echo RULERVAL_DONE; date
