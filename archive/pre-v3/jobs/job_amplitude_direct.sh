#!/bin/bash
#SBATCH --job-name=ampdir
#SBATCH --time=02:00:00
#SBATCH --mem=250G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cluster
#SBATCH --output=/home/z/Zekang.Zhang/logs/ampdir_%j.out

# THE amplitude test, done directly: the same pairs differenced at Dg=0.2 and at Dg=0.05.
# Possible only because the g=0.05 legs exist on disk, so the response catalogue could be REBUILT at
# small shear from existing shape/cross-match files (job 15329737) -- same galaxies, same estimator,
# same pairing, same rejection, same unsheared reference leg. See scripts/eval_amplitude_direct.py.
# FIREWALL: response catalogues only; constgold is never touched.
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
D=/project/ls-gruen/users/zekang.zhang/sbsi_caches/ablation/eval; mkdir -p $D
echo "### AMPLITUDE DIRECT (Dg=0.2 vs Dg=0.05, same pairs) job=$SLURM_JOB_ID ###"; date
python -u scripts/eval_amplitude_direct.py --max-dist ${MAXDIST:-3.0} --max-case ${MAXCASE:-100} \
    --true-re-min 0.3 --true-mag-max 26.0 \
    --output "$D/amplitude_direct.npz" 2>&1 | grep -v "module command"
echo AMPDIR_DONE; date
