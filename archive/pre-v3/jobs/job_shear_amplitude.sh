#!/bin/bash
#SBATCH --job-name=shamp
#SBATCH --time=03:00:00
#SBATCH --mem=250G
#SBATCH --cpus-per-task=12
#SBATCH --partition=inter
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/shamp_%j.out

# THE decisive test for the -41.5% close-pair deficit. Jobs 15328417/15328418 established that the
# emulator fits its own labels at close separation and that the bright-neighbour rejection explains
# nothing -- so the deficit is a disagreement between the LABELS (Dg=0.2) and the RULER (Dg=0.05).
# Both come from the SAME sim suite and the same ngmix estimator, so the two finite differences can
# be compared PAIR BY PAIR, which removes every population argument at once.
# See scripts/eval_shear_amplitude.py for how the join is keyed.
# FIREWALL: training catalogue + half-shear legs; constgold is never read.
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
export XGB_DEVICE=cpu
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
D=/project/ls-gruen/users/zekang.zhang/sbsi_caches/ablation/eval; mkdir -p $D
echo "### SHEAR AMPLITUDE (same pairs, Dg=0.2 vs Dg=0.05) job=$SLURM_JOB_ID ###"; date
python -u scripts/eval_shear_amplitude.py \
    --true-re-min 0.3 --true-mag-max 26.0 --max-dist ${MAXDIST:-5.0} \
    --tag ${TAG:-lsst_r_extnbr_ho} \
    --output "$D/shear_amplitude.npz" 2>&1 | grep -v "module command"
echo SHAMP_DONE; date
