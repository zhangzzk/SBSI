#!/bin/bash
#SBATCH --job-name=rbsummed
#SBATCH --time=00:40:00
#SBATCH --mem=120G
#SBATCH --cpus-per-task=4
#SBATCH --partition=inter
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/rbsummed_%j.out

# Per-PRIMARY SUMMED R_blend (the quantity `m` actually uses) from the per-pair ruler dump.
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
echo "### SUMMED R_BLEND job=$SLURM_JOB_ID ###"; date
python -u scripts/eval_rblend_gap_summed.py ${SEPMAX:+--sep-max $SEPMAX}
echo "---- and restricted to neighbours within 3\" (the nn3 aperture) ----"
python -u scripts/eval_rblend_gap_summed.py --sep-max 3.0
echo RBSUMMED_DONE; date
