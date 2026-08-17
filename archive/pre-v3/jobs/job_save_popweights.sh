#!/bin/bash
#SBATCH --job-name=popw
#SBATCH --time=01:00:00
#SBATCH --mem=110G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cluster
#SBATCH --output=/home/z/Zekang.Zhang/logs/popw_%j.out
# Write the DELIVERABLE population's per-cell occupancy for the global anchor. Occupancy only -- no
# measured response, no r_sim, no m -- so training stays firewall-clean.
set -o pipefail
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
echo "### SAVE POP WEIGHTS job=$SLURM_JOB_ID ###"; date
python -u scripts/eval_population_reweight.py \
  --tag ablate_s2c_lt500_dom6x6 \
  --target /home/z/Zekang.Zhang/SBSI/results/response_target_crowd_rblend_snc_c0-99_6x6x5_dom.npz \
  --save-weights /home/z/Zekang.Zhang/SBSI/results/popw_indomain_6x6x5.npz \
  2>&1 | grep -v --line-buffered "module command" || { echo FAILED; exit 1; }
echo POPW_DONE; date
