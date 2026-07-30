#!/bin/bash
#SBATCH --job-name=cutscan
#SBATCH --time=01:30:00
#SBATCH --mem=140G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cluster
#SBATCH --output=/home/z/Zekang.Zhang/logs/cutscan_%j.out
# Where in TRUE-property cut space is |m|<=0.3% actually met? The boundary size bin is the one place
# the pin residual and target defect ALIGN instead of cancelling, so the acceptance number is unusually
# sensitive to the lower size edge. CPU-only from existing dumps.
set -o pipefail
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
echo "### CUT SCAN job=$SLURM_JOB_ID ###"; date
for T in ablate_s2c_lt500_dom6x6 ablate_s2c_coupling_lt500_dom2; do
  echo; echo "############ $T ############"
  python -u scripts/eval_cut_scan.py --tag $T 2>&1 | grep -v --line-buffered "module command" \
    || { echo "FAILED $T"; exit 1; }
done
echo CUTSCAN_DONE; date
