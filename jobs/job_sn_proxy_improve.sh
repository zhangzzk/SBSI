#!/bin/bash
#SBATCH --job-name=snimp
#SBATCH --time=00:30:00
#SBATCH --mem=150G
#SBATCH --cpus-per-task=16
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/snimp_%j.out

# Diagnose WHY m_sel ~ 0 on the magnitude axis and below the PSF: physics, or a dead estimator?
# Measures the moving boundary (who enters/leaves under shear) and the shear response of the cut
# variable itself. No GPU: this touches no model, only the catalogues.
set -o pipefail
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
echo "### SN PROXY IMPROVE job=$SLURM_JOB_ID ###"; date
python -u scripts/eval_sn_proxy_improve.py 2>&1 | grep -v "module command" \
  || { echo SNIMP_FAILED; exit 1; }
echo SNIMP_ALL_DONE; date
