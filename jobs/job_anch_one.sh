#!/bin/bash
#SBATCH --job-name=anch1
#SBATCH --time=01:00:00
#SBATCH --mem=110G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cluster
#SBATCH --output=/home/z/Zekang.Zhang/logs/anch1_%j.out
set -o pipefail
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
T=/home/z/Zekang.Zhang/SBSI/results/response_target_crowd_rblend_snc_c0-99_6x6x5_dom.npz
D=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/indist_constgold_dumps
echo "### ANCHOR ARM ${TAG} job=$SLURM_JOB_ID ###"; date
echo "PRE-REGISTERED: training-count anchor pins <R_flow> to 0.7149 -> in-domain m about +0.94% [+0.4,+1.5]"
echo "                constgold-occupancy anchor pins to 0.6957 -> about +3.27%"
echo "                baseline (no anchor) dom6x6 = -0.508%"
python -u scripts/eval_target_vs_constgold.py --dump-dir "$D" --tag "${TAG}" --target "$T" \
  2>&1 | grep -v --line-buffered "module command" | head -30
echo ANCH1_DONE; date
