#!/bin/bash
#SBATCH --job-name=bfagg
#SBATCH --time=02:00:00
#SBATCH --mem=180G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/bfagg_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/bfagg_%j.err

# Aggregate the 16-seed flow-#2 ensemble (WORKLOG 2026-08-02m). CPU only -- it reads 16 npz of
# 10.3M rows and forms binned statistics; no model is evaluated. Submitted rather than run on the
# login node because 16 x 10.3M x 5 metrics is not negligible work.
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
CACHE=/project/ls-gruen/users/zekang.zhang/sbsi_caches/blendflow
python -u scripts/agg_blendflow_ensemble.py --glob "$CACHE/eval_ens_s*.npz" --expect 16 \
  || { echo "AGG FAILED"; exit 1; }
echo "BFAGG_DONE"; date
