#!/bin/bash
#SBATCH --job-name=mag_sn10
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --partition=cluster
#SBATCH --output=/home/z/Zekang.Zhang/logs/mag_sn10_%j.out

set -euo pipefail
SIMS=/project/ls-gruen/users/zekang.zhang/envs/sims1
export PATH="$SIMS/bin:$PATH"
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}"
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot

for f in results/true_mag_before_after_sn10.png \
         results/true_mag_before_after_sn10.pdf \
         results/true_mag_before_after_sn10.json; do
  test ! -e "$f" || { echo "REFUSING: output exists: $f"; exit 1; }
done

python -u scripts/plot_true_mag_sn_cut.py
echo TRUE_MAG_SN_HIST_DONE
