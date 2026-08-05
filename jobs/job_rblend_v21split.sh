#!/bin/bash
#SBATCH --job-name=rbsplit
#SBATCH --time=00:40:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=4
#SBATCH --partition=cluster
#SBATCH --output=/home/z/Zekang.Zhang/logs/rbsplit_%j.out
# Tests whether the emulator's exactness on the fiducial domain (-0.02%) is ITSELF a cancellation
# across the V2.1 resolution cut, the way the flow's m turned out to be (2026-08-05j). If the
# emulator is unbiased on both halves, the +1.64% / -2.21% split belongs to the flow; if it flips
# sign, that attribution is premature.
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
E=/project/ls-gruen/users/zekang.zhang/sbsi_caches/ablation/eval
set -e
echo "######## FIDUCIAL-domain ruler, split by the V2.1 resolution cut ########"
python -u scripts/eval_rblend_gap_summed.py \
  --npz $E/rblend_measured_allnbr_FIDdom_ap7g0.2_ladder.npz --v21-split
echo RBSPLIT_DONE
