#!/bin/bash
#SBATCH --job-name=ab_pc244
#SBATCH --time=00:20:00
#SBATCH --mem=12G
#SBATCH --cpus-per-task=2
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/ab_pc244_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/ab_pc244_%j.err
set -euo pipefail
ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
ENV=/project/ls-gruen/users/zekang.zhang/envs/sims1
export PATH=$ENV/bin:$PATH
export LD_LIBRARY_PATH=$ENV/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=/home/z/Zekang.Zhang/blendemu:$ROOT:$ROOT/scripts:${PYTHONPATH:-}
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
cd "$ROOT"
python -u scripts/rescore_anchor_independent_pair_cap.py \
  --base /project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_anchorblend_independent_local10_c200-299 \
  --measured-dir /project/ls-gruen/users/zekang.zhang/sbsi_gap_anchor_independent_fast_c200-299 \
  --output-dir /project/ls-gruen/users/zekang.zhang/sbsi_gap_anchor_independent_paircap64_c200-299 \
  --case-offset 244 --n-cases 1 --pair-k 64
python -u scripts/rescore_anchor_coherent_pair_cap.py \
  --base /project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_anchorblend_g005_local10_c200-299 \
  --reference results/anchorblend_g005_response_v22_c100-299.feather \
  --output-dir /project/ls-gruen/users/zekang.zhang/sbsi_gap_anchor_coherent_paircap64_c200-299 \
  --case-offset 244 --n-cases 1 --wide-k 64
echo ANCHORBLEND_PAIRCAP_244_RECOVERY_DONE
date
