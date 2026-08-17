#!/bin/bash
#SBATCH --job-name=ab_higha_xfer_an
#SBATCH --time=00:30:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/ab_higha_xfer_an_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/ab_higha_xfer_an_%j.err
set -euo pipefail

ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
ENV=/project/ls-gruen/users/zekang.zhang/envs/sims1
PARTS=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/anchor_weak_weight_transfer_g002_c400-899
OUT=$ROOT/results/anchorblend_response_alpha_scan_transfer_v22_g002_c400-899
export PATH=$ENV/bin:$PATH
export LD_LIBRARY_PATH=$ENV/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=$ROOT:$ROOT/scripts:${PYTHONPATH:-}
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
cd "$ROOT"
python -u scripts/analyze_anchor_response_weak_weight_transfer.py \
  --parts-root "$PARTS" \
  --model-dir /home/z/Zekang.Zhang/blendemu/models \
  --tags \
    lsst_r_extnbr_v22_rpowposw0035 \
    lsst_r_extnbr_v22_rpowposw0050 \
    lsst_r_extnbr_v22_rpowposa010 \
    lsst_r_extnbr_v22_rpowposw0200 \
  --model-names alpha_0p035 alpha_0p050 alpha_0p100 alpha_0p200 \
  --case-min 400 \
  --case-max 899 \
  --previously-inspected \
  --output "$OUT"
echo ANCHOR_RESPONSE_HIGH_ALPHA_TRANSFER_ANALYSIS_DONE
