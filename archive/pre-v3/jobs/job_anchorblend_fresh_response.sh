#!/bin/bash
#SBATCH --job-name=abfr_resp
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/abfr_resp_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/abfr_resp_%j.err
set -euo pipefail
ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
ENV=/project/ls-gruen/users/zekang.zhang/envs/sims1
BASE=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_anchorblend_g005_c300-399
OUT=$ROOT/results/anchorblend_g005_response_v22_c300-399.feather
export PATH=$ENV/bin:$PATH
export LD_LIBRARY_PATH=$ENV/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=$ROOT:/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
test ! -e "$OUT" || { echo "REFUSING existing $OUT"; exit 1; }
cd "$ROOT"
python -u scripts/build_anchorblend_response.py \
  --base "$BASE" --cases $(seq 300 399) --g 0.05 \
  --tags lsst_r_extnbr_v22 --output "$OUT"
echo ANCHORBLEND_FRESH_RESPONSE_DONE
date
