#!/bin/bash
#SBATCH --job-name=v21cancel
#SBATCH --time=01:30:00
#SBATCH --mem=140G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cluster
#SBATCH --output=/home/z/Zekang.Zhang/logs/v21cancel_%j.out
# Tests whether V2.1's +1.6% m is a NEW bias or the fiducial's known bias with its cancelling
# population removed. Uses the 16 FIDUCIAL dumps, so the m it reports is spec-compliant.
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
D=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v2_domain_dumps
CAT=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant/constant_response_catalogue_train.feather
set -e
python -u scripts/diag_v21_cancellation.py \
  --dump-glob "$D/ablate_s2c_lt500_dom6x6_perobj_s*.feather" --catalogue "$CAT" --min-case 40
echo V21CANCEL_DONE
