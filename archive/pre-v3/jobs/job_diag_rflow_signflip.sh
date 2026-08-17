#!/bin/bash
#SBATCH --job-name=rflowsign
#SBATCH --time=01:30:00
#SBATCH --mem=140G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cluster
#SBATCH --output=/home/z/Zekang.Zhang/logs/rflowsign_%j.out
# WHERE does the flow's closure residual change sign? 2026-08-05l/m: the V2.1 half of the flow's
# training box closes at -1.82% and its complement at +2.86%, so the residual must pass through zero
# somewhere between. Binning INSIDE V2.1 (job 15536811) cannot show that -- the crossing lives at the
# boundary. This bins the WHOLE training box on both axes the V2.1 cut is built from, on the 16
# FIDUCIAL dom6x6 dumps (not the 8 V2.1 ones), and splits the cut into its two conditions to see
# which one carries the split. Residuals only; no m is reported.
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
D=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v2_domain_dumps
CAT=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant/constant_response_catalogue_train.feather
set -e
echo "fiducial dumps present:"; ls -1 $D/ablate_s2c_lt500_dom6x6_perobj_s*.feather | wc -l
python -u scripts/diag_rflow_v21.py --scope train \
  --dump-glob "$D/ablate_s2c_lt500_dom6x6_perobj_s*.feather" \
  --catalogue "$CAT" --min-case 40
echo RFLOWSIGN_DONE
