#!/bin/bash
#SBATCH --job-name=rfv21flow
#SBATCH --time=01:30:00
#SBATCH --mem=180G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cluster
#SBATCH --output=/home/z/Zekang.Zhang/logs/rfv21flow_%j.out
# THE DECISIVE TEST for the size tilt (WORKLOG 2026-08-05z): the fiducial dom6x6 flow's closure
# residual falls linearly with true size, -8.1 %/arcsec, crossing zero at Re = 0.555 +- 0.072 --
# essentially AT the V2.1 cut of 0.5. Does a flow TRAINED on the V2.1 domain still carry it? If yes,
# the tilt is not a training-coverage effect and retraining cannot fix it.
#
# R_blend MUST be swapped. The V2.1 dumps were built with blend_lookup_v21 (tag lsst_r_extnbr_v21),
# which the ruler convicts at -37.05% +- 2.07 on the full V2.1 sample. Comparing them to the fiducial
# map as-is would compare flow AND emulator at once. --blend-lookup forces both maps onto the same
# _ho R_blend, so the only thing that differs is the flow.
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
CAT=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant/constant_response_catalogue_train.feather
set -e
echo "V2.1-trained flow dumps available:"; ls -1 /project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v21_domain_dumps/ablate_s2c_lt500_v21_perobj_s*.feather | wc -l
echo "(array 15536012 is still filling these; this is a PRELIMINARY read at whatever count exists.)"
python -u scripts/diag_rflow_v21.py --scope train \
  --dump-glob "/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v21_domain_dumps/ablate_s2c_lt500_v21_perobj_s*.feather" \
  --catalogue "$CAT" --min-case 40 \
  --blend-lookup results/blend_lookup_extnbrho_c40-139.feather
echo RFV21FLOW_DONE
