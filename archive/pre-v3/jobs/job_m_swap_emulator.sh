#!/bin/bash
#SBATCH --job-name=mswapemu
#SBATCH --time=01:30:00
#SBATCH --mem=180G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cluster
#SBATCH --output=/home/z/Zekang.Zhang/logs/mswapemu_%j.out
# On the FULL V2.1 population the ruler convicts the fiducial emulator (-15.56% +- 2.78, 5.6 sigma)
# and clears lsst_r_extnbr_ho (+1.18% +- 3.33), whose regression box (mag 18-28, Re 0.1-1.5) covers
# V2.1 outright while the fiducial box (mag 18-26) does not -- V2.1's S/N curve has no magnitude
# ceiling. That is the ruler-based promotion argument AGENTS.md requires. This applies the swap to
# the reported m WITHOUT a GPU re-score: the dumps store R_flow and R_blend separately, so only
# R_blend has to be re-joined. 16 fiducial seeds, ratio formed inside each seed.
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
D=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v2_domain_dumps
CAT=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant/constant_response_catalogue_train.feather
set -e
python -u scripts/eval_m_swap_emulator.py \
  --dump-glob "$D/ablate_s2c_lt500_dom6x6_perobj_s*.feather" \
  --catalogue "$CAT" --min-case 40 \
  --lookup results/blend_lookup_extnbrho_c40-139.feather \
  --lookup-label "lsst_r_extnbr_ho"
echo MSWAPEMU_DONE
