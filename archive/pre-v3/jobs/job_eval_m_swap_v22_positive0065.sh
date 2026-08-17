#!/bin/bash
#SBATCH --job-name=mswap_v22_pos065
#SBATCH --time=01:30:00
#SBATCH --mem=110G
#SBATCH --cpus-per-task=4
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/mswap_v22_pos065_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/mswap_v22_pos065_%j.err
set -euo pipefail
PY=/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python
ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
export LD_LIBRARY_PATH=/project/ls-gruen/users/zekang.zhang/envs/sims1/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=$ROOT:/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}
cd "$ROOT"
"$PY" -u scripts/eval_m_swap_emulator_v22.py \
  --dump-glob '/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v22_cip_domain_dumps/ablate_s2c_lt500_v22_perobj_s*.feather' \
  --catalogue /project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant/constant_response_catalogue_train.feather \
  --lookup results/blend_lookup_v22_rpowposa0065_c40-139.feather \
  --candidate-tag lsst_r_extnbr_v22_rpowposa0065 \
  --flow-checkpoint /project/ls-gruen/users/zekang.zhang/sbsi_caches/ablation/measurement_flow_g0_ngmix_ablate_s2c_lt500_v22_s501_swaavg.pt \
  --output results/constgold_m_swap_v22_rpowposa0065_16seed.json
echo V22_POSITIVE0065_M_SWAP_DONE
