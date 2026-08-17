#!/bin/bash
#SBATCH --job-name=emu_indom_v3b
#SBATCH --time=01:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/emu_indom_v3b_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/emu_indom_v3b_%j.err

# Bridge test for the proposed V3 response-weighted loss on the historical domain.
# Everything in the loss recipe is fixed from the V2.2/V3 milestone; only the source
# model/configuration changes to the old r<26, Re>0.3 arcsec setup.  All 200 half-shear
# cases are training input.  Constgold is not read here.
set -euo pipefail
PY=/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python
export LD_LIBRARY_PATH=/project/ls-gruen/users/zekang.zhang/envs/sims1/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=/home/z/Zekang.Zhang/blendemu:/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:${PYTHONPATH:-}
export XGB_DEVICE=cpu
export CONFIG_PATH=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot/configs/fs2_lsst_r_extnbr_indom_tuned.yaml
export HELDOUT_MIN_CASE=0
export RESPONSE_WEIGHT_SOURCE_TAG=lsst_r_extnbr_indom_tuned
export RESPONSE_WEIGHT_OUTPUT_TAG=lsst_r_extnbr_indom_tuned_rpowposa0065_all200
export RESPONSE_WEIGHT_ALPHA=0.065
export RESPONSE_WEIGHT_CAP=50
export RESPONSE_WEIGHT_MODE=positive_square
export RESPONSE_WEIGHT_N_TREES=271
export RESPONSE_WEIGHT_TOKEN=posa0065_all200
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
"$PY" -u scripts/retrain_emulator_v22_response_weighted.py
