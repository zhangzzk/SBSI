#!/bin/bash
#SBATCH --job-name=emu_g005_c200
#SBATCH --time=04:00:00
#SBATCH --mem=180G
#SBATCH --cpus-per-task=32
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/emu_g005_c200_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/emu_g005_c200_%j.err
set -euo pipefail

ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
PY=/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python
BASE=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876
export LD_LIBRARY_PATH="/project/ls-gruen/users/zekang.zhang/envs/sims1/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="/home/z/Zekang.Zhang/blendemu:$ROOT:${PYTHONPATH:-}"
export XGB_DEVICE=cpu
export CONFIG_PATH=$ROOT/configs/fs2_lsst_r_extnbr_v22_g005_c200.yaml
export G005_MODEL_TAG=lsst_r_extnbr_v22_g005_c200
export G005_CATALOGUES="$BASE/response_catalogue_g005_train.feather:$BASE/response_catalogue_g005_c100-199.feather"
export G005_TRAIN_CASE_MIN=0
export G005_TRAIN_CASE_MAX=199
export G005_EXPECTED_TRAIN_CASES=200
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}
META=/home/z/Zekang.Zhang/blendemu/models/emulator_metadata_lsst_r_extnbr_v22_g005_c200.json

[ ! -e "$META" ] || { echo "REFUSING to overwrite $META"; exit 1; }
for catalogue in "$BASE/response_catalogue_g005_train.feather" "$BASE/response_catalogue_g005_c100-199.feather"; do
  [ -s "$catalogue" ] || { echo "missing catalogue $catalogue"; exit 1; }
done
cd "$ROOT"
echo "### TRAIN V2.2 g=0.05 EMULATOR ON CASES 0--199 job=$SLURM_JOB_ID ###"; date
"$PY" -u scripts/retrain_emulator_v22_g005.py
echo EMU_V22_G005_C200_JOB_DONE; date
