#!/bin/bash
# Quarantined recovery for the final, missing V2-complement image block.
# The V2 implementation remains under archive/pre-v3; this wrapper does not
# make it part of the supported V3 API.
#SBATCH --job-name=lv2asrec
#SBATCH --time=03:00:00
#SBATCH --mem=500G
#SBATCH --ntasks-per-node=100
#SBATCH --nodes=1
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --exclude=usm-cl-183r01,usm-cl-826bac01,usm-cl-826bac02,usm-cl-826bac03,usm-cl-1116cs01,usm-cl-1116cs02,usm-cl-1116cs03,usm-cl-seitz3
#SBATCH --output=/home/z/Zekang.Zhang/logs/lv2asrec_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/lv2asrec_%j.err
set -euo pipefail

ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
LEGACY=$ROOT/archive/pre-v3
BE=/home/z/Zekang.Zhang/blendemu
ENV=/project/ls-gruen/users/zekang.zhang/envs/sims1
START=800
STOP=899
CFG=$LEGACY/configs/fs2_lsst_r_anchorblend_g002_c800-899.yaml
BASE=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_anchorblend_g002_v2complement_c800-899

export PATH="$ENV/bin:$PATH"
export PYTHONPATH="$LEGACY:$LEGACY/scripts:$BE:${PYTHONPATH:-}"
export BLENDEMU_SIM_RUN=/home/z/Zekang.Zhang/MultiBand_ImSim/modules/Run.py
module load sextractor
export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"
cd "$BE/scripts"
python -u run_pipeline.py --config "$CFG" --steps 2 --n-mpi 100 --output-path "$BASE"
for case in $(seq "$START" "$STOP"); do
  for shear in 0.02 -0.02; do
    test -s "$BASE/case${case}_${shear}/real0/catalogues/SExtractor/tile180.0_-0.5_bandr_rot0.feather"
  done
done
echo "LEGACY_V2_ANCHOR_SIM_RECOVERY_DONE cases=$START-$STOP"
