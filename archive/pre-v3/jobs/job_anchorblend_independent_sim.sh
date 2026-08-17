#!/bin/bash
#SBATCH --job-name=ab_indsim
#SBATCH --time=04:00:00
#SBATCH --mem=500G
#SBATCH --ntasks-per-node=100
#SBATCH --nodes=1
#SBATCH --partition=cluster
#SBATCH --exclude=usm-cl-183r01,usm-cl-826bac01,usm-cl-826bac02,usm-cl-826bac03,usm-cl-1116cs01,usm-cl-1116cs02,usm-cl-1116cs03,usm-cl-seitz3
#SBATCH --output=/home/z/Zekang.Zhang/logs/ab_indsim_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/ab_indsim_%j.err
set -euo pipefail

ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
SIMS=/project/ls-gruen/users/zekang.zhang/envs/sims1
BASE=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_anchorblend_independent_local10_c200-299
CONFIG=$ROOT/configs/fs2_lsst_r_anchorblend_independent_local10_c200-299.yaml
export PATH=$SIMS/bin:$PATH
module load sextractor
export LD_LIBRARY_PATH=$SIMS/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}
export BLENDEMU_SIM_RUN=/home/z/Zekang.Zhang/MultiBand_ImSim/modules/Run.py
cd /home/z/Zekang.Zhang/blendemu/scripts
python -u run_pipeline.py --config "$CONFIG" --steps 2 --n-mpi 100
for case in $(seq 200 299); do
  for shear in 0.05 -0.05; do
    test -s "$BASE/case${case}_${shear}/real0/images/original/tile180.0_-0.5_bandr_rot0.fits"
    test -s "$BASE/case${case}_${shear}/real0/catalogues/SExtractor/tile180.0_-0.5_bandr_rot0.feather"
  done
done
echo ANCHORBLEND_INDEPENDENT_SIM_DONE
date
