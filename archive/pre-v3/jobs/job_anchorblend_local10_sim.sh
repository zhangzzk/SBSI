#!/bin/bash
#SBATCH --job-name=ab_l10sim
#SBATCH --time=16:00:00
#SBATCH --mem=500G
#SBATCH --ntasks-per-node=100
#SBATCH --nodes=1
#SBATCH --partition=cluster
#SBATCH --exclude=usm-cl-183r01,usm-cl-826bac01,usm-cl-826bac02,usm-cl-826bac03,usm-cl-1116cs01,usm-cl-1116cs02,usm-cl-1116cs03,usm-cl-seitz3
#SBATCH --output=/home/z/Zekang.Zhang/logs/ab_l10sim_%j.%N.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/ab_l10sim_%j.%N.err
set -euo pipefail

SIMS=/project/ls-gruen/users/zekang.zhang/envs/sims1
export PATH="$SIMS/bin:$PATH"
module load sextractor
export LD_LIBRARY_PATH="$SIMS/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}"
export BLENDEMU_SIM_RUN=/home/z/Zekang.Zhang/MultiBand_ImSim/modules/Run.py
cd /home/z/Zekang.Zhang/blendemu/scripts
BASE=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_anchorblend_g005_local10_c200-299
for case in $(seq 200 299); do
  for shear in 0.05 -0.05; do
    cfg="$BASE/sim_config_case${case}_${shear}.ini"
    [ -s "$cfg" ] || { echo "Missing simulation config: $cfg"; exit 1; }
  done
done
python -u run_pipeline.py \
  --config /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot/configs/fs2_lsst_r_anchorblend_g005_local10_c200-299.yaml \
  --steps 2 --n-mpi 100

# run_sim.py currently logs child failures without returning them.  Require the
# two concrete products needed downstream for every case/shear leg.
missing=0
for case in $(seq 200 299); do
  for shear in 0.05 -0.05; do
    leg="$BASE/case${case}_${shear}/real0"
    image="$leg/images/original/tile180.0_-0.5_bandr_rot0.fits"
    detection="$leg/catalogues/SExtractor/tile180.0_-0.5_bandr_rot0.feather"
    [ -s "$image" ] && [ -s "$detection" ] || {
      echo "Incomplete simulation leg: case=$case shear=$shear"
      missing=$((missing + 1))
    }
  done
done
[ "$missing" -eq 0 ] || { echo "Incomplete simulation legs: $missing"; exit 1; }

echo ANCHORBLEND_LOCAL10_SIM_JOB_DONE; date
