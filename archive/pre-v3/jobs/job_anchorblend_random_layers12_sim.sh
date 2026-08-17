#!/bin/bash
#SBATCH --job-name=ab_rlsim
#SBATCH --array=0-3
#SBATCH --time=16:00:00
#SBATCH --mem=500G
#SBATCH --ntasks-per-node=100
#SBATCH --nodes=1
#SBATCH --partition=cluster
#SBATCH --exclude=usm-cl-183r01,usm-cl-826bac01,usm-cl-826bac02,usm-cl-826bac03,usm-cl-1116cs01,usm-cl-1116cs02,usm-cl-1116cs03,usm-cl-seitz3
#SBATCH --output=/home/z/Zekang.Zhang/logs/ab_rlsim_%A_%a.%N.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/ab_rlsim_%A_%a.%N.err
set -euo pipefail
TASK=${SLURM_ARRAY_TASK_ID:?array task required}; LAYER=$((TASK / 2 + 1))
if (( TASK % 2 == 0 )); then RADIUS=10; else RADIUS=15; fi
ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
SIMS=/project/ls-gruen/users/zekang.zhang/envs/sims1
BASE=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_anchorblend_random_layer${LAYER}_local${RADIUS}_c200-299
CONFIG=$ROOT/configs/fs2_lsst_r_anchorblend_random_layer${LAYER}_local${RADIUS}_c200-299.yaml
export PATH="$SIMS/bin:$PATH"; module load sextractor
export LD_LIBRARY_PATH="$SIMS/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}"
export BLENDEMU_SIM_RUN=/home/z/Zekang.Zhang/MultiBand_ImSim/modules/Run.py
cd /home/z/Zekang.Zhang/blendemu/scripts
for case in $(seq 200 299); do for shear in 0.05 -0.05; do
  cfg="$BASE/sim_config_case${case}_${shear}.ini"; [ -s "$cfg" ] || { echo "Missing $cfg"; exit 1; }
done; done
python -u run_pipeline.py --config "$CONFIG" --steps 2 --n-mpi 100
missing=0
for case in $(seq 200 299); do for shear in 0.05 -0.05; do
  leg="$BASE/case${case}_${shear}/real0"
  [ -s "$leg/images/original/tile180.0_-0.5_bandr_rot0.fits" ] && \
  [ -s "$leg/catalogues/SExtractor/tile180.0_-0.5_bandr_rot0.feather" ] || missing=$((missing + 1))
done; done
[ "$missing" -eq 0 ] || { echo "Incomplete simulation legs: $missing"; exit 1; }
echo "ANCHORBLEND_RANDOM_LAYER${LAYER}_LOCAL${RADIUS}_SIM_DONE"; date
