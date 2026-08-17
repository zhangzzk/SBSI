#!/bin/bash
#SBATCH --job-name=abep_sim
#SBATCH --array=0-18%6
#SBATCH --time=03:00:00
#SBATCH --mem=300G
#SBATCH --ntasks-per-node=32
#SBATCH --nodes=1
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --exclude=usm-cl-183r01,usm-cl-826bac01,usm-cl-826bac02,usm-cl-826bac03,usm-cl-1116cs01,usm-cl-1116cs02,usm-cl-1116cs03,usm-cl-seitz3
#SBATCH --output=/home/z/Zekang.Zhang/logs/abep_sim_%A_%a.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/abep_sim_%A_%a.err
set -euo pipefail

ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
BE=/home/z/Zekang.Zhang/blendemu
ENV=/project/ls-gruen/users/zekang.zhang/envs/sims1
CFG=$ROOT/configs/fs2_lsst_r_anchorblend_eachpair_g1_pilot_c400-409.yaml
PREFIX=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_anchorblend_eachpair_g1
TASK=${SLURM_ARRAY_TASK_ID:?array task required}
BASE=$(printf '%s_rank%02d_pilot_c400-409' "$PREFIX" "$TASK")
export PATH="$ENV/bin:$PATH"
module load sextractor
export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$BE:${PYTHONPATH:-}"
export BLENDEMU_SIM_RUN=/home/z/Zekang.Zhang/MultiBand_ImSim/modules/Run.py
cd "$BE/scripts"
python -u run_pipeline.py --config "$CFG" --steps 2 --n-mpi 32 --output-path "$BASE"
for case in $(seq 400 409); do
  for shear in 0.05 -0.05; do
    test -s "$BASE/case${case}_${shear}/real0/catalogues/SExtractor/tile180.0_-0.5_bandr_rot0.feather"
  done
done
echo "ANCHORBLEND_EACHPAIR_G1_SIM_DONE rank=$TASK"
date
