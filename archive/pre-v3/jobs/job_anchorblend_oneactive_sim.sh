#!/bin/bash
#SBATCH --job-name=ab1n_sim
#SBATCH --array=0-1%2
#SBATCH --time=03:00:00
#SBATCH --mem=500G
#SBATCH --ntasks-per-node=100
#SBATCH --nodes=1
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --exclude=usm-cl-183r01,usm-cl-826bac01,usm-cl-826bac02,usm-cl-826bac03,usm-cl-1116cs01,usm-cl-1116cs02,usm-cl-1116cs03,usm-cl-seitz3
#SBATCH --output=/home/z/Zekang.Zhang/logs/ab1n_sim_%A_%a.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/ab1n_sim_%A_%a.err
set -euo pipefail

ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
BE=/home/z/Zekang.Zhang/blendemu
ENV=/project/ls-gruen/users/zekang.zhang/envs/sims1
CFG=$ROOT/configs/fs2_lsst_r_anchorblend_oneactive_orthogonal_c400-499.yaml
TASK=${SLURM_ARRAY_TASK_ID:?array task required}
if [ "$TASK" -eq 0 ]; then MODE=u; else MODE=v; fi
BASE=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_anchorblend_oneactive_${MODE}_c400-499
export PATH="$ENV/bin:$PATH"
module load sextractor
# The module prepends an older GCC runtime.  Restore the conda runtime after
# loading it so pandas sees the GLIBCXX version used to build sims1.
export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$BE:${PYTHONPATH:-}"
export BLENDEMU_SIM_RUN=/home/z/Zekang.Zhang/MultiBand_ImSim/modules/Run.py
cd "$BE/scripts"
python -u run_pipeline.py --config "$CFG" --steps 2 --n-mpi 100 --output-path "$BASE"
for case in $(seq 400 499); do
  for shear in 0.05 -0.05; do
    test -s "$BASE/case${case}_${shear}/real0/catalogues/SExtractor/tile180.0_-0.5_bandr_rot0.feather"
  done
done
echo "ANCHORBLEND_ONEACTIVE_SIM_DONE mode=$MODE"
date
