#!/bin/bash
#SBATCH --job-name=abx_sim
#SBATCH --time=24:00:00
#SBATCH --mem=500G
#SBATCH --ntasks-per-node=100
#SBATCH --nodes=1
#SBATCH --partition=cluster
#SBATCH --exclude=usm-cl-183r01,usm-cl-826bac01,usm-cl-826bac02,usm-cl-826bac03,usm-cl-1116cs01,usm-cl-1116cs02,usm-cl-1116cs03,usm-cl-seitz3
#SBATCH --output=/home/z/Zekang.Zhang/logs/abx_sim_%j.%N.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/abx_sim_%j.%N.err
set -euo pipefail
SIMS=/project/ls-gruen/users/zekang.zhang/envs/sims1
export PATH="$SIMS/bin:$PATH"
module load sextractor
export LD_LIBRARY_PATH="$SIMS/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}"
export BLENDEMU_SIM_RUN=/home/z/Zekang.Zhang/MultiBand_ImSim/modules/Run.py
cd /home/z/Zekang.Zhang/blendemu/scripts
python -u run_pipeline.py --config /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot/configs/fs2_lsst_r_anchorblend_g005_ext.yaml --steps 2 --n-mpi 100
echo ANCHORBLEND_EXT_SIM_DONE
