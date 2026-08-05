#!/bin/bash
#SBATCH --job-name=absm_sim
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --ntasks-per-node=2
#SBATCH --nodes=1
#SBATCH --partition=cluster
#SBATCH --output=/home/z/Zekang.Zhang/logs/absm_sim_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/absm_sim_%j.err
set -euo pipefail
SIMS=/project/ls-gruen/users/zekang.zhang/envs/sims1
export PATH="$SIMS/bin:$PATH"
module load sextractor
export LD_LIBRARY_PATH="$SIMS/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}"
export BLENDEMU_SIM_RUN=/home/z/Zekang.Zhang/MultiBand_ImSim/modules/Run.py
cd /home/z/Zekang.Zhang/blendemu/scripts
python -u run_pipeline.py --config /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot/configs/fs2_lsst_r_anchorblend_smoke.yaml --steps 2 --n-mpi 2
echo ANCHORBLEND_SMOKE_SIM_DONE
