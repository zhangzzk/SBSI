#!/bin/bash
#SBATCH --job-name=anti_sim
#SBATCH --time=16:00:00
#SBATCH --mem=500G
#SBATCH --ntasks-per-node=100
#SBATCH --nodes=1
#SBATCH --partition=cluster
#SBATCH --exclude=usm-cl-183r01,usm-cl-826bac01,usm-cl-826bac02,usm-cl-826bac03,usm-cl-1116cs01,usm-cl-1116cs02,usm-cl-1116cs03,usm-cl-seitz3
#SBATCH --output=/home/z/Zekang.Zhang/logs/anti_sim_%j.%N.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/anti_sim_%j.%N.err

# Render/detect the -0.02 leg. This is additive: case*_0.02 and all existing
# catalogues are untouched. The submit wrapper refuses if a -0.02 tree exists.
set -euo pipefail
SIMS=/project/ls-gruen/users/zekang.zhang/envs/sims1
export CONDA_PREFIX="$SIMS"
export PATH="$SIMS/bin:$PATH"
module load sextractor
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}"
export BLENDEMU_SIM_RUN="/home/z/Zekang.Zhang/MultiBand_ImSim/modules/Run.py"
cd /home/z/Zekang.Zhang/blendemu/scripts
CFG=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot/configs/fs2_lsst_r_antithetic_self_gm002.yaml

echo "ANTITHETIC SELF SIM job=$SLURM_JOB_ID"; date
python -u run_pipeline.py --config "$CFG" --steps 2 --n-mpi 100
echo ANTITHETIC_SELF_SIM_DONE; date
