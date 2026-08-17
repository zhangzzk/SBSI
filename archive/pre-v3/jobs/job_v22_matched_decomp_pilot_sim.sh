#!/bin/bash
#SBATCH --job-name=v22dc_sim
#SBATCH --array=0-4
#SBATCH --time=12:00:00
#SBATCH --mem=500G
#SBATCH --ntasks-per-node=64
#SBATCH --nodes=1
#SBATCH --partition=cluster
#SBATCH --exclude=usm-cl-183r01,usm-cl-826bac01,usm-cl-826bac02,usm-cl-826bac03,usm-cl-1116cs01,usm-cl-1116cs02,usm-cl-1116cs03,usm-cl-seitz3
#SBATCH --output=/home/z/Zekang.Zhang/logs/v22dc_sim_%A_%a.%N.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/v22dc_sim_%A_%a.%N.err

set -euo pipefail
ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
BE=/home/z/Zekang.Zhang/blendemu
PY=/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python
CFG=$ROOT/configs/fs2_lsst_r_v22_matched_decomp_pilot.yaml
PREFIX=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_v22decomp_pilot
MODES=(total self neighbour pair0 pair1)
TASK=${SLURM_ARRAY_TASK_ID:?array task required}
MODE=${MODES[$TASK]}
export PATH="/project/ls-gruen/users/zekang.zhang/envs/sims1/bin:$PATH"
module load sextractor
export LD_LIBRARY_PATH="/project/ls-gruen/users/zekang.zhang/envs/sims1/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$BE:${PYTHONPATH:-}"
export BLENDEMU_SIM_RUN=/home/z/Zekang.Zhang/MultiBand_ImSim/modules/Run.py
cd "$BE/scripts"
"$PY" -u run_pipeline.py --config "$CFG" --steps 2 --n-mpi 64 \
  --output-path "${PREFIX}_${MODE}"
echo "V22_MATCHED_DECOMP_PILOT_SIM_DONE mode=$MODE"

