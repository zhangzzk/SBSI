#!/bin/bash
#SBATCH --job-name=v22q3c_sim
#SBATCH --array=0-3
#SBATCH --time=06:00:00
#SBATCH --mem=500G
#SBATCH --ntasks-per-node=64
#SBATCH --nodes=1
#SBATCH --partition=cluster
#SBATCH --exclude=usm-cl-183r01,usm-cl-826bac01,usm-cl-826bac02,usm-cl-826bac03,usm-cl-1116cs01,usm-cl-1116cs02,usm-cl-1116cs03,usm-cl-seitz3
#SBATCH --output=/home/z/Zekang.Zhang/logs/v22q3c_sim_%A_%a.%N.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/v22q3c_sim_%A_%a.%N.err

set -euo pipefail
[ "$#" -eq 2 ] || { echo "usage: $0 START COUNT"; exit 2; }
START=$1
COUNT=$2
[[ "$START" =~ ^[0-9]+$ && "$COUNT" =~ ^[0-9]+$ ]] || { echo "numeric START/COUNT required"; exit 2; }
END=$((START + COUNT - 1))
(( START >= 48 && END <= 139 && COUNT >= 1 && COUNT <= 20 )) || { echo "invalid chunk $START..$END"; exit 2; }
ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
BE=/home/z/Zekang.Zhang/blendemu
PY=/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python
CFG=$ROOT/configs/fs2_lsst_r_v22_constgold_q3_decomp_c48-139.yaml
PREFIX=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_v22cgq3_c48-139
MODES=(total self deployed other)
TASK=${SLURM_ARRAY_TASK_ID:?array task required}
MODE=${MODES[$TASK]}
export PATH="/project/ls-gruen/users/zekang.zhang/envs/sims1/bin:$PATH"
module load sextractor
export LD_LIBRARY_PATH="/project/ls-gruen/users/zekang.zhang/envs/sims1/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$BE:${PYTHONPATH:-}"
export BLENDEMU_SIM_RUN=/home/z/Zekang.Zhang/MultiBand_ImSim/modules/Run.py
cd "$BE/scripts"
"$PY" -u run_pipeline.py --config "$CFG" --steps 2 --n-mpi 64 \
  --n-cases "$COUNT" --case-offset "$START" --output-path "${PREFIX}_${MODE}"
echo "V22_CONSTGOLD_Q3_DECOMP_C100_SIM_DONE mode=$MODE cases=$START-$END"
