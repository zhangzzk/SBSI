#!/bin/bash
#SBATCH --job-name=cgind_sim
#SBATCH --time=08:00:00
#SBATCH --mem=500G
#SBATCH --ntasks-per-node=100
#SBATCH --nodes=1
#SBATCH --partition=cluster
#SBATCH --exclude=usm-cl-183r01,usm-cl-826bac01,usm-cl-826bac02,usm-cl-826bac03,usm-cl-1116cs01,usm-cl-1116cs02,usm-cl-1116cs03,usm-cl-seitz3
#SBATCH --output=/home/z/Zekang.Zhang/logs/cgind_sim_%j.%N.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/cgind_sim_%j.%N.err

set -euo pipefail
[ "$#" -eq 2 ] || { echo "usage: $0 START COUNT"; exit 2; }
START=$1
COUNT=$2
[[ "$START" =~ ^[0-9]+$ && "$COUNT" =~ ^[0-9]+$ ]] || { echo "numeric START/COUNT required"; exit 2; }
END=$((START + COUNT - 1))
(( START >= 40 && END <= 89 && COUNT >= 1 && COUNT <= 25 )) || { echo "invalid chunk $START..$END"; exit 2; }
BE=/home/z/Zekang.Zhang/blendemu
CFG=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot/configs/fs2_lsst_r_constant_indomain_c40-89.yaml
PY=/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python
export CONDA_PREFIX=/project/ls-gruen/users/zekang.zhang/envs/sims1
export PATH="$CONDA_PREFIX/bin:$PATH"
if [[ -f /etc/profile.d/modules.sh ]]; then source /etc/profile.d/modules.sh; fi
module load sextractor
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$BE:${PYTHONPATH:-}"
export BLENDEMU_SIM_RUN=/home/z/Zekang.Zhang/MultiBand_ImSim/modules/Run.py
cd "$BE/scripts"
"$PY" -u run_pipeline.py --config "$CFG" --steps 2 --n-mpi 100 \
  --case-offset "$START" --n-cases "$COUNT"
echo "CONSTGOLD_INDOMAIN_SIM_DONE cases=$START-$END"
