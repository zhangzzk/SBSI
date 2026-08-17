#!/bin/bash
#SBATCH --job-name=emu_global_mean
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --array=0-1
#SBATCH --output=/home/z/Zekang.Zhang/logs/emu_global_mean_%A_%a.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/emu_global_mean_%A_%a.err
set -euo pipefail

PY=/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python
export LD_LIBRARY_PATH=/project/ls-gruen/users/zekang.zhang/envs/sims1/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=/home/z/Zekang.Zhang/blendemu:/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:${PYTHONPATH:-}
export HELDOUT_MIN_CASE=0
export XGB_DEVICE=cpu
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
mkdir -p results

if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then
  CONFIG=configs/fs2_lsst_r_extnbr_v22.yaml
  SOURCE=lsst_r_extnbr_v22
  CANDIDATE=lsst_r_extnbr_v22_rpowposa0065_all200
  OUTPUT=results/emulator_global_mean_v22_rpowposa0065_all200.json
elif [ "$SLURM_ARRAY_TASK_ID" -eq 1 ]; then
  CONFIG=configs/fs2_lsst_r_extnbr_indom_tuned.yaml
  SOURCE=lsst_r_extnbr_indom_tuned
  CANDIDATE=lsst_r_extnbr_indom_tuned_rpowposa0065_all200
  OUTPUT=results/emulator_global_mean_indom_rpowposa0065_all200.json
else
  echo "unexpected array index $SLURM_ARRAY_TASK_ID" >&2
  exit 1
fi

"$PY" -u scripts/score_emulator_global_mean.py \
  --config "$CONFIG" --source-tag "$SOURCE" --candidate-tag "$CANDIDATE" --output "$OUTPUT"
