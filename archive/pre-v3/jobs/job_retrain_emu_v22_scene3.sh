#!/bin/bash
#SBATCH --job-name=emu_v22_s3
#SBATCH --time=03:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=12
#SBATCH --partition=cluster
#SBATCH --output=/home/z/Zekang.Zhang/logs/emu_v22_s3_%j.out
set -euo pipefail
PY=/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python
export LD_LIBRARY_PATH="/project/ls-gruen/users/zekang.zhang/envs/sims1/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="/home/z/Zekang.Zhang/blendemu:/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:${PYTHONPATH:-}"
export XGB_DEVICE=cpu CONFIG_PATH=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot/configs/fs2_lsst_r_extnbr_v22_scene3.yaml HELDOUT_MIN_CASE=40
META=/home/z/Zekang.Zhang/blendemu/models/emulator_metadata_lsst_r_extnbr_v22_scene3.json
[ ! -e "$META" ] || { echo "REFUSING to overwrite $META"; exit 1; }
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
echo "### TRAIN V2.2 SCENE3 EMULATOR job=$SLURM_JOB_ID ###"; date
"$PY" -u scripts/retrain_emulator_v22_scene3.py
"$PY" -u - <<'PY'
import json
b='/home/z/Zekang.Zhang/blendemu/models/emulator_metadata_lsst_r_extnbr_v22.json'
p='/home/z/Zekang.Zhang/blendemu/models/emulator_metadata_lsst_r_extnbr_v22_scene3.json'
base=json.load(open(b)); meta=json.load(open(p))
for task in ('classification','self_response'): meta['tasks'][task]=base['tasks'][task]
json.dump(meta,open(p,'w'),indent=2,sort_keys=True,allow_nan=False)
print('tasks',sorted(meta['tasks'])); print('features',meta['tasks']['regression']['features'])
PY
echo EMU_V22_SCENE3_ALL_DONE; date
