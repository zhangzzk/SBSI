#!/bin/bash
#SBATCH --job-name=emu_v22_o3
#SBATCH --time=03:00:00
#SBATCH --mem=80G
#SBATCH --cpus-per-task=16
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/emu_v22_o3_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/emu_v22_o3_%j.err
set -euo pipefail

ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
PY=/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python
export LD_LIBRARY_PATH="/project/ls-gruen/users/zekang.zhang/envs/sims1/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="/home/z/Zekang.Zhang/blendemu:$ROOT:${PYTHONPATH:-}"
export XGB_DEVICE=cpu
export CONFIG_PATH=$ROOT/configs/fs2_lsst_r_extnbr_v22_other3abs.yaml
export HELDOUT_MIN_CASE=40
META=/home/z/Zekang.Zhang/blendemu/models/emulator_metadata_lsst_r_extnbr_v22_other3abs.json

[ ! -e "$META" ] || { echo "REFUSING to overwrite $META"; exit 1; }
cd "$ROOT"
echo "### TRAIN V2.2 PAIR-OTHER-ABS EMULATOR job=$SLURM_JOB_ID ###"
date
"$PY" -u scripts/retrain_emulator_v22_other3abs.py
"$PY" -u - <<'PY'
import json
base_path = '/home/z/Zekang.Zhang/blendemu/models/emulator_metadata_lsst_r_extnbr_v22.json'
new_path = '/home/z/Zekang.Zhang/blendemu/models/emulator_metadata_lsst_r_extnbr_v22_other3abs.json'
with open(base_path, encoding='utf-8') as handle:
    base = json.load(handle)
with open(new_path, encoding='utf-8') as handle:
    metadata = json.load(handle)
for task in ('classification', 'self_response'):
    metadata['tasks'][task] = base['tasks'][task]
with open(new_path, 'w', encoding='utf-8') as handle:
    json.dump(metadata, handle, indent=2, sort_keys=True, allow_nan=False)
    handle.write('\n')
print('tasks', sorted(metadata['tasks']))
print('features', metadata['tasks']['regression']['features'])
PY
echo EMU_V22_OTHER3ABS_ALL_DONE
date
