#!/bin/bash
#SBATCH --job-name=emu_v22_phys2
#SBATCH --time=02:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cluster
#SBATCH --output=/home/z/Zekang.Zhang/logs/emu_v22_phys2_%j.out
set -euo pipefail

# Feature-only V2.2 ablation: exactly two dimensionless pair coordinates.
PY=/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python
export LD_LIBRARY_PATH="/project/ls-gruen/users/zekang.zhang/envs/sims1/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="/home/z/Zekang.Zhang/blendemu:/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:${PYTHONPATH:-}"
export XGB_DEVICE=cpu
export CONFIG_PATH=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot/configs/fs2_lsst_r_extnbr_v22_phys2.yaml
export HELDOUT_MIN_CASE=40
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot

MODEL=/home/z/Zekang.Zhang/blendemu/models/regression_model_lsst_r_extnbr_v22_phys2.json
META=/home/z/Zekang.Zhang/blendemu/models/emulator_metadata_lsst_r_extnbr_v22_phys2.json
CURVE=/home/z/Zekang.Zhang/blendemu/models/regression_train_curve_lsst_r_extnbr_v22_phys2.npz
for output in "$MODEL" "$META" "$CURVE"; do
    [ ! -e "$output" ] || { echo "REFUSING to overwrite $output"; exit 1; }
done

echo "### V2.2 PHYS2 EMULATOR job=$SLURM_JOB_ID ###"
date
"$PY" -u scripts/retrain_emulator_v22_phys2.py 2>&1 \
    | grep -v --line-buffered "module command"

# Keep the V2.2 classifier and self-response tasks unchanged so the new tag is
# directly loadable as a complete BlendEMU suite.
"$PY" -u -c "
import json
import os
baseline_path='/home/z/Zekang.Zhang/blendemu/models/emulator_metadata_lsst_r_extnbr_v22.json'
target_path='$META'
with open(baseline_path, encoding='utf-8') as handle:
    baseline=json.load(handle)
with open(target_path, encoding='utf-8') as handle:
    target=json.load(handle)
for task in ('classification', 'self_response'):
    target['tasks'][task]=baseline['tasks'][task]
tmp=target_path+'.tasks.tmp'
with open(tmp, 'w', encoding='utf-8') as handle:
    json.dump(target, handle, indent=2, sort_keys=True, allow_nan=False)
    handle.write('\\n')
os.replace(tmp, target_path)
print('tasks:', list(target['tasks']))
print('regression features:', target['tasks']['regression']['features'])
print('feature definitions:', target['tasks']['regression']['metrics']['feature_definitions'])
"
echo V22_PHYS2_EMU_ALL_DONE
date
