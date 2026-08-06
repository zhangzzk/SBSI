#!/bin/bash
#SBATCH --job-name=emu_v22
#SBATCH --time=02:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cluster
#SBATCH --output=/home/z/Zekang.Zhang/logs/emu_v22_%j.out
set -euo pipefail

# V2.2 BlendEMU: same V2 inputs/features/split/secondary support and inherited tuned-V2
# hyperparameters; only the primary box changes to true r<25.8 and Re>0.5 arcsec.
PY=/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python
export LD_LIBRARY_PATH="/project/ls-gruen/users/zekang.zhang/envs/sims1/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="/home/z/Zekang.Zhang/blendemu:/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:${PYTHONPATH:-}"
export XGB_DEVICE=cpu
export CONFIG_PATH=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot/configs/fs2_lsst_r_extnbr_v22.yaml
export HELDOUT_MIN_CASE=40
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot

META=/home/z/Zekang.Zhang/blendemu/models/emulator_metadata_lsst_r_extnbr_v22.json
[ ! -e "$META" ] || { echo "REFUSING to overwrite $META"; exit 1; }
echo "### V2.2 EMULATOR job=$SLURM_JOB_ID ###"; date
"$PY" -u scripts/retrain_emulator_v22.py 2>&1 | grep -v --line-buffered "module command"

"$PY" -u -c "
import json
prod=json.load(open('/home/z/Zekang.Zhang/blendemu/models/emulator_metadata_lsst_r.json'))
p='$META'
ext=json.load(open(p))
for t in ('classification','self_response'):
    task=dict(prod['tasks'][t])
    task['model_file']=task['model_file'].replace('_lsst_r.json','_lsst_r_extnbr_ho.json')
    ext['tasks'][t]=task
json.dump(ext, open(p,'w'), indent=2, sort_keys=True, allow_nan=False)
print('tasks:', list(ext['tasks']))
print('regression cuts:', ext['tasks']['regression']['cuts'])
print('domain:', ext['tasks']['regression']['metrics']['sbsi_domain'])
"
echo V22_EMU_ALL_DONE; date
