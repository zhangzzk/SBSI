#!/bin/bash
#SBATCH --job-name=extho_tr
#SBATCH --time=04:00:00
#SBATCH --mem=110G
#SBATCH --cpus-per-task=16
#SBATCH --partition=cluster
#SBATCH --output=/home/z/Zekang.Zhang/logs/extho_tr_%j.out
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="${BLENDEMU_ROOT:-/home/z/Zekang.Zhang/blendemu}:${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}:$PYTHONPATH"
export XGB_DEVICE=cpu
export CONFIG_PATH=/home/z/Zekang.Zhang/blendemu/configs/fs2_lsst_r_extnbr_ho.yaml
export HELDOUT_MIN_CASE=40
cd /home/z/Zekang.Zhang/blendemu
echo "### HELD-OUT retrain: extended emulator on cases 40-199 ONLY (gold 0-39 excluded) ###"
python -u scripts/retrain_extnbr.py 2>&1 | grep -v "module command" || { echo EXTHO_TRAIN_FAILED; exit 1; }
python -u -c "
import json
prod=json.load(open('models/emulator_metadata_lsst_r.json')); ext=json.load(open('models/emulator_metadata_lsst_r_extnbr_ho.json'))
for t in ('classification','self_response'):
    task=dict(prod['tasks'][t]); task['model_file']=task['model_file'].replace('_lsst_r.json','_lsst_r_extnbr_ho.json'); ext['tasks'][t]=task
json.dump(ext, open('models/emulator_metadata_lsst_r_extnbr_ho.json','w'), indent=2, sort_keys=True, allow_nan=False)
print('ho tasks:', list(ext['tasks'].keys()))
" 2>&1 | grep -v "module command"
echo EXTHO_TRAIN_DONE
