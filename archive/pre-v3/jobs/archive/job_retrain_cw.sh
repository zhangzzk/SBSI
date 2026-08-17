#!/bin/bash
#SBATCH --job-name=extcw_tr
#SBATCH --time=04:00:00
#SBATCH --mem=110G
#SBATCH --cpus-per-task=16
#SBATCH --partition=cluster
#SBATCH --output=/home/z/Zekang.Zhang/logs/extcw_tr_%j.out
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/blendemu:/home/z/Zekang.Zhang/SBSI:$PYTHONPATH"
export XGB_DEVICE=cpu
export CONFIG_PATH=/home/z/Zekang.Zhang/blendemu/configs/fs2_lsst_r_extnbr_cw.yaml
export WEIGHT_CLOSE=3
cd /home/z/Zekang.Zhang/blendemu
echo "### RETRAIN extended-domain emulator with CLOSE-PAIR upweighting (w=1+3*exp(-d)) to lift <1\" fit ###"
python -u scripts/retrain_extnbr.py 2>&1 | grep -v "module command" || { echo EXTCW_TRAIN_FAILED; exit 1; }
echo "### merge aux (classification/self_response) metadata into cw tag ###"
python -u -c "
import json
prod=json.load(open('models/emulator_metadata_lsst_r.json'))
ext=json.load(open('models/emulator_metadata_lsst_r_extnbr_cw.json'))
for t in ('classification','self_response'):
    task=dict(prod['tasks'][t]); task['model_file']=task['model_file'].replace('_lsst_r.json','_lsst_r_extnbr_cw.json')
    ext['tasks'][t]=task
json.dump(ext, open('models/emulator_metadata_lsst_r_extnbr_cw.json','w'), indent=2, sort_keys=True, allow_nan=False)
print('cw tasks:', list(ext['tasks'].keys()))
" 2>&1 | grep -v "module command"
echo EXTCW_TRAIN_DONE
