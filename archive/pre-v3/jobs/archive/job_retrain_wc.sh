#!/bin/bash
#SBATCH --job-name=indom_tr
#SBATCH --time=04:00:00
#SBATCH --mem=110G
#SBATCH --cpus-per-task=16
#SBATCH --partition=cluster
#SBATCH --output=/home/z/Zekang.Zhang/logs/indom_tr_%j.out

# IN-DOMAIN emulator + CLOSE-PAIR LOSS WEIGHTING. The in-domain retrain alone did NOT fix the
# close-pair deficit (-40.3% vs -41.5% certified), refuting the population-mixing explanation. This
# adds WEIGHT_CLOSE (w = 1 + K*exp(-d/1")), the lever already present in retrain_extnbr.py, whose own
# comment records a known close-pair under-fit. Feature-based, so the conditional mean stays unbiased.
# certified `lsst_r_extnbr_ho`) except CONFIG_PATH, which differs from the certified config on exactly
# two lines: model_tag, and regression_cuts narrowing the PRIMARY to the deliverable domain
# (mag 28->26, Re 0.1->0.3). SECONDARY cuts are untouched, so neighbours stay full-population per
# GOALS.md. Same hyperparameters, same preprocessing, same split -> any difference is the domain cut.
#
# WHY (WORKLOG 2026-07-28j): the certified emulator is essentially UNBIASED outside our domain
# (+0.3%, 0.1 sigma) and badly biased INSIDE it (-11.9%; -41.5% for pairs closer than 1"). At close
# separations it predicts ~0.030 for both populations while truth is 0.0518 in-domain vs 0.0277 out --
# it is fitting one average across two populations that behave oppositely, and the out-of-domain half
# pulls the fit away from ours. Restricting the training population removes that pull.
#
# FIREWALL: HELDOUT_MIN_CASE=40 as in the certified run -> trains on cases 40-199, gold cases 0-39
# excluded. R_blend never sees constgold.
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/blendemu:/home/z/Zekang.Zhang/SBSI:$PYTHONPATH"
export XGB_DEVICE=cpu
export CONFIG_PATH=${CONFIG_PATH:?set CONFIG_PATH}
export HELDOUT_MIN_CASE=40
cd /home/z/Zekang.Zhang/blendemu
echo "### IN-DOMAIN retrain: primary mag<26, Re>0.3, cases 40-199 (gold 0-39 excluded) ###"; date
python -u scripts/retrain_extnbr.py 2>&1 | grep -v "module command" || { echo INDOM_TRAIN_FAILED; exit 1; }

# The retrain writes only the REGRESSION task; copy the classification/self_response task blocks from
# production so the metadata is loadable, exactly as the certified job does.
python -u -c "
import json
prod=json.load(open('models/emulator_metadata_lsst_r.json'))
ext=json.load(open('models/emulator_metadata_${MODELTAG}.json'))
for t in ('classification','self_response'):
    task=dict(prod['tasks'][t])
    task['model_file']=task['model_file'].replace('_lsst_r.json','_lsst_r_extnbr_ho.json')
    ext['tasks'][t]=task
json.dump(ext, open('models/emulator_metadata_${MODELTAG}.json','w'), indent=2, sort_keys=True, allow_nan=False)
print('indom tasks:', list(ext['tasks'].keys()))
print('regression cuts:', ext['tasks']['regression']['cuts'])
" 2>&1 | grep -v "module command"
echo INDOM_TRAIN_DONE; date
