#!/bin/bash
#SBATCH --job-name=indist_tr
#SBATCH --time=04:00:00
#SBATCH --mem=110G
#SBATCH --cpus-per-task=16
#SBATCH --partition=cluster
#SBATCH --output=/home/z/Zekang.Zhang/logs/indist_tr_%j.out

# Retrain the R_blend emulator with an INPUT-FRAME `distance`, so the feature means the same thing
# at training time as it does when the m pipeline queries it.
#
# WHY (WORKLOG 2026-07-28n): `retrieve_response` measures the pair separation from the primary's
# DETECTED centroid, while `_nearest_neighbor_features` -- which builds both the half-shear ruler and
# the constgold catalogue the m pipeline uses -- measures it input-to-input. Measured over the whole
# training catalogue (job 15328617), the training distance runs SHORT by 0.110" below 0.5" and 0.119"
# at 0.5-1" (25% and 14% of the separation, 86% of pairs biased the same way), decaying to 0.001" by
# 5". That is the same separation profile as the close-pair deficit (-41.5% / -5.4% / -1.4%), and the
# sign is right: querying at the true separation returns what the model learned for wider pairs.
#
# Everything else is held fixed. The config differs from the certified fs2_lsst_r_extnbr_ho.yaml on
# exactly two lines (output_path, model_tag); same features, same regression_cuts, same
# hyperparameters, same test_size/random_state. The LABELS are untouched -- only the feature column
# was recomputed by scripts/fix_response_distance.py. So any difference is the distance definition.
#
# FIREWALL: HELDOUT_MIN_CASE=40 as in the certified run -> trains on cases 40-199, gold cases 0-39
# excluded. R_blend never sees constgold.
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/blendemu:/home/z/Zekang.Zhang/SBSI:$PYTHONPATH"
export XGB_DEVICE=cpu
export HELDOUT_MIN_CASE=40
export CONFIG_PATH=${CONFIG_PATH:-/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot/configs/fs2_lsst_r_extnbr_indist.yaml}
export MODELTAG=${MODELTAG:-lsst_r_extnbr_indist}
cd /home/z/Zekang.Zhang/blendemu
echo "### INPUT-FRAME DISTANCE retrain: tag=$MODELTAG  cases 40-199 ###"; date
echo "config: $CONFIG_PATH"
python -u scripts/retrain_extnbr.py 2>&1 | grep -v "module command" \
    || { echo INDIST_TRAIN_FAILED; exit 1; }

# The retrain writes only the REGRESSION task; copy the classification/self_response blocks from
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
print('tasks:', list(ext['tasks'].keys()))
print('regression cuts:', ext['tasks']['regression']['cuts'])
print('regression features:', ext['tasks']['regression']['features'])
" 2>&1 | grep -v "module command"
echo INDIST_TRAIN_DONE; date
