#!/bin/bash
#SBATCH --job-name=tune_indom
#SBATCH --time=12:00:00
#SBATCH --mem=140G
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a40:1
#SBATCH --partition=inter
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/tune_indom_%j.out

# Optuna search for the IN-DOMAIN blend emulator, on ITS OWN training population.
#
# Owner asked whether `_indom` was ever hyperparameter-tuned. It was not, and neither was the
# certified `_ho`: production `lsst_r`, `_ho` and `_indom` all carry byte-identical XGBoost params
# from one search on the ORIGINAL full-population fit, and models/studies/ holds no study for any
# `_extnbr*` tag. The `n_trials` in the configs is inert on the retrain_extnbr code path.
#
# This matters because WORKLOG 28l called the -41.5% close-pair deficit a REPRESENTATIONAL limit --
# a claim about capacity -- on a population whose capacity knobs were never re-searched. 28l's
# WEIGHT_CLOSE sweep does partly cover it (min_child_weight and gamma both scale with sample weight,
# so K=20 relaxed 154 -> ~8 effective and bought nothing), but max_depth is weight-INVARIANT and has
# never been probed. See scripts/tune_emulator_indom.py for the full argument and the
# PRE-REGISTERED expectation (small global-R2 gain, little or no close-pair movement).
#
# GPU: a single CPU fit on the 62.6M-row training set takes ~8 min (indom_tr logs), so 100 trials is
# ~14 h on CPU. XGBoost `hist` on an a40 makes this tractable. XGBoost does not use the CUDA VMM
# APIs, so the `cip` vGPU caveat in CLAUDE.md (NO_EXPANDABLE_SEGMENTS) does not apply here.
#
# FIREWALL: HELDOUT_MIN_CASE=40 restricts training AND the Optuna validation split to cases 40-199.
# Constgold (0-39) is never read, so no hyperparameter is selected on constgold m. Writes a NEW tag;
# the certified `_ho` and the existing `_indom` are untouched. WEIGHT_CLOSE deliberately unset.
#
# RESUMABLE: the Optuna study is persisted to models/studies/regression_lsst_r_extnbr_indom_tuned.db
# and `_load_or_create_study` reloads it, so a timeout can simply be resubmitted to continue.
set -o pipefail
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/blendemu:/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:$PYTHONPATH"
export CONFIG_PATH=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot/configs/fs2_lsst_r_extnbr_indom_tuned.yaml
export HELDOUT_MIN_CASE=40
export XGB_DEVICE=${XGB_DEVICE:-cuda}
export N_TRIALS=${N_TRIALS:-100}
unset WEIGHT_CLOSE          # no close-pair reweighting: this run isolates hyperparameters alone
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot

echo "### TUNE INDOM job=$SLURM_JOB_ID  n_trials=$N_TRIALS  device=$XGB_DEVICE ###"; nvidia-smi -L; date
python -u scripts/tune_emulator_indom.py 2>&1 | grep -v "module command" \
  || { echo TUNE_INDOM_FAILED; exit 1; }

# Copy the classification / self_response task blocks from production so the metadata is loadable,
# exactly as job_retrain_indom.sh and job_retrain_ho.sh do.
python -u -c "
import json
prod=json.load(open('/home/z/Zekang.Zhang/blendemu/models/emulator_metadata_lsst_r.json'))
p='/home/z/Zekang.Zhang/blendemu/models/emulator_metadata_lsst_r_extnbr_indom_tuned.json'
ext=json.load(open(p))
for t in ('classification','self_response'):
    task=dict(prod['tasks'][t])
    task['model_file']=task['model_file'].replace('_lsst_r.json','_lsst_r_extnbr_ho.json')
    ext['tasks'][t]=task
json.dump(ext, open(p,'w'), indent=2, sort_keys=True, allow_nan=False)
print('tuned tasks:', list(ext['tasks'].keys()))
print('regression cuts:', ext['tasks']['regression']['cuts'])
" 2>&1 | grep -v "module command"
echo TUNE_INDOM_ALL_DONE; date
