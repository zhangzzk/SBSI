#!/bin/bash
#SBATCH --job-name=tune_fin
#SBATCH --time=01:30:00
#SBATCH --mem=140G
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a40:1
#SBATCH --partition=inter
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/tune_fin_%j.out

# FINALIZER for the parallel in-domain search: refit at the study's best params and SAVE. The search
# workers deliberately save nothing, so this is the single writer of
# models/regression_model_lsst_r_extnbr_indom_tuned.json.
#
# FINALIZE_ONLY=1 makes tune_emulator_indom.py skip the search and read the existing study. It still
# does the full baseline fit with the inherited production params on the SAME split, so the reported
# delta is a like-for-like comparison and not a number lifted from another run. It refuses to run on
# a study with fewer than 10 completed trials.
#
# It also records the REAL completed-trial count into the metadata, which is what
# scripts/check_emulator_provenance.py reads before any promotion -- so the 2-trial smoke-test
# artifact currently sitting at this path cannot be mistaken for the finished search.
set -o pipefail
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/blendemu:/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:$PYTHONPATH"
export CONFIG_PATH=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot/configs/fs2_lsst_r_extnbr_indom_tuned.yaml
export HELDOUT_MIN_CASE=40
export XGB_DEVICE=${XGB_DEVICE:-cuda}
export FINALIZE_ONLY=1
unset WEIGHT_CLOSE
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot

echo "### TUNE FINALIZE job=$SLURM_JOB_ID ###"; nvidia-smi -L; date
python -u scripts/tune_emulator_indom.py 2>&1 | grep -v "module command" \
  || { echo TUNE_FIN_FAILED; exit 1; }

# Copy the classification / self_response task blocks from production so the metadata is loadable,
# exactly as job_tune_indom.sh / job_retrain_indom.sh do.
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
" 2>&1 | grep -v "module command"

echo; echo "----- provenance (must now show the real trial count, not 2) -----"
python -u scripts/check_emulator_provenance.py --tag lsst_r_extnbr_indom_tuned || true
echo TUNE_FIN_ALL_DONE; date
