#!/bin/bash
#SBATCH --job-name=emu_v21
#SBATCH --time=04:00:00
#SBATCH --mem=24G          # measured run 15520060: MaxRSS 9.13G
#SBATCH --cpus-per-task=8  # measured run 15520060: 7.2 of 16 cores busy (xgboost hist)
#SBATCH --partition=cluster
#SBATCH --output=/home/z/Zekang.Zhang/logs/emu_v21_%j.out

# STEP 3 of V2.1: the blending-response emulator, retrained on the V2.1 domain.
#
# ONE LEVER vs the fiducial `lsst_r_extnbr_indom_tuned`: the training population. Same catalogue,
# features, preprocessing, split, hyperparameters (inherited from the tuned emulator's own metadata
# -- see the script docstring for why that, and not production's) and the same case firewall.
#
# The V2.1 domain is half box and half CURVE, so it cannot live entirely in the YAML cuts; the
# script wraps blendemu's row selection with sbs_shear.domain and ASSERTS that the YAML box is
# exactly the V2.1 bounding box. It also refuses to proceed if the curve removed no rows, which is
# what a silently-broken patch would look like.
#
# FIREWALL: HELDOUT_MIN_CASE=40 -> cases 40-199 only; gold cases 0-39 excluded. R_blend never sees
# constgold, and this emulator is NOT promoted on constgold m -- promotion is argued on the
# per-pair ruler (scripts/eval_rblend_gap.py).
set -o pipefail
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/blendemu:/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:$PYTHONPATH"
export XGB_DEVICE=cpu
export CONFIG_PATH=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot/configs/fs2_lsst_r_extnbr_v21.yaml
export HELDOUT_MIN_CASE=40
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot

echo "### V2.1 EMULATOR RETRAIN job=$SLURM_JOB_ID ###"; date
python -u scripts/retrain_emulator_v21.py 2>&1 | grep -v --line-buffered "module command" \
  || { echo V21_EMU_FAILED; exit 1; }

# The retrain writes only the REGRESSION task; copy the classification / self_response blocks from
# production so the metadata is loadable, exactly as job_retrain_indom.sh and job_tune_indom.sh do.
python -u -c "
import json
prod=json.load(open('/home/z/Zekang.Zhang/blendemu/models/emulator_metadata_lsst_r.json'))
p='/home/z/Zekang.Zhang/blendemu/models/emulator_metadata_lsst_r_extnbr_v21.json'
ext=json.load(open(p))
for t in ('classification','self_response'):
    task=dict(prod['tasks'][t])
    task['model_file']=task['model_file'].replace('_lsst_r.json','_lsst_r_extnbr_ho.json')
    ext['tasks'][t]=task
json.dump(ext, open(p,'w'), indent=2, sort_keys=True, allow_nan=False)
print('v21 tasks:', list(ext['tasks'].keys()))
print('regression cuts:', ext['tasks']['regression']['cuts'])
print('domain:', ext['tasks']['regression'].get('metrics',{}).get('sbsi_domain',{}).get('description'))
" 2>&1 | grep -v "module command"
echo V21_EMU_ALL_DONE; date
