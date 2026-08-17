#!/bin/bash
#SBATCH --job-name=rblend_hon
#SBATCH --time=02:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --constraint=x86-64-v3
#SBATCH --partition=cluster
#SBATCH --output=/home/z/Zekang.Zhang/logs/rblend_hon_%j.out

# cont.89 (task #33): HONEST half-shear R_blend override, FIREWALL-CLEAN (constgold r_sim NEVER read).
# Recipe (production, NOT the rejected _hs flow retrain, NOT the circular build_scene_rblend):
#   [1] compute_deltaet_target.py on the blendemu BLEND response catalogue (UNSHEARED primary,
#       neighbour sheared g=0.2 -> clean neighbour-shear leakage) -> R(flux x size x blend) grid.
#   [2] harvest_grid_perobj.py --isolated-zero: look the grid up per-object on the constgold cert
#       catalogue (COORDS ONLY, r_sim never used) -> {case,input_index,value} R_blend override,
#       with isolated objects forced to 0 (fixes the emulator defect: emu R_blend ~+0.11-0.17 on
#       isolated objects that have NO neighbour -> no neighbour-shear leakage).
# Output override is a drop-in for eval_selection_robustness.py --rblend-override on the szfine dump.
set -e
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}:${BLENDEMU_ROOT:-/home/z/Zekang.Zhang/blendemu}:$PYTHONPATH"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
cd "${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}"

BLENDCAT=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876/response_catalogue_train_cases0_99.feather
CONSTCAT=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant/constant_response_catalogue_train.feather
GRID=results/response_target_blend_deltaet_c0-99.npz
OVR=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/rblend_honest_deltaet_c40-139.npz

echo "### RBLEND_HON job=$SLURM_JOB_ID node=$SLURMD_NODENAME ###"; date
echo "--- [1/2] build BLEND deltaet grid (g=0.2, constgold NEVER read) ---"
python -B -u scripts/compute_deltaet_target.py \
  --catalogue $BLENDCAT \
  --nominal-g 0.2 --deltaet-col delta_et1 \
  --n-flux 6 --n-size 3 --n-dist 3 --max-case 99 --min-count 500 \
  --output $GRID 2>&1 | grep -v "module command"

echo "--- [2/2] harvest per-object R_blend override (isolated -> 0; coords only) ---"
python -B -u scripts/harvest_grid_perobj.py \
  --grid $GRID --catalogue $CONSTCAT \
  --min-case 40 --isolated-zero \
  --out $OVR 2>&1 | grep -v "module command"

echo "RBLEND_HON_DONE grid=$GRID ovr=$OVR"; date
