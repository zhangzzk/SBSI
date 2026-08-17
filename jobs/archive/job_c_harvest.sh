#!/bin/bash
#SBATCH --job-name=c_harvest
#SBATCH --time=02:00:00
#SBATCH --mem=90G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=inter
#SBATCH --exclude=met-cl-vis01,th-cl-naples01,th-cl-naples02,th-cl-naples03,th-cl-naples04,th-cl-naples06,th-cl-naples07,th-cl-naples08,th-cl-naples09,th-cl-naples10,th-cl-naples11,cip-cl-compute9
#SBATCH --output=/home/z/Zekang.Zhang/logs/c_harvest_%j.out

# ADDITIVE-BIAS c harvest (WORKLOG cont.35). Runs measure_flow_c.py per model on the OOS c40-139 split
# with the conc (near/far/max) + meas-prim lookups so it works for the realistic szfl / szfl_noz flows.
# Reports sim_c1/c2, flow_c1/c2, and the calibration-relevant RESIDUAL sim_c - flow_c per model.
# Pass MODELS as a space-separated list of model paths via env. Small cards excluded (OOM-safe).
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}:${BLENDEMU_ROOT:-/home/z/Zekang.Zhang/blendemu}:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}"

MODELS=${MODELS:?set MODELS (space-separated model paths)}
CD=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant
CAT=${CAT:-$CD/constant_response_catalogue_c40-139.feather}
FL=results/crowd_flux_conc_c0-199.feather
MP=results/meas_prim_lookup_c0-139.feather
ROWS=${ROWS:-1500000}

echo "### C_HARVEST job=$SLURM_JOB_ID cat=$(basename $CAT) rows=$ROWS ###"; nvidia-smi -L; date
for M in $MODELS; do
  echo "===== C-MEASURE $M ====="; date
  if [ ! -f "$M" ]; then echo "MISSING MODEL $M"; continue; fi
  stdbuf -oL -eL python -u scripts/measure_flow_c.py \
    --measurement-model "$M" --catalogue "$CAT" \
    --crowd-flux-lookup $FL --meas-prim-lookup $MP \
    --model-max-rows $ROWS --n-samples 128 --batch-size 16384 2>&1 \
    | grep --line-buffered -vE "module command" \
    | stdbuf -oL sed "s|^|[$(basename $M .pt)] |"
  echo "C_DONE $M"; date
done
echo "C_HARVEST_ALL_DONE job=$SLURM_JOB_ID"; date
