#!/usr/bin/env bash
# Paired five-block catalogue-profile calibration for one shear component.

#SBATCH --job-name=sbsi_cp_bias
#SBATCH --partition=inter,cip
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=52G
#SBATCH --time=02:00:00
#SBATCH --array=0-14%4
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/realflow_cases0_9_v1/logs/bias_profile_%x_%A_%a.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/realflow_cases0_9_v1/logs/bias_profile_%x_%A_%a.err

set -euo pipefail

repo=/home/z/Zekang.Zhang/SBSI
root=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/realflow_cases0_9_v1

: "${COMPONENT:=g1}"
: "${CALIBRATION_NAME:=bias_sizeonly_v1}"
: "${N_BLOCKS:=5}"

if [[ "$N_BLOCKS" -ne 5 ]]; then
  echo "this frozen 0-14 array requires N_BLOCKS=5" >&2
  exit 2
fi
if [[ "$COMPONENT" != g1 && "$COMPONENT" != g2 ]]; then
  echo "COMPONENT must be g1 or g2" >&2
  exit 2
fi

task=${SLURM_ARRAY_TASK_ID:?}
shear_slot=$((task / N_BLOCKS))
block=$((task % N_BLOCKS))
if ((shear_slot < 0 || shear_slot > 2)); then
  echo "array index $task is outside the three-shear design" >&2
  exit 2
fi

injections=(-0.02 0 0.02)
arms=(minus g0 plus)
profiles=(
  "-0.03,-0.025,-0.02,-0.015,-0.01"
  "-0.01,-0.005,0,0.005,0.01"
  "0.01,0.015,0.02,0.025,0.03"
)
injected=${injections[$shear_slot]}
arm=${arms[$shear_slot]}
profile=${profiles[$shear_slot]}

if [[ "$COMPONENT" == g1 ]]; then
  injected_g1=$injected
  injected_g2=0
  direction_g1=1
  direction_g2=0
  selection_cache="$root/selection_logr145_only_profile_coarse_${arm}_selection_cache"
else
  injected_g1=0
  injected_g2=$injected
  direction_g1=0
  direction_g2=1
  selection_cache="$root/selection_logr145_g2_profile_coarse_${arm}_selection_cache"
fi

scene_seed=$((1601 + block))
detection_seed=$((2601 + block))
flow_seed=$((3601 + block))
proposal_seed=$((4601 + block))
block_name=$(printf 'b%02d' "$block")

env \
  RUN_NAME="$CALIBRATION_NAME/$COMPONENT/$block_name" \
  ARM="$arm" \
  INJECTED_G1="$injected_g1" INJECTED_G2="$injected_g2" \
  DIRECTION_G1="$direction_g1" DIRECTION_G2="$direction_g2" \
  PROFILE="$profile" LADDER=32768 \
  CUT_ABS=none CUT_BOUND=measured_log_flux_radius:1.45: \
  SELECTION_CACHE="$selection_cache" \
  SCENE_SEED="$scene_seed" DETECTION_SEED="$detection_seed" FLOW_SEED="$flow_seed" \
  PROPOSAL_SEEDS="$proposal_seed" \
  bash "$repo/jobs/job_catalogue_selection_profile.sh"
