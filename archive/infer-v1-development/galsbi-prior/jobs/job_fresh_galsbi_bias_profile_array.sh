#!/usr/bin/env bash
# Archived GalSBI-prior job.
# Five paired zero/sign blocks for one shear component on the fresh prior.

#SBATCH --job-name=sbsi_fresh_bias
#SBATCH --partition=inter,cip
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --array=0-14%4
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/fresh_galsbi_v1/logs/bias_%x_%A_%a.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/fresh_galsbi_v1/logs/bias_%x_%A_%a.err

set -euo pipefail

repo=/home/z/Zekang.Zhang/SBSI
: "${PRIOR_ROOT:=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/fresh_galsbi_v1}"
: "${COMPONENT:=g1}"
: "${CALIBRATION_NAME:=bias_rblend_v1}"
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
arms=(minus zero plus)
profiles=(
  "-0.035,-0.03,-0.025,-0.02,-0.015,-0.01,-0.005"
  "-0.015,-0.01,-0.005,0,0.005,0.01,0.015"
  "0.005,0.01,0.015,0.02,0.025,0.03,0.035"
)
injected=${injections[$shear_slot]}
arm=${arms[$shear_slot]}
profile=${profiles[$shear_slot]}

if [[ "$COMPONENT" == g1 ]]; then
  injected_g1=$injected
  injected_g2=0
  direction_g1=1
  direction_g2=0
else
  injected_g1=0
  injected_g2=$injected
  direction_g1=0
  direction_g2=1
fi

scene_seed=$((1901 + block))
detection_seed=$((2901 + block))
flow_seed=$((3901 + block))
proposal_seed=$((5901 + block))
block_name=$(printf 'b%02d' "$block")

env \
  PRIOR_ROOT="$PRIOR_ROOT" \
  RUN_NAME="$CALIBRATION_NAME/$COMPONENT/$block_name/$arm" \
  INJECTED_G1="$injected_g1" INJECTED_G2="$injected_g2" \
  DIRECTION_G1="$direction_g1" DIRECTION_G2="$direction_g2" \
  PROFILE="$profile" LADDER=32768 PROPOSAL_SEEDS="$proposal_seed" \
  N_DETECTED=2048 \
  SCENE_SEED="$scene_seed" DETECTION_SEED="$detection_seed" FLOW_SEED="$flow_seed" \
  bash "$repo/jobs/job_fresh_galsbi_closure_profile.sh"
