#!/usr/bin/env bash
# Recenter the two existing ConstGold cases-40--44 image arms independently.

#SBATCH --job-name=sbsi_fs2_imgrec
#SBATCH --partition=inter,cip
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-1
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/fs2_full_cases0_9_v1/logs/%x_%A_%a.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/fs2_full_cases0_9_v1/logs/%x_%A_%a.err

set -euo pipefail

repo=/home/z/Zekang.Zhang/SBSI
prior_root=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/fs2_full_cases0_9_v1
image_root=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/image_closure_fs2_constgold_v1
: "${SELECTION_MODE:=none}"
: "${SELECTION_SAMPLES:=64}"
: "${SELECTION_SEED:=8101}"
: "${DRAWS:=8192}"
: "${PROPOSAL_CANDIDATES:=16384}"
: "${PROPOSAL_SEED:=8701}"
: "${PROPOSAL_EPSILON:=0.1}"
: "${PROPOSAL_METHOD:=initial_center_posterior_adapted}"
: "${H:=0.001}"
: "${INITIAL_G1:=0.0}"
: "${INITIAL_G2:=0.0}"
: "${INITIAL:=${INITIAL_G1},${INITIAL_G2}}"
: "${TOLERANCE:=0.0001}"
: "${MAX_BACKTRACKS:=8}"
: "${BLOCK_TAG:=c40_44_n2048}"
: "${POSITIVE_MOCK:=$image_root/mock_g1p002_c40_44_n2048_v2}"
: "${NEGATIVE_MOCK:=$image_root/mock_g1m002_c40_44_n2048}"

case "$SLURM_ARRAY_TASK_ID" in
  0)
    arm=g1p002
    mock="$POSITIVE_MOCK"
    ;;
  1)
    arm=g1m002
    mock="$NEGATIVE_MOCK"
    ;;
  *)
    echo "unsupported array index $SLURM_ARRAY_TASK_ID" >&2
    exit 2
    ;;
esac

cut_args=()
selection_cache=
h_tag=${H//./p}
epsilon_tag=${PROPOSAL_EPSILON//./p}
case "$PROPOSAL_METHOD" in
  initial_center_posterior_adapted)
    proposal_tag=adapted
    ;;
  distance_kernel)
    proposal_tag=distance
    ;;
  *)
    echo "unsupported proposal method $PROPOSAL_METHOD" >&2
    exit 2
    ;;
esac
case "$SELECTION_MODE" in
  none)
    ;;
  realistic)
    cut_args+=(CUT_MAG_HI=25.8 CUT_LOG_RADIUS_LO=0.9162907318741551)
    selection_cache="$prior_root/selection_cache_realistic_${arm}_${BLOCK_TAG}_${proposal_tag}_e${epsilon_tag}_h${h_tag}_m${DRAWS}_k${PROPOSAL_CANDIDATES}_p${PROPOSAL_SEED}_qmc${SELECTION_SAMPLES}_q${SELECTION_SEED}"
    ;;
  stress)
    cut_args+=(CUT_LOG_RADIUS_LO=1.45)
    selection_cache="$prior_root/selection_cache_stress_${arm}_${BLOCK_TAG}_${proposal_tag}_e${epsilon_tag}_h${h_tag}_m${DRAWS}_k${PROPOSAL_CANDIDATES}_p${PROPOSAL_SEED}_qmc${SELECTION_SAMPLES}_q${SELECTION_SEED}"
    ;;
  *)
    echo "SELECTION_MODE must be none, realistic, or stress" >&2
    exit 2
    ;;
esac

output="$prior_root/constgold_${arm}_${BLOCK_TAG}_${SELECTION_MODE}_${proposal_tag}_h${h_tag}_m${DRAWS}_k${PROPOSAL_CANDIDATES}_e${epsilon_tag}_p${PROPOSAL_SEED}_qmc${SELECTION_SAMPLES}_q${SELECTION_SEED}_v1"

exec env \
  SCENE_STORE="$prior_root/scene_store" \
  MODEL_CACHE="$prior_root/model_cache_numerical_v1" \
  PROPOSAL_CACHE="$prior_root/proposal_cache_alltargets_v1" \
  BLEND_RESPONSE_CACHE="$prior_root/blend_response_v1" \
  MOCK_INPUT="$mock" OUTPUT="$output" \
  SELECTION_CACHE="$selection_cache" \
  SELECTION_SAMPLES="$SELECTION_SAMPLES" SELECTION_SEED="$SELECTION_SEED" \
  DRAWS="$DRAWS" PROPOSAL_CANDIDATES="$PROPOSAL_CANDIDATES" \
  PROPOSAL_SEED="$PROPOSAL_SEED" PROPOSAL_EPSILON="$PROPOSAL_EPSILON" \
  PROPOSAL_METHOD="$PROPOSAL_METHOD" \
  H="$H" INITIAL="$INITIAL" \
  TOLERANCE="$TOLERANCE" MAX_BACKTRACKS="$MAX_BACKTRACKS" \
  "${cut_args[@]}" \
  bash "$repo/jobs/job_infer_v1.sh"
