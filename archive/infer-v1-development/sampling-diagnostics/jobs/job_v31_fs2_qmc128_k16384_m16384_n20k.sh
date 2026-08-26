#!/usr/bin/env bash
# Archived Infer V1 sampling diagnostic job.
# Run one K=16,384, M<=16,384 arm on the frozen FS2/QMC-128 20k diagnostic rows.

#SBATCH --job-name=sbsi_k16384_m16384
#SBATCH --partition=inter
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=01:00:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/v31_fs2_default_qmc128_km_n20k_v1/logs/%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/v31_fs2_default_qmc128_km_n20k_v1/logs/%x_%j.err

set -euo pipefail

repo=/home/z/Zekang.Zhang/SBSI
source_root=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/v31_fs2_default_qmc128_n100k_v1
root=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/v31_fs2_default_qmc128_km_n20k_v1
proposal_seed=${PROPOSAL_SEED:?set PROPOSAL_SEED to 8701 or 8702}
output="$root/results/k16384_e0p1_p${proposal_seed}_m512_16384"

case "$proposal_seed" in
  8701|8702) ;;
  *)
    echo "unsupported PROPOSAL_SEED=$proposal_seed" >&2
    exit 2
    ;;
esac

mkdir -p "$root/results"
CUDA_VISIBLE_DEVICES=0 env \
  OUTPUT="$output" \
  MOCK_INPUT="$source_root/prepare/mock" \
  SCENE_STORE="$source_root/compact_global/scene_store" \
  MODEL_CACHE="$source_root/compact_global/model_cache" \
  PROPOSAL_CACHE="$source_root/compact_global/proposal_cache" \
  FLOW=/project/ls-gruen/users/zekang.zhang/sbsi_caches/mixed_shear_cde/measurement_flow_mixed_g0_g005_E_s501_swaavg.pt \
  BLEND_RESPONSE_CACHE= \
  PROPOSAL_FLOW_SAMPLES=128 \
  PROPOSAL_STATISTIC=mean PROPOSAL_DISPERSION_STATISTIC=std \
  PROPOSAL_COORDINATE_SEED=8201 \
  PROPOSAL_CANDIDATES=16384 PROPOSAL_PREFILTER_CANDIDATES=131072 \
  PROPOSAL_EPSILON=0.1 PROPOSAL_SEED="$proposal_seed" \
  PROPOSAL_METHOD=initial_center_posterior_adapted \
  INITIAL_STRATEGY=mean_observed_shape ADAPTIVE_ONE_STEP=1 \
  ADAPTIVE_DRAW_LADDER="512 1024 2048 4096 8192 16384" \
  RETAIN_FULL_LADDER=1 COMPILE_FLOW=1 OBJECT_CHUNK=128 ATOM_CHUNK=4096 \
  OBSERVATION_START=0 OBSERVATION_STOP=20000 \
  OMP_NUM_THREADS=8 \
  bash "$repo/jobs/job_infer_v1.sh" \
  >"$root/logs/k16384_p${proposal_seed}.out" \
  2>"$root/logs/k16384_p${proposal_seed}.err"

echo "K=16384 M<=16384 seed=$proposal_seed complete"
