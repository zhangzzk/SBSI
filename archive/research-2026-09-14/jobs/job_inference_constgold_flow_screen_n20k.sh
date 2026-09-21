#!/usr/bin/env bash
# Matched 20k complete-likelihood gate: original versus selected Flow-E, same detector.

#SBATCH --job-name=sbsi_Etheta_I20k
#SBATCH --partition=inter
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=180G
#SBATCH --time=04:00:00
#SBATCH --array=0-1%2
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/mixed_shear_e_coupling_ladder_v1/inference_n20k/logs/%x_%A_%a.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi_caches/mixed_shear_e_coupling_ladder_v1/inference_n20k/logs/%x_%A_%a.err

set -euo pipefail

repo=/home/z/Zekang.Zhang/SBSI
python=/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python
cache=/project/ls-gruen/users/zekang.zhang/sbsi_caches
screen=$cache/mixed_shear_e_coupling_ladder_v1
root=$screen/inference_n20k
source_sample=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/inference_proxy_topk_size075_n100k_v1/mock_constgold_comparison_100k_v3/constgold_plus_selected_sample_100k.parquet
prior=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/v31_fs2_default_qmc128_n100k_v1/compact_global
blend=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/blend_response_inference_scene_v1/blend_response_v1
threshold=1.3217558399823195

mkdir -p "$root/logs"
[[ -s "$screen/summary.json" && -s "$screen/selected_likelihood.json" ]] \
  || { echo "missing completed ladder selection" >&2; exit 2; }

case "${SLURM_ARRAY_TASK_ID:?submit as array 0-1}" in
  0)
    arm=original_transition
    flow=$cache/mixed_shear_cde/measurement_flow_mixed_g0_g005_E_s501_swaavg.pt
    likelihood=$root/original_transition_likelihood.json
    if [[ ! -e "$likelihood" ]]; then
      "$python" - "$screen/selected_likelihood.json" "$flow" "$likelihood" <<'PY'
import hashlib, json, sys
source, model, output = sys.argv[1:]
payload = json.load(open(source))
h = hashlib.sha256(open(model, "rb").read()).hexdigest()
payload["release"] = "v3.3-original-flow-transition-screen"
payload["description"] = "Original Flow-E with the transition detector; matched screen control."
payload["measurement_model"]["path"] = model
payload["measurement_model"]["sha256"] = h
open(output, "w").write(json.dumps(payload, indent=2) + "\n")
PY
    fi
    ;;
  1)
    arm=selected_retrain_transition
    likelihood=$screen/selected_likelihood.json
    flow=$("$python" -c "import json; print(json.load(open('$screen/summary.json'))['selected']['model'])")
    ;;
  *) echo "unknown array task $SLURM_ARRAY_TASK_ID" >&2; exit 2 ;;
esac

arm_root=$root/$arm
input=$arm_root/input
model_cache=$arm_root/cache/model_cache
proposal_cache=$arm_root/cache/proposal_cache
run_root=$arm_root/hybrid
mkdir -p "$arm_root/cache"

expected_sha=$("$python" -c "import json; print(json.load(open('$likelihood'))['measurement_model']['sha256'])")
echo "$expected_sha  $flow" | sha256sum --check --status

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-12}
export PYTHONPATH="$repo:${BLENDEMU_ROOT:-/home/z/Zekang.Zhang/blendemu}${PYTHONPATH:+:$PYTHONPATH}"

if [[ ! -f "$input/image_mock_manifest.json" ]]; then
  "$python" -u "$repo/scripts/prepare_constgold_inference_input.py" \
    --source-sample "$source_sample" --output "$input" \
    --measurement-model "$flow" --likelihood-config "$likelihood" \
    --injected-g1 0.02 --injected-g2 0.0 --pixel-size 0.2
fi

if [[ ! -f "$model_cache/manifest.json" || ! -f "$proposal_cache/manifest.json" ]]; then
  "$python" -u "$repo/scripts/run_inference.py" \
    --inference-config "$repo/configs/inference_v1_2.json" \
    --likelihood-config "$likelihood" \
    --scene-store "$prior/scene_store" --measurement-model "$flow" \
    --model-cache "$model_cache" --proposal-cache "$proposal_cache" \
    --blend-response-cache "$blend" --prepare-only --n-detected 1 \
    --device cuda --output "$arm_root/cache/prepare"
fi

export SBSI_PYTHON="$python"
export SOURCE_MOCK="$input"
export CHAIN_ROOT="$run_root"
export N_OBJECTS=20000
export N_GPUS=1
export PASSES=0
export MAG_CUT=22.0
export TAG=constgold_plus20k_k1024_m8192_exactnorm_pass0
export INFERENCE_CONFIG="$repo/configs/inference_v1_2.json"
export LIKELIHOOD_CONFIG="$likelihood"
export MEASUREMENT_MODEL="$flow"
export MODEL_CACHE="$model_cache"
export PROPOSAL_CACHE="$proposal_cache"
export ADAPTIVE_DRAW_LADDER="512 1024 2048 4096 8192"
export RESUME=1
unset SELECTION_NORMALIZATION_CACHE
export BUILD_SELECTION_NORMALIZATION_QUADRATIC=0
export BLEND_CACHE="$blend"
export CUT_ABS_EHAT=0.6
export CUT_BOUND="measured_mag_auto::25.8 measured_log_flux_radius:${threshold}:"
export PROPOSAL_CANDIDATE_SOURCE=whole_catalogue_gaussian_proxy

bash "$repo/jobs/job_inference_hybrid_chain.sh"
echo "CONSTGOLD_FLOW_SCREEN_20K_DONE arm=$arm"
