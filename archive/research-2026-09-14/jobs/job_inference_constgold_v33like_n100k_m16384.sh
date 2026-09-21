#!/usr/bin/env bash
# V3.3-like repeat of the five-state 100k ConstGold complete-likelihood run.

#SBATCH --job-name=sbsi_cg100k_v33
#SBATCH --partition=inter
#SBATCH --gres=gpu:v100:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=8:00:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/inference_constgold_v33like_size075_n100k_m16384_v1/logs/%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/inference_constgold_v33like_size075_n100k_m16384_v1/logs/%x_%j.err

set -euo pipefail

repo=${SBSI_REPOSITORY:-/home/z/Zekang.Zhang/SBSI}
python=${SBSI_PYTHON:-/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python}
prior_root=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/v31_fs2_default_qmc128_n100k_v1/compact_global
old_root=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/inference_constgold_plus_size075_n100k_m16384_v1
root=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/inference_constgold_v33like_size075_n100k_m16384_v1
input=$root/input
cache=$root/cache
likelihood=$repo/configs/likelihood_v3_3_like.json
flow=/project/ls-gruen/users/zekang.zhang/sbsi_caches/mixed_shear_e_photometric/measurement_flow_mixed_g0_g005_E_r500_t500_s501_swaavg.pt
blend=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/blend_response_inference_scene_v1/blend_response_v1
threshold=1.3217558399823195

mkdir -p "$root/logs" "$cache"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export PYTHONPATH="$repo:${BLENDEMU_ROOT:-/home/z/Zekang.Zhang/blendemu}${PYTHONPATH:+:$PYTHONPATH}"

# Freeze the identical measured rows under the new likelihood identity.
if [[ ! -f "$input/image_mock_manifest.json" ]]; then
  "$python" -u "$repo/scripts/prepare_constgold_inference_input.py" \
    --source-sample /project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/inference_proxy_topk_size075_n100k_v1/mock_constgold_comparison_100k_v3/constgold_plus_selected_sample_100k.parquet \
    --output "$input" \
    --measurement-model "$flow" \
    --likelihood-config "$likelihood" \
    --injected-g1 0.02 --injected-g2 0.0 --pixel-size 0.2
fi

# The V3.3-like flow needs its own whole-prior Gaussian coordinate cache.  Build
# it once before concurrent observation workers so they never race to create it.
if [[ ! -f "$cache/model_cache/manifest.json" || ! -f "$cache/proposal_cache/manifest.json" ]]; then
  "$python" -u "$repo/scripts/run_inference.py" \
    --inference-config "$repo/configs/inference_v1_2.json" \
    --likelihood-config "$likelihood" \
    --scene-store "$prior_root/scene_store" \
    --measurement-model "$flow" \
    --model-cache "$cache/model_cache" \
    --proposal-cache "$cache/proposal_cache" \
    --blend-response-cache "$blend" \
    --prepare-only --n-detected 1 --device cuda \
    --output "$cache/prepare"
fi

export SBSI_PYTHON="$python"
export SOURCE_MOCK="$input"
export CHAIN_ROOT="$root/hybrid"
export N_OBJECTS=100000
export MAG_CUT=22.0
export PASSES=4
export TAG=constgold_plus100k_v33like_proxytop_k1024_m16384_iter0_4_exactnorm_bfgs_v1
export INFERENCE_CONFIG="$repo/configs/inference_v1_2.json"
export LIKELIHOOD_CONFIG="$likelihood"
export MEASUREMENT_MODEL="$flow"
export MODEL_CACHE="$cache/model_cache"
export PROPOSAL_CACHE="$cache/proposal_cache"
export ADAPTIVE_DRAW_LADDER="512 1024 2048 4096 8192 16384"
export RESUME=1
unset SELECTION_NORMALIZATION_CACHE
export BUILD_SELECTION_NORMALIZATION_QUADRATIC=0
export SCORE_ROOT_BFGS=1
export BLEND_CACHE="$blend"
export CUT_ABS_EHAT=0.6
export CUT_BOUND="measured_mag_auto::25.8 measured_log_flux_radius:${threshold}:"
export PROPOSAL_CANDIDATE_SOURCE=whole_catalogue_gaussian_proxy

bash "$repo/jobs/job_inference_hybrid_chain.sh"

run="$CHAIN_ROOT/$TAG"
"$python" - "$run" "$threshold" <<'CHECK'
import json
import sys
from pathlib import Path

run = Path(sys.argv[1])
threshold = float(sys.argv[2])
chain = json.loads((run / "chain.json").read_text())
result = json.loads((run / "pass0_p0" / "result.json").read_text())
proposal = result["pipeline_config"]["proposal"]
estimator = result["pipeline_config"]["estimator"]
expected_cut = (
    "|xhat|<0.6;measured_mag_auto:None:25.8;"
    f"measured_log_flux_radius:{threshold!r}:None"
)
if chain.get("likelihood_release") != "v3.3-like":
    raise SystemExit("chain did not use the V3.3-like likelihood")
if chain["n_objects"] != 100000 or len(chain["chain"]) != 5:
    raise SystemExit("chain is not the requested five-state 100k run")
if [row["pass"] for row in chain["chain"]] != list(range(5)):
    raise SystemExit("chain does not contain iterations 0--4")
if result["likelihood_component_sha256"]["measurement"] != (
    "4e816ba5cd4be86771008fbbf5f7fec26d3d39cfb2fd90afe52acd0012373d73"
):
    raise SystemExit("wrong V3.3-like measurement flow")
if result["likelihood_component_sha256"]["detection"] != (
    "9966cfbc191f11b049bf7419dbdb45d65d1262428889a91bb3c9caf928703455"
):
    raise SystemExit("wrong V3.3-like transition detector")
if proposal.get("candidates") != 1024:
    raise SystemExit("K is not 1024")
if proposal.get("candidate_source") != "whole_catalogue_gaussian_proxy":
    raise SystemExit("direct whole-catalogue proxy shortlist is absent")
if proposal.get("prefilter_candidates") is not None:
    raise SystemExit("direct shortlist unexpectedly retained a prefilter")
if estimator.get("draw_ladder") != [512, 1024, 2048, 4096, 8192, 16384]:
    raise SystemExit("complement ladder does not end at M=16384")
if result["selection"]["cut_key"] != expected_cut:
    raise SystemExit("measured selection differs from the original 100k run")
for pass_index in range(5):
    worker = json.loads((run / f"pass{pass_index}_p0" / "result.json").read_text())
    normalization = worker["selection"]["normalization_cache"]
    if normalization is None or not normalization.get("method", "").startswith("exact_"):
        raise SystemExit(f"pass {pass_index} did not use exact normalization")
    if worker["selection"]["normalization_surrogate"] is not None:
        raise SystemExit(f"pass {pass_index} unexpectedly used a surrogate")
print(json.dumps(chain, indent=2))
CHECK
