#!/usr/bin/env bash
# GPU-memory and end-to-end smoke test for the direct whole-catalogue shortlist.
# It deliberately omits measured selection so the test exercises ranking and
# inference without spending an hour on a full-prior normalization stencil.
# N=1,024 is large enough for the catalogue information matrix to be stable;
# the original N=128 smoke reached the summary but was too noisy to be positive
# definite.

#SBATCH --job-name=sbsi_proxytop_smoke1024
#SBATCH --partition=inter
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=2:00:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/inference_proxy_topk_size075_n20k_v1/logs/%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/inference_proxy_topk_size075_n20k_v1/logs/%x_%j.err

set -euo pipefail

repo=${SBSI_REPOSITORY:-/home/z/Zekang.Zhang/SBSI}
python=${SBSI_PYTHON:-/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python}
source_root=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/v31_fs2_default_qmc128_n100k_v1
compact="$source_root/compact_global"
mock=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/inference_candidate_ladder_n125k_v1/prepare_cont303/mock
root=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/inference_proxy_topk_size075_n20k_v1
output="$root/smoke_proxy_topk_n1024"

if [[ -e "$output" ]]; then
  echo "refusing to overwrite $output" >&2
  exit 2
fi

export PYTHONPATH="$repo:${BLENDEMU_ROOT:-/home/z/Zekang.Zhang/blendemu}${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

"$python" "$repo/scripts/run_inference.py" \
  --inference-config "$repo/configs/inference_v1_2.json" \
  --likelihood-config "$repo/configs/likelihood.json" \
  --scene-store "$compact/scene_store" \
  --measurement-model /project/ls-gruen/users/zekang.zhang/sbsi_caches/mixed_shear_cde/measurement_flow_mixed_g0_g005_E_s501_swaavg.pt \
  --model-cache "$compact/model_cache" \
  --proposal-cache "$compact/proposal_cache" \
  --mock-input "$mock" \
  --proposal-candidate-source whole_catalogue_gaussian_proxy \
  --observation-start 0 \
  --observation-stop 1024 \
  --output "$output" \
  --device cuda

"$python" - "$output/result.json" <<'CHECK'
import json
import sys

result = json.load(open(sys.argv[1]))
proposal = result["pipeline_config"]["proposal"]
if result["pipeline_release"] != "custom":
    raise SystemExit("smoke did not record the custom pipeline")
if proposal.get("candidate_source") != "whole_catalogue_gaussian_proxy":
    raise SystemExit("smoke did not use the direct proxy shortlist")
if proposal.get("prefilter_candidates") is not None:
    raise SystemExit("smoke unexpectedly used a prefilter")
if result["observation_partition"]["n_partition"] != 1024:
    raise SystemExit("smoke observation count is wrong")
print("direct proxy shortlist smoke passed")
CHECK
