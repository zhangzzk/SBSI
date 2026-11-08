#!/usr/bin/env bash
# Five-state (iterations 0--4) 100k complete-likelihood hybrid using direct
# whole-catalogue Gaussian-proxy top K and exact distributed selection
# normalization at every iteration centre.

#SBATCH --job-name=sbsi_proxytop_size075_100k
#SBATCH --partition=inter
#SBATCH --gres=gpu:a40:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=8:00:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/inference_proxy_topk_size075_n100k_v1/logs/%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/inference_proxy_topk_size075_n100k_v1/logs/%x_%j.err

set -euo pipefail

repo=${SBSI_REPOSITORY:-/home/z/Zekang.Zhang/SBSI}
python=${SBSI_PYTHON:-/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python}
root=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/inference_proxy_topk_size075_n100k_v1
threshold=1.3217558399823195

export SOURCE_MOCK="$root/prepare_g002/mock"
export CHAIN_ROOT="$root/hybrid"
export N_OBJECTS=100000
export MAG_CUT=22.0
export PASSES=4
export TAG=complete100k_proxytop_k1024_m8192_iter0_4_exactnorm_bfgs_v1
unset RESUME
unset SELECTION_NORMALIZATION_CACHE
export BUILD_SELECTION_NORMALIZATION_QUADRATIC=0
export SCORE_ROOT_BFGS=1
export BLEND_CACHE=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/blend_response_inference_scene_v1/blend_response_v1
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
expected_cut = (
    "|xhat|<0.6;measured_mag_auto:None:25.8;"
    f"measured_log_flux_radius:{threshold!r}:None"
)
if chain["n_objects"] != 100000:
    raise SystemExit("chain is not the requested 100k window")
if len(chain["chain"]) != 5 or [row["pass"] for row in chain["chain"]] != list(range(5)):
    raise SystemExit("chain does not contain iterations 0--4")
if result["pipeline_release"] != "custom" or result["pipeline_base_release"] != "v1.2-infer":
    raise SystemExit("direct shortlist was not recorded as a custom V1.2 arm")
if proposal.get("candidate_source") != "whole_catalogue_gaussian_proxy":
    raise SystemExit("whole-catalogue proxy shortlist is absent from provenance")
if proposal.get("prefilter_candidates") is not None:
    raise SystemExit("the direct shortlist unexpectedly retained a prefilter")
if result["selection"]["cut_key"] != expected_cut:
    raise SystemExit("size cut is absent from the inference likelihood")
for pass_index in range(5):
    worker = json.loads((run / f"pass{pass_index}_p0" / "result.json").read_text())
    normalization = worker["selection"]["normalization_cache"]
    if normalization is None or not normalization.get("method", "").startswith("exact_"):
        raise SystemExit(f"pass {pass_index} did not use exact normalization")
    if worker["selection"]["normalization_surrogate"] is not None:
        raise SystemExit(f"pass {pass_index} unexpectedly used a surrogate")
print(json.dumps(chain, indent=2))
CHECK
