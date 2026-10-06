#!/usr/bin/env bash
# Freeze a mock drawn from the COMPLETE catalogue likelihood.
#
# WHY.  Every closure test so far drew its mock from a likelihood that used the
# measurement flow alone.  The detection classifier was already active, but two
# terms were not:
#
#   R_blend         the neighbour-induced shape response.  configs/likelihood.json
#                   calls this "optional_fixed_atom_cache" and every prior run
#                   left it null, so r_blend and blend_shift were identically
#                   zero.  doc/CONVENTIONS.md section 1 defines
#                   R_model = R_flow + R_blend, so those runs measured the SELF
#                   response only.
#   measured cut    the |e| and magnitude selection an actual shear catalogue
#                   applies after measurement, with its population
#                   normalization B_W(g).
#
# The cache could not be built before because merge_sharded_prior_model_qmc.py
# writes the inference scene with a deliberately empty neighbour graph
# ("neighbour_graph_usage": "precomputed_in_shard_model_views_only"), so
# CatalogueBlendResponse.from_emulator found no pairs inside the aperture and
# returned all zeros.  scripts/build_sharded_blend_response.py builds the
# response per shard, where the 2,864,350,310 directed edges still exist, and
# concatenates in the shard order the merge itself uses.  Both sides are keyed
# to prior manifest sha256 9d3568495c194c7efc3d05b7b6e783640f6ea860304d4af8d0cb4eed6fcf6090.
#
# THE CUT.  |measured shape| < 0.6 and measured_mag_auto < 25.8, on the model's
# OWN sampled measurements.  Per doc/CONVENTIONS.md section 6e the model side
# must never inherit the simulation's pass mask -- predicting which objects
# survive is the thing under test.  --selection-cache stores P_pass per atom per
# trial shear; its identity binds to blend_response_sha256, so a cache built
# without R_blend cannot be silently reused here.
#
# THIS IS A NEW DRAW.  Seeds 13001/13002/13003 are distinct from every existing
# mock, so there is no stored hash to reproduce; the job reports the hashes it
# produced.
#
#   sbatch jobs/job_prepare_complete_likelihood_mock.sh

#SBATCH --job-name=sbsi_prepare_complete_mock
#SBATCH --partition=inter
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=12:00:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/inference_complete_likelihood_v1/logs/%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/inference_complete_likelihood_v1/logs/%x_%j.err

set -euo pipefail

repo=${SBSI_REPOSITORY:-/home/z/Zekang.Zhang/SBSI}
python=${SBSI_PYTHON:-/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python}

source_root=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/v31_fs2_default_qmc128_n100k_v1
compact="$source_root/compact_global"
blend=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/blend_response_inference_scene_v1/blend_response_v1
root=${PREP_ROOT:-/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/inference_complete_likelihood_v1}
output=${PREP_OUTPUT:-$root/prepare_g002}
selection=${SELECTION_CACHE:-$root/selection_cache}

if [[ -e "$output" ]]; then
  echo "refusing to overwrite $output" >&2
  exit 2
fi
if [[ ! -f "$blend/manifest.json" || ! -f "$blend/r_blend.npy" ]]; then
  echo "missing merged R_blend cache at $blend" >&2
  exit 2
fi
mkdir -p "$root/logs"

# See cont.307: py31 carries no editable install.
export PYTHONPATH="$repo:${BLENDEMU_ROOT:-/home/z/Zekang.Zhang/blendemu}${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

"$python" - <<'PREFLIGHT' || exit 2
import sbsi, blendemu  # noqa: F401
PREFLIGHT

cut_args=(--cut-bound "measured_mag_auto::25.8")
if [[ -n "${CUT_BOUND_EXTRA:-}" ]]; then
  for spec in $CUT_BOUND_EXTRA; do
    cut_args+=(--cut-bound "$spec")
  done
fi

"$python" "$repo/scripts/run_inference.py" \
  --inference-config "$repo/configs/inference.json" \
  --likelihood-config "$repo/configs/likelihood.json" \
  --scene-store "$compact/scene_store" \
  --measurement-model /project/ls-gruen/users/zekang.zhang/sbsi_caches/mixed_shear_cde/measurement_flow_mixed_g0_g005_E_s501_swaavg.pt \
  --model-cache "$compact/model_cache" \
  --proposal-cache "$compact/proposal_cache" \
  --blend-response-cache "$blend" \
  --selection-cache "$selection" \
  --cut-abs-ehat 0.6 \
  "${cut_args[@]}" \
  --output "$output" \
  --n-detected "${N_DETECTED:-125000}" \
  --injected-g1 0.02 \
  --injected-g2 0.0 \
  --scene-seed 13001 \
  --detection-seed 13002 \
  --flow-seed 13003 \
  --prepare-only \
  --device cuda

mock=$(dirname "$(find "$output" -name likelihood_mock_manifest.json -print -quit)")
if [[ -z "$mock" || ! -f "$mock/likelihood_mock_manifest.json" ]]; then
  echo "preparation produced no likelihood_mock_manifest.json under $output" >&2
  exit 2
fi
echo "prepared complete-likelihood mock: $mock"
for f in measurements.parquet truth.parquet; do
  printf '%-22s %s\n' "$f" "$(sha256sum "$mock/$f" | cut -d' ' -f1)"
done

# Refuse to report a run whose two new terms silently did nothing.
"$python" - "$mock/likelihood_mock_manifest.json" <<'CHECK'
import json, sys, pathlib
manifest = json.loads(pathlib.Path(sys.argv[1]).read_text())
identity = manifest["generation_identity"]
cut = identity.get("selection_cut_key")
sha = identity.get("blend_response_sha256")
print("selection_cut_key:", json.dumps(cut))
print("blend_response_sha256:", json.dumps(sha))
if not cut:
    raise SystemExit("the measured cut did not reach the likelihood identity")
if not sha:
    raise SystemExit("R_blend did not reach the likelihood identity")
CHECK
