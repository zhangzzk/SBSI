#!/usr/bin/env bash
# Regenerate the cont.299 frozen 500k likelihood mock under the cont.303 code.
#
# WHY THIS EXISTS.  The cont.306/cont.307 candidate sweep died at 33 s on
#
#   RuntimeError: likelihood mock was generated with different scene/model/
#   selection settings
#
# raised by `_validate_loaded_likelihood_manifest` (scripts/run_inference.py:448).
# The gate is correct and the mock's science is unchanged; what changed is the
# provenance schema.  The cont.303 restructure added two fields to
# `likelihood_mock_identity` (`likelihood_release`, `likelihood_config_sha256`,
# run_inference.py:1025-1046) that did not exist when this mock was frozen, so
# the dict comparison cannot match no matter what the data contain.  The
# implementation hash list changed too: the stored manifest names
# `configs/infer_v1.json`, `sbsi/amortized_proposal.py`, `sbsi/score_inference.py`
# and `scripts/run_section5_numerical_recenter.py`, three of which no longer
# exist in the tree.
#
# The eleven shared identity fields are all still satisfied -- verified by
# rehashing the artifacts the manifest names:
#
#   scene_store/galaxies.parquet   fae16fcd...f968   MATCHES
#   scene_store/manifest.json      dea4295c...9002   MATCHES
#   scene_store/neighbours.npz     4403f2bc...c88a   MATCHES
#   model_cache/detection_zero.pq  b98bdde6...0f98   MATCHES
#   model_cache/flow_zero.parquet  50b8842b...a070   MATCHES
#   model_cache/manifest.json      40129ac0...25a7   MATCHES
#
# So the mock is scientifically the same object; only its paperwork predates the
# restructure.
#
# WHY REGENERATE RATHER THAN RELAX THE GATE.  cont.256 hit exactly this class of
# failure and set the precedent: "The gate was not weakened and the old manifest
# was not edited."  Editing `implementation_sha256` in place would assert this
# mock was produced by code that in fact never touched it, which is a false
# provenance record, and adding an `--allow-legacy-mock` escape hatch would
# weaken a gate that is doing its job.  Regeneration is also cheap here: the
# original preparation, job 16058006, took 40 s.
#
# V100 PIN.  cont.256 found that regenerating on a different GPU type reproduced
# the truth table byte-for-byte but NOT the stochastic measurement table.  The
# original preparation ran on th-cl-nv02, which `sinfo` reports as gpu:v100:4
# (hence the `_a40_v100_` in the source path).  This job therefore pins v100.
#
# ACCEPTANCE.  This regeneration is only usable if it reproduces the frozen mock
# exactly.  Compare against the stored `output_sha256`:
#
#   measurements.parquet  fdfecacc13adabd6005e21150ab0dc74767cad34da3d3ea95e362515ce38365b
#   truth.parquet         78d1c8e02f43e1ac4673b2ca309b484d3565de3b9400bf736ca98bf584e5dc30
#
# Both matching means the candidate sweep stays paired with the cont.304/305 M
# ladder under common random numbers (doc/CONVENTIONS.md section 6d) and the
# cont.306 validation anchor remains checkable.  If `truth.parquet` matches but
# `measurements.parquet` does not, this is the cont.256 GPU-stochasticity case:
# the sweep would still be internally valid but the anchor would be void, and
# that must be recorded rather than worked around.  The job verifies this itself
# and exits non-zero on any mismatch.
#
#   sbatch jobs/job_prepare_stacked_n500k_mock.sh

#SBATCH --job-name=sbsi_prepare_stacked_n500k_mock
#SBATCH --partition=inter
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=02:00:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/inference_candidate_ladder_n125k_v1/logs/%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/inference_candidate_ladder_n125k_v1/logs/%x_%j.err

set -euo pipefail

repo=${SBSI_REPOSITORY:-/home/z/Zekang.Zhang/SBSI}
python=${SBSI_PYTHON:-/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python}

source_root=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/v31_fs2_default_qmc128_n100k_v1
compact="$source_root/compact_global"
reference=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/infer_v1_stacked_n500k_a40_v100_v1/prepare/mock
root=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/inference_candidate_ladder_n125k_v1
output="$root/prepare_cont303"

if [[ -e "$output" ]]; then
  echo "refusing to overwrite $output" >&2
  exit 2
fi
mkdir -p "$root"

# See cont.307: py31 carries no editable install, so the package must be put on
# the path by hand.  Delete once doc/ENVIRONMENT.md#installation is satisfied.
export PYTHONPATH="$repo:${BLENDEMU_ROOT:-/home/z/Zekang.Zhang/blendemu}${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

"$python" - <<'PREFLIGHT' || exit 2
import sbsi, blendemu  # noqa: F401
PREFLIGHT

# Seeds, injection and object count are copied from the frozen manifest so this
# is a reproduction, not a new draw.
"$python" "$repo/scripts/run_inference.py" \
  --inference-config "$repo/configs/inference.json" \
  --likelihood-config "$repo/configs/likelihood.json" \
  --scene-store "$compact/scene_store" \
  --measurement-model /project/ls-gruen/users/zekang.zhang/sbsi_caches/mixed_shear_cde/measurement_flow_mixed_g0_g005_E_s501_swaavg.pt \
  --model-cache "$compact/model_cache" \
  --proposal-cache "$compact/proposal_cache" \
  --output "$output" \
  --n-detected 500000 \
  --injected-g1 0.02 \
  --injected-g2 0.0 \
  --scene-seed 12001 \
  --detection-seed 12002 \
  --flow-seed 12003 \
  --prepare-only \
  --device cuda

# Locate the frozen mock the runner just wrote.
mock=$(dirname "$(find "$output" -name likelihood_mock_manifest.json -print -quit)")
if [[ -z "$mock" || ! -f "$mock/likelihood_mock_manifest.json" ]]; then
  echo "preparation produced no likelihood_mock_manifest.json under $output" >&2
  exit 2
fi
echo "regenerated mock: $mock"

status=0
for f in measurements.parquet truth.parquet; do
  new=$(sha256sum "$mock/$f" | cut -d' ' -f1)
  old=$(sha256sum "$reference/$f" | cut -d' ' -f1)
  if [[ "$new" == "$old" ]]; then
    printf 'MATCH     %-22s %s\n' "$f" "$new"
  else
    printf 'MISMATCH  %-22s new=%s old=%s\n' "$f" "$new" "$old"
    status=1
  fi
done

if (( status != 0 )); then
  echo "regenerated mock does NOT reproduce the frozen mock; see cont.256 GPU stochasticity" >&2
  exit 1
fi
echo "regenerated mock reproduces the cont.299 frozen mock byte-for-byte"
