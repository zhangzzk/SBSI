#!/usr/bin/env bash
# Prepare the g=0 partner of the cont.299/cont.308 500k likelihood mock.
#
# WHY.  cont.314 established that ghat1 at injected g1=0.02 is 3.2 sigma high at
# 125,000 objects with no extrapolation, and that reducing sampling error moves
# it further from truth rather than closer.  The sampler is therefore not the
# binding problem and a residual of about +0.0027 is open.  One measurement
# separates its two possible causes, and no sampler work substitutes for it:
#
#   multiplicative (m):  ghat1(g=0) = 0,        ghat1(0.02) = 0.02 (1+m)
#   additive (c):        ghat1(g=0) = +0.0027,  the same offset at both shears
#
# PAIRING.  Scene, detection and flow seeds are identical to the g=0.02 mock
# (12001/12002/12003), as are the object count, scene store, model cache and
# measurement flow.  Only --injected-g1 changes.  Under doc/CONVENTIONS.md
# section 6d that makes the two mocks common-random-number partners, so the
# difference between the two shears carries far less realization noise than
# either estimate does on its own.
#
# V100 PIN.  cont.256 found the stochastic measurement table does not reproduce
# across GPU types.  The partner mock was prepared on v100 (job 16090436), so
# this one is pinned to v100 as well; drawing the pair on different hardware
# would break exactly the pairing this test depends on.
#
# THIS IS A NEW DRAW, NOT A REPRODUCTION.  There is no stored hash to match
# against, so unlike job_prepare_stacked_n500k_mock.sh this job does not compare
# against a reference.  It reports the hashes it produced and checks only that
# the runner retained the requested injected shear, which run_inference.py
# enforces itself.
#
#   sbatch jobs/job_prepare_n500k_mock_g0.sh

#SBATCH --job-name=sbsi_prepare_n500k_mock_g0
#SBATCH --partition=inter
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=02:00:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/inference_shear_null_n500k_v1/logs/%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/inference_shear_null_n500k_v1/logs/%x_%j.err

set -euo pipefail

repo=${SBSI_REPOSITORY:-/home/z/Zekang.Zhang/SBSI}
python=${SBSI_PYTHON:-/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python}

source_root=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/v31_fs2_default_qmc128_n100k_v1
compact="$source_root/compact_global"
root=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/inference_shear_null_n500k_v1
output="$root/prepare_g0"

if [[ -e "$output" ]]; then
  echo "refusing to overwrite $output" >&2
  exit 2
fi
mkdir -p "$root/logs"

# See cont.307: py31 carries no editable install.
export PYTHONPATH="$repo:${BLENDEMU_ROOT:-/home/z/Zekang.Zhang/blendemu}${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

"$python" - <<'PREFLIGHT' || exit 2
import sbsi, blendemu  # noqa: F401
PREFLIGHT

"$python" "$repo/scripts/run_inference.py" \
  --inference-config "$repo/configs/inference.json" \
  --likelihood-config "$repo/configs/likelihood.json" \
  --scene-store "$compact/scene_store" \
  --measurement-model /project/ls-gruen/users/zekang.zhang/sbsi_caches/mixed_shear_cde/measurement_flow_mixed_g0_g005_E_s501_swaavg.pt \
  --model-cache "$compact/model_cache" \
  --proposal-cache "$compact/proposal_cache" \
  --output "$output" \
  --n-detected 500000 \
  --injected-g1 0.0 \
  --injected-g2 0.0 \
  --scene-seed 12001 \
  --detection-seed 12002 \
  --flow-seed 12003 \
  --prepare-only \
  --device cuda

mock=$(dirname "$(find "$output" -name likelihood_mock_manifest.json -print -quit)")
if [[ -z "$mock" || ! -f "$mock/likelihood_mock_manifest.json" ]]; then
  echo "preparation produced no likelihood_mock_manifest.json under $output" >&2
  exit 2
fi
echo "prepared g=0 mock: $mock"
for f in measurements.parquet truth.parquet; do
  printf '%-22s %s\n' "$f" "$(sha256sum "$mock/$f" | cut -d' ' -f1)"
done
