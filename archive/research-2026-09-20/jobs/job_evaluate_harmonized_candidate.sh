#!/usr/bin/env bash
# Frozen diagnostics only; never feed ConstGold back into checkpoint selection.
#SBATCH --job-name=sbsi_harmonized_candidate
#SBATCH --partition=inter,cip
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/harmonized_secondary_20260917_v1/logs/%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi_caches/harmonized_secondary_20260917_v1/logs/%x_%j.err
set -euo pipefail
data=/project/ls-gruen/users/zekang.zhang
cache=$data/sbsi_caches
root=$cache/harmonized_secondary_20260917_v1
parent=$cache/constgold_truth_parent_20260917_v1
baseline=$cache/full_domain_true_mag26_guard_v1/flow/guard/selected.pt
python=$data/envs/py31/bin/python
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /home/z/Zekang.Zhang/SBSI
mode=${1:?identity, density_smoke, density, joint_smoke, or joint}
case $mode in
  identity|density_smoke|joint_smoke) cases=(40 41) ;;
  density|joint) mapfile -t cases < <(seq 40 119) ;;
  *) exit 2 ;;
esac
if [[ $mode == identity ]]; then
  flow=$baseline
else
  flow=$root/refinement/selected.pt
  step=$("$python" - "$root/refinement/completed.json" "$flow" <<'PY'
import hashlib
import json
from pathlib import Path
import sys
record = json.loads(Path(sys.argv[1]).read_text())
if record['full_density_gate_passed'] is not True:
    raise RuntimeError('selected candidate failed full validation density gate')
if hashlib.sha256(Path(sys.argv[2]).read_bytes()).hexdigest() != record['selected_sha256']:
    raise RuntimeError('selected checkpoint changed since refinement')
print(record['best_step'])
PY
  )
  if [[ $step == 0 ]]; then
    echo "HARMONIZED_CANDIDATE_SKIPPED initial checkpoint selected; baseline disk candidate already evaluated"
    exit 0
  fi
fi
case $mode in
  identity|density_smoke|density)
    "$python" -u -m scripts.audit_disk_candidate_density \
      --flow "$flow" --reference-flow "$baseline" \
      --cases "${cases[@]}" --output "$root/candidate_${mode}.json"
    if [[ $mode == identity ]]; then
      "$python" - "$root/candidate_identity.json" <<'PY'
import json
import sys
with open(sys.argv[1]) as stream:
    result = json.load(stream)
if result['flow_sha256'] != result['reference_flow_sha256']:
    raise RuntimeError('identity check requires identical flow artifacts')
for report in result['reports']:
    for values in report['cuts'].values():
        if values['paired_delta_nll'] != 0. or values['paired_object_se'] != 0.:
            raise RuntimeError('transported reference identity failed')
print('HARMONIZED_DENSITY_REFERENCE_IDENTITY_PASSED')
PY
    fi
    ;;
  joint_smoke|joint)
    options=()
    for seed in 20260913 20260914 20260915; do
      options+=(--classifier "$cache/fixed_g0_m258_r060_v1/classifier/seed${seed}/selected.pt")
    done
    if [[ $mode == joint_smoke ]]; then
      options+=(--rblend "$cache/output_conditioned_response_20260917_v1/joint_smoke_rblend.feather")
    else
      for first in $(seq 40 10 110); do
        options+=(--rblend "$parent/rblend/cases${first}_$((first+9)).feather")
      done
    fi
    "$python" -u -m scripts.evaluate_constgold_truth_parent \
      --parent-root "$parent/parent_v2" --constgold-root "$data/lsst_sims_fs2_25876_constant" \
      --flow "$flow" --response-candidate "$data/blendemu_runs/disk_response_moment_20260918_v1/model" \
      --response-composition mobius --response-device cuda --response-threads 8 \
      --pair-root "$cache/output_conditioned_response_20260917_v1/pairs" \
      "${options[@]}" --cases "${cases[@]}" --output "$root/candidate_${mode}.json"
    if [[ $mode == joint ]]; then
      "$python" -u -m scripts.compare_truth_parent_response \
        --baseline "$cache/disk_response_moment_20260918_v1/joint_full.json" \
        --candidate "$root/candidate_joint.json" --output "$root/candidate_joint_comparison.json"
    fi
    ;;
esac
