#!/usr/bin/env bash
# One-shot scheduler/log capture; submit with --begin=now+2hours.
#SBATCH --job-name=sbsi_popguard_check
#SBATCH --partition=cluster,inter
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:05:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/population_guard_ab_20260917_v1/logs/%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi_caches/population_guard_ab_20260917_v1/logs/%x_%j.err
set -euo pipefail
root=/project/ls-gruen/users/zekang.zhang/sbsi_caches/population_guard_ab_20260917_v1
jobs=${1:?comma-separated training/evaluation/summary job IDs required}
date --iso-8601=seconds
squeue -j "$jobs" -o '%.20i %.24j %.10T %.10M %.10l %.6D %b %R' || true
sacct -j "$jobs" --format=JobID,JobName%24,State,ExitCode,Elapsed,MaxRSS -n -P
shopt -s nullglob
for log in "$root"/logs/sbsi_popguard_ab_*.out "$root"/logs/sbsi_popguard_cg_*.out "$root"/logs/sbsi_popguard_sum_*.out; do
  echo "$log"
  tail -n 5 "$log"
done
for log in "$root"/logs/sbsi_popguard_ab_*.err "$root"/logs/sbsi_popguard_cg_*.err "$root"/logs/sbsi_popguard_sum_*.err; do
  if [[ -s $log ]]; then
    echo "$log"
    tail -n 20 "$log"
  fi
done
for audit in guard_control_variate_20260917_v1 neighbour_amplitude_audit_20260917_v1 output_conditioned_response_20260917_v1 disk_response_moment_20260918_v1 harmonized_secondary_20260917_v1 flow_shape_augmentation_20260918_v1; do
  audit_root=/project/ls-gruen/users/zekang.zhang/sbsi_caches/$audit
  for log in "$audit_root"/logs/*.out "$audit_root"/logs/*.err; do
    if [[ -s $log ]]; then
      echo "$log"
      tail -n 5 "$log"
    fi
  done
  for result in "$audit_root/result.json" "$audit_root/audit_v2/result.json" "$audit_root/refinement/completed.json" "$audit_root/refinement_truth_parent_response.json" "$audit_root/label_amplitude.json" "$audit_root/joint_truth_only_comparison.json" "$audit_root/joint_output_conditioned_comparison.json" "$audit_root/measurement_seed_case0.json" "$audit_root/measurement_seed_case1.json"; do
    if [[ -f $result ]]; then
      echo "AUDIT_RESULT_AVAILABLE $result"
    fi
  done
  if [[ -f $audit_root/common_optimizer_pilot.json ]]; then
    echo "COMMON_OPTIMIZER_PILOT_AVAILABLE $audit_root/common_optimizer_pilot.json"
  fi
  for result in centering_pilot.json centering_fingerprint_summary.json flow_measurement_provenance.json secondary_measurement_pilot.json conditional_support.json harmonized_pilot_density.json joint_smoke.json joint_full.json joint_comparison.json joint_latent_seed7302.json joint_latent_repeat_comparison.json native_smoke.json native_result.json density_smoke.json density_full.json primary_amplitude_pilot.json fresh_amplitude_pilot.json candidate_identity.json candidate_density.json candidate_joint.json candidate_joint_comparison.json independent_baseline_smoke.json independent_candidate_smoke.json independent_baseline_full.json independent_candidate_full.json independent_joint_comparison.json independent_pair_preparation.json evaluator_staging/validation.json shape_symmetry.json measurement_symmetry_pilot.json control/completed.json augmented/completed.json full_density.json guard_reference/result.json; do
    if [[ -f $audit_root/$result ]]; then
      echo "MEASUREMENT_AUDIT_AVAILABLE $audit_root/$result"
    fi
  done
done
emulator_root=/project/ls-gruen/users/zekang.zhang/blendemu_runs/output_conditioned_response_20260917_v1
for log in "$emulator_root"/logs/*.out "$emulator_root"/logs/*.err; do
  if [[ -s $log ]]; then
    echo "$log"
    tail -n 5 "$log"
  fi
done
if [[ -f $emulator_root/models_cpu/completed.json ]]; then
  echo "EMULATOR_COMPARISON_TRAINED $emulator_root/models_cpu/completed.json"
fi
if [[ -f "$root/comparison.json" ]]; then
  echo "JOINT_COMPARISON_AVAILABLE $root/comparison.json"
fi
pilot_root=/project/ls-gruen/users/zekang.zhang/blendemu_runs/independent_g005_subpixel_pilot_20260918_v1
for log in "$pilot_root"/logs/*.out "$pilot_root"/logs/*.err; do
  if [[ -s $log ]]; then
    echo "$log"
    tail -n 5 "$log"
  fi
done
/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python - <<'PY'
from pathlib import Path
base = Path('/project/ls-gruen/users/zekang.zhang')
root = base/'blendemu_runs/harmonized_secondary_20260917_v1'
cases = [case for case in range(200) if (root/f'case{case:03d}_manifest.json').is_file()]
print('HARMONIZED_MEASUREMENT_CASES_COMPLETED', len(cases), 'of 200')
prepared = base/'sbsi_caches/harmonized_secondary_20260917_v1/domain/manifest.json'
print('HARMONIZED_FULL_DOMAIN_AVAILABLE', prepared.is_file(), str(prepared))
model = base/'blendemu_runs/disk_response_moment_20260918_v1/model/metadata.json'
print('DISK_RESPONSE_MODEL_AVAILABLE', model.is_file(), str(model))
pilot = base/'blendemu_runs/independent_g005_subpixel_pilot_20260918_v1'
print('INDEPENDENT_G005_PILOT_COMPLETED',
      [case for case in (200, 201) if (pilot/f'case{case:03d}_manifest.json').is_file()])
PY
