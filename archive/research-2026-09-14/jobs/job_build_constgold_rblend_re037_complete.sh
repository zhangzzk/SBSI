#!/usr/bin/env bash
# Rebuild Re>0.37 ConstGold response lookup while preserving isolated primaries.

#SBATCH --job-name=sbsi_cgrb_r037c
#SBATCH --partition=cluster
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=02:00:00
#SBATCH --array=0-9
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/plain_complete_flow_response_re037_v1/constgold/logs/%x_%A_%a.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi_caches/plain_complete_flow_response_re037_v1/constgold/logs/%x_%A_%a.err

set -euo pipefail
repo=/home/z/Zekang.Zhang/SBSI
python=/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python
root=/project/ls-gruen/users/zekang.zhang/sbsi_caches/plain_complete_flow_response_re037_v1/constgold
loader_flow=/project/ls-gruen/users/zekang.zhang/sbsi_caches/mixed_shear_cde/measurement_flow_mixed_g0_g005_E_s501_swaavg.pt
emulator_root=/project/ls-gruen/users/zekang.zhang/blendemu_runs/extnbr_reweighted_vector_re037_optuna30_v1/models
emulator=$emulator_root/regression_model_lsst_r_extnbr_re037.json
metadata=$emulator_root/emulator_metadata_lsst_r_extnbr_re037.json
emulator_sha256=167640dd0b876ed54721493c52f3b9fe9412d005b4973e7948566eb3c4001bde
input_pattern=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant/case{case}_0.02/real0/catalogues/input/gals_info_tile180.0_-0.5.feather

first=$((40 + 10 * SLURM_ARRAY_TASK_ID))
last=$((first + 9))
output=$root/rblend_re037_complete_c${first}-${last}.feather
[[ ! -e "$output" ]] || { echo "refusing to overwrite $output" >&2; exit 2; }
mapfile -t cases < <(seq "$first" "$last")
export PYTHONPATH=$repo:/home/z/Zekang.Zhang/blendemu
export LD_LIBRARY_PATH=/project/ls-gruen/users/zekang.zhang/envs/sims1/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export XGB_DEVICE=cpu
"$python" -u "$repo/scripts/build_constgold_blend_lookup.py" \
  --cases "${cases[@]}" --input-pattern "$input_pattern" \
  --measurement-model "$loader_flow" --emulator-metadata "$metadata" \
  --emulator-model "$emulator" --emulator-sha256 "$emulator_sha256" \
  --output "$output"
echo "CONSTGOLD_RBLEND_RE037_COMPLETE_DONE cases=$first-$last output=$output"
