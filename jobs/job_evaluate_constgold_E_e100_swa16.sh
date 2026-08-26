#!/usr/bin/env bash
# Direct antithetic ConstGold response closure for the 100-epoch arm-E model.

#SBATCH --job-name=sbsi_e100_constgold
#SBATCH --partition=inter,cip
#SBATCH --gres=gpu:a40-16gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/mixed_shear_cde/constgold/logs/%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi_caches/mixed_shear_cde/constgold/logs/%x_%j.err

set -euo pipefail
repo=/home/z/Zekang.Zhang/SBSI
python=/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python
root=/project/ls-gruen/users/zekang.zhang/sbsi_caches/mixed_shear_cde/constgold
model=/project/ls-gruen/users/zekang.zhang/sbsi_caches/mixed_shear_cde/measurement_flow_mixed_g0_g005_E_e100_swa16_s501_swaavg.pt
catalogue=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant/constant_response_catalogue_train.feather
crowd=/project/ls-gruen/users/zekang.zhang/sbsi_caches/crowd_flux_conc_c0-199.feather
output=$root/constgold_response_E_e100_swa16_s501_c40-139.json

[[ -s "$model" ]] || { echo "missing model $model" >&2; exit 2; }
[[ ! -e "$output" ]] || { echo "refusing to overwrite $output" >&2; exit 2; }
lookups=()
for first in $(seq 40 10 130); do
  last=$((first + 9))
  path=$root/rblend_v3_c${first}-${last}.feather
  [[ -s "$path" ]] || { echo "missing lookup shard $path" >&2; exit 2; }
  lookups+=("$path")
done
export PYTHONPATH=$repo
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
"$python" -u "$repo/scripts/evaluate_constgold_response.py" \
  --model "$model" --catalogue "$catalogue" \
  --blend-lookup "${lookups[@]}" --crowd-lookup "$crowd" \
  --output "$output" --min-case 40 --max-case 140 \
  --batch-size 65536 --n-boot 10000 --device cuda
echo "CONSTGOLD_E100_DONE output=$output"
