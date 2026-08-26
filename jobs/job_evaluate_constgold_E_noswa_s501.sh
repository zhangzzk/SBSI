#!/usr/bin/env bash
# Direct ConstGold response closure for original E's best-val, non-SWA checkpoint.

#SBATCH --job-name=sbsi_e_noswa_cg
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
cache=/project/ls-gruen/users/zekang.zhang/sbsi_caches
root=$cache/mixed_shear_cde/constgold
model=$cache/mixed_shear_cde/measurement_flow_mixed_g0_g005_E_s501_swabase.pt
catalogue=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant/constant_response_catalogue_train.feather
crowd=$cache/crowd_flux_conc_c0-199.feather
output=$root/constgold_response_E_noswa_bestep78_s501_c40-139.json

[[ -s "$model" ]] || { echo "missing model $model" >&2; exit 2; }
[[ -s "$catalogue" ]] || { echo "missing catalogue $catalogue" >&2; exit 2; }
[[ -s "$crowd" ]] || { echo "missing crowd lookup $crowd" >&2; exit 2; }
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
echo "CONSTGOLD_E_NOSWA_DONE output=$output"
