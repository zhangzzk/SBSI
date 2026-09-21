#!/usr/bin/env bash

#SBATCH --job-name=sbsi_Etheta_Icmp
#SBATCH --partition=cip
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/mixed_shear_e_coupling_ladder_v1/inference_n20k/logs/%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi_caches/mixed_shear_e_coupling_ladder_v1/inference_n20k/logs/%x_%j.err

set -euo pipefail
repo=/home/z/Zekang.Zhang/SBSI
python=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
root=/project/ls-gruen/users/zekang.zhang/sbsi_caches/mixed_shear_e_coupling_ladder_v1/inference_n20k

"$python" "$repo/scripts/compare_flow_inference_screen.py" \
  --root "$root" --output "$root/comparison.json"
echo "CONSTGOLD_FLOW_SCREEN_20K_COMPARISON_DONE output=$root/comparison.json"
