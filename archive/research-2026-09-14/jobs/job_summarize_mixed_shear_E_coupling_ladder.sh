#!/usr/bin/env bash
# Select the density-admissible coupling-ladder arm after all held-out audits.

#SBATCH --job-name=sbsi_Etheta_pick
#SBATCH --partition=cip
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/mixed_shear_e_coupling_ladder_v1/logs/%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi_caches/mixed_shear_e_coupling_ladder_v1/logs/%x_%j.err

set -euo pipefail

repo=/home/z/Zekang.Zhang/SBSI
python=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
phot=/project/ls-gruen/users/zekang.zhang/sbsi_caches/mixed_shear_e_photometric
root=/project/ls-gruen/users/zekang.zhang/sbsi_caches/mixed_shear_e_coupling_ladder_v1

"$python" "$repo/scripts/summarize_flow_coupling_ladder.py" \
  --root "$root" \
  --baseline-audit "$phot/heldout_derivative_losses_E_original_s501_v2.json" \
  --endpoint-audit "$phot/heldout_derivative_losses_E_r450_t500_s501_v2.json" \
  --endpoint-audit "$phot/heldout_derivative_losses_E_r500_t500_s501_v2.json" \
  --base-likelihood "$repo/configs/likelihood_v3_3_like.json" \
  --output "$root/summary.json" \
  --likelihood-output "$root/selected_likelihood.json" \
  --nll-margin 0.05 --shape-factor 1.05

echo "FLOW_E_COUPLING_LADDER_SELECTION_DONE output=$root/summary.json"
