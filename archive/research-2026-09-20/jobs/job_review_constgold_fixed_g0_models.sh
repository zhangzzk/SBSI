#!/usr/bin/env bash
# Validate and tabulate the completed fixed-g0 ConstGold model comparison.

#SBATCH --job-name=sbsi_cgfg0_review
#SBATCH --partition=cluster,inter
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/fixed_g0_m258_r060_v2/constgold_response_c40_89_v1/logs/%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi_caches/fixed_g0_m258_r060_v2/constgold_response_c40_89_v1/logs/%x_%j.err

set -euo pipefail
repo=/home/z/Zekang.Zhang/SBSI
python=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
root=/project/ls-gruen/users/zekang.zhang/sbsi_caches/fixed_g0_m258_r060_v2/constgold_response_c40_89_v1
json_output=$root/review.json
markdown_output=$root/review.md
[[ ! -e "$json_output" ]] || { echo "refusing to overwrite $json_output" >&2; exit 2; }
[[ ! -e "$markdown_output" ]] || { echo "refusing to overwrite $markdown_output" >&2; exit 2; }

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-2}
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONPATH="$repo${PYTHONPATH:+:$PYTHONPATH}"
cd "$repo"
"$python" -u -m scripts.review_constgold_fixed_g0_response \
  --result "$root/result.json" \
  --measured-audit "$root/measured_audit.json" \
  --truth-pairing-audit "$root/truth_pairing_audit.json" \
  --expected-model old_fixed_v1_nll \
  --expected-model new_v2_nll \
  --expected-model new_v2_staged \
  --expected-model new_v2_direct \
  --output-json "$json_output" \
  --output-markdown "$markdown_output"

echo "CONSTGOLD_FIXED_G0_REVIEW_DONE json=$json_output markdown=$markdown_output"
