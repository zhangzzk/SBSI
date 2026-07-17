#!/bin/bash
#SBATCH --job-name=mu_corr_build
#SBATCH --time=03:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=16
#SBATCH --partition=inter
#SBATCH --output=/home/z/Zekang.Zhang/logs/mu_corr_build_%j.out
# Location-misfit table U(e; r-mag cell) from the g0 TRAIN sample (WORKLOG cont.33).
# CPU-only: mean-net evals dominate at ~1min/8M; streaming is the cost.
set -o pipefail
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"; cd /home/z/Zekang.Zhang/SBSI
MODEL=${MODEL:-models/measurement_flow_g0_ngmix_crowdflux_conc_snc100_central02_lam300_v1.pt}
OUT=${OUT:-/project/ls-gruen/users/zekang.zhang/sbsi_caches/mu_correction_conc_v1.npz}

echo "### mu-correction build, 8M g0 rows ###"; date
stdbuf -oL -eL python -u scripts/build_mu_correction.py \
  --measurement-model $MODEL --max-rows 8000000 --device cpu \
  --output "$OUT" 2>&1 | grep --line-buffered -vE "module command"
date
echo "### DONE ###"
