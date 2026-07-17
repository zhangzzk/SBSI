#!/bin/bash
#SBATCH --job-name=flow_resp_mag
#SBATCH --time=04:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=16
#SBATCH --partition=inter
#SBATCH --output=/home/z/Zekang.Zhang/logs/flow_resp_mag_%j.out
# CPU-only on purpose: 2 mean-net evals/object, no grid -- streaming dominates.
# 4M rows + seed 7 reproduces stage B's exact gold sample (job 15064545), so the
# per-bin R_sim/R_flow here joins 1:1 with the etilde dump for the leak-model test
#   m_pred(bin) = K_intr(bin) * (1 - R_flow/R_sim)(bin)   vs measured m_she(bin).
set -o pipefail
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"; cd /home/z/Zekang.Zhang/SBSI
MODEL=${MODEL:-models/measurement_flow_g0_ngmix_crowdflux_conc_snc100_central02_lam300_v1.pt}
OUT=${OUT:-/project/ls-gruen/users/zekang.zhang/sbsi_caches/flow_response_by_mag_r4M.csv}

echo "### flow-vs-sim mean response by magnitude, 4M rows ###"; date
stdbuf -oL -eL python -u scripts/flow_response_by_mag.py \
  --measurement-model $MODEL --max-rows 4000000 --seed 7 --device cpu \
  --output "$OUT" --dump-objects "${OUT%.csv}_objects.feather" 2>&1 | grep --line-buffered -vE "module command"
date
echo "### DONE ###"
