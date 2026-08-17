#!/bin/bash
#SBATCH --job-name=cg22_cip
#SBATCH --time=01:00:00
#SBATCH --mem=36G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a40-16gb:1
#SBATCH --partition=cip
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/cg22_cip_%a_%j.out
set -euo pipefail

PY=/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python
export LD_LIBRARY_PATH="/project/ls-gruen/users/zekang.zhang/envs/sims1/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}"
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot

IFS=: read -r -a SEEDS <<< "${V22_SEEDS:-501:502}"
NCHUNK=10
TASK=${SLURM_ARRAY_TASK_ID:?array task required}
SEED=${SEEDS[$((TASK / NCHUNK))]}
CHUNK=$((TASK % NCHUNK))
MIN_CASE=$((40 + 10 * CHUNK))
MAX_CASE=$((MIN_CASE + 10))
TAG=ablate_s2c_lt500_v22
CACHE=/project/ls-gruen/users/zekang.zhang/sbsi_caches/ablation
CAT=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant/constant_response_catalogue_train.feather
SHARDDIR=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v22_domain_shards
CK="$CACHE/measurement_flow_g0_ngmix_${TAG}_s${SEED}_swaavg.pt"
LOOKUP=results/blend_lookup_v22_c40-139.feather
CROWD=/home/z/Zekang.Zhang/SBSI/results/crowd_flux_conc_c0-199.feather
mkdir -p "$SHARDDIR"
for f in "$CK" "$LOOKUP" "$CROWD"; do [ -f "$f" ] || { echo "MISSING $f"; exit 1; }; done
OUT="$SHARDDIR/${TAG}_perobj_s${SEED}_c$(printf '%03d' "$MIN_CASE")-$(printf '%03d' "$MAX_CASE").feather"
[ ! -e "$OUT" ] || { echo "REFUSING to overwrite $OUT"; exit 1; }

echo "### V2.2 CIP SHARD seed=$SEED cases=[$MIN_CASE,$MAX_CASE) job=$SLURM_JOB_ID ###"
nvidia-smi -L; date
"$PY" -u scripts/validate_constant_with_blend.py \
  --measurement-model "$CK" --catalogue "$CAT" \
  --min-case "$MIN_CASE" --max-case "$MAX_CASE" \
  --blend-lookup "$LOOKUP" --crowd-flux-lookup "$CROWD" \
  --global-only --flow-seed 12345 --n-samples 64 --max-rows 0 --batch-size 16384 \
  --dump "$OUT"
echo "CG_V22_CIP_SHARD_DONE seed=$SEED cases=[$MIN_CASE,$MAX_CASE)"; date
