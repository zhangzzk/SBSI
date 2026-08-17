#!/bin/bash
#SBATCH --job-name=cg26_cip
#SBATCH --time=01:00:00
#SBATCH --mem=36G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a40-16gb:1
#SBATCH --partition=cip
#SBATCH --constraint=x86-64-v3
#SBATCH --array=0-159
#SBATCH --output=/home/z/Zekang.Zhang/logs/cg26_cip_%A_%a.out
set -euo pipefail

PY=/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python
export LD_LIBRARY_PATH="/project/ls-gruen/users/zekang.zhang/envs/sims1/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}"
unset PYTORCH_CUDA_ALLOC_CONF
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot

IFS=: read -r -a SEEDS <<< "${V26_SEEDS:-501:502:503:505:506:507:508:509:510:511:512:513:514:515:516:517}"
NCHUNK=10
TASK=${SLURM_ARRAY_TASK_ID:?array task required}
SEED=${SEEDS[$((TASK / NCHUNK))]}
CHUNK=$((TASK % NCHUNK))
MIN_CASE=$((40 + 10 * CHUNK))
MAX_CASE=$((MIN_CASE + 10))
TAG=ablate_s2c_lt500_v26_scene_target
CACHE=/project/ls-gruen/users/zekang.zhang/sbsi_caches/ablation
CAT=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant/constant_response_catalogue_train.feather
SHARDDIR=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v26_scene_domain_shards
CK=$CACHE/measurement_flow_g0_ngmix_${TAG}_s${SEED}_swaavg.pt
LOOKUP=results/blend_lookup_v22_c40-139.feather
CROWD=/home/z/Zekang.Zhang/SBSI/results/crowd_flux_conc_c0-199.feather
SCENE=/project/ls-gruen/users/zekang.zhang/sbsi_caches/scene/neighbor_scene_const_c40-139.feather
mkdir -p "$SHARDDIR"
for f in "$CK" "$LOOKUP" "$CROWD" "$SCENE"; do [ -f "$f" ] || { echo "MISSING $f"; exit 1; }; done
OUT=$SHARDDIR/${TAG}_perobj_s${SEED}_c$(printf '%03d' "$MIN_CASE")-$(printf '%03d' "$MAX_CASE").feather
[ ! -e "$OUT" ] || { echo "REFUSING to overwrite $OUT"; exit 1; }

echo "### V2.6 CONSTGOLD seed=$SEED cases=[$MIN_CASE,$MAX_CASE) job=$SLURM_JOB_ID ###"; nvidia-smi -L; date
"$PY" -u scripts/validate_constant_with_blend.py \
  --measurement-model "$CK" --catalogue "$CAT" \
  --min-case "$MIN_CASE" --max-case "$MAX_CASE" \
  --blend-lookup "$LOOKUP" --crowd-flux-lookup "$CROWD" --scene-lookup "$SCENE" \
  --global-only --flow-seed 12345 --n-samples 64 --max-rows 0 --batch-size 16384 \
  --dump "$OUT"
echo "CG_V26_CIP_SHARD_DONE seed=$SEED cases=[$MIN_CASE,$MAX_CASE)"; date
