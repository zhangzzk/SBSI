#!/bin/bash
#SBATCH --job-name=cgind_eval
#SBATCH --array=0-3
#SBATCH --time=03:00:00
#SBATCH --mem=180G
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a40:1
#SBATCH --partition=inter
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/cgind_eval_%A_%a.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/cgind_eval_%A_%a.err

set -euo pipefail
ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
PY=/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python
SEEDS=(501 502 503 505)
TASK=${SLURM_ARRAY_TASK_ID:?array task required}
SEED=${SEEDS[$TASK]}
CACHE=/project/ls-gruen/users/zekang.zhang/sbsi_caches/ablation
CK=$CACHE/measurement_flow_g0_ngmix_ablate_s2c_lt500_v22_s${SEED}_swaavg.pt
CAT=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant_indomain/constant_response_catalogue_train.feather
BLEND=$ROOT/results/blend_lookup_lsst_r_const_indomain_c40-89.feather
CROWD=$ROOT/results/crowd_flux_const_indomain_c40-89.feather
OUTDIR=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v22_constgold_indomain_c40-89
OUT=$OUTDIR/ablate_s2c_lt500_v22_indomain_perobj_s${SEED}.feather
mkdir -p "$OUTDIR"
for f in "$CK" "$CAT" "$BLEND" "$CROWD"; do [ -f "$f" ] || { echo "MISSING $f"; exit 1; }; done
[ ! -e "$OUT" ] || { echo "REFUSING overwrite $OUT"; exit 1; }
export LD_LIBRARY_PATH="/project/ls-gruen/users/zekang.zhang/envs/sims1/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT:/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$ROOT"
nvidia-smi -L
"$PY" -u scripts/validate_constant_with_blend.py \
  --measurement-model "$CK" --catalogue "$CAT" --min-case 40 --max-case 90 \
  --primary-mag-max 25.8 --primary-re-min 0.5 \
  --blend-lookup "$BLEND" --crowd-flux-lookup "$CROWD" \
  --global-only --flow-seed 12345 --n-samples 64 --max-rows 0 --batch-size 16384 \
  --dump "$OUT"
echo "CONSTGOLD_INDOMAIN_EVAL_DONE seed=$SEED"
