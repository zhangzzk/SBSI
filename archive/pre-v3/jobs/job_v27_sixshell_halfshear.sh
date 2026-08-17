#!/bin/bash
#SBATCH --job-name=hsv27sh
#SBATCH --array=0-1
#SBATCH --time=03:00:00
#SBATCH --mem=40G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a40-16gb:1
#SBATCH --partition=cip
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/hsv27sh_%A_%a.out
set -euo pipefail
ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
PY=/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python
export LD_LIBRARY_PATH="/project/ls-gruen/users/zekang.zhang/envs/sims1/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT:/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}"
unset PYTORCH_CUDA_ALLOC_CONF
cd "$ROOT"
TASK=${SLURM_ARRAY_TASK_ID:?array task required}
IFS=',' read -r -a SEEDS <<< "${V27_SEEDS:-501,502}"
(( TASK < ${#SEEDS[@]} )) || { echo "NO SEED for task $TASK"; exit 1; }
SEED=${SEEDS[$TASK]}
C=/project/ls-gruen/users/zekang.zhang/sbsi_caches/v27_scene
CK=/project/ls-gruen/users/zekang.zhang/sbsi_caches/ablation/measurement_flow_g0_ngmix_ablate_s2c_lt500_v27_sixshell_s${SEED}_swaavg.pt
OUTDIR=$C/halfshear_scores; OUT=$OUTDIR/sixshell_s${SEED}.feather
mkdir -p "$OUTDIR"
for f in "$CK" "$C/halfshear_sixshell_v27_base_c40-199_v22domain.feather"; do [ -f "$f" ] || { echo "MISSING $f"; exit 1; }; done
[ ! -e "$OUT" ] || { echo "REFUSING overwrite $OUT"; exit 1; }
"$PY" -u scripts/dump_halfshear_selfresp.py \
  --ckpt "$CK" --min-case 40 --max-case 199 \
  --base-cache "$C/halfshear_sixshell_v27_base_c40-199_v22domain.feather" \
  --n-samples 32 --batch-size 16384 --blind --out "$OUT"
echo V27_SIXSHELL_HALFSHEAR_DONE seed=$SEED; date
