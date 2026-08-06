#!/bin/bash
#SBATCH --job-name=cg_v22
#SBATCH --time=03:00:00
#SBATCH --mem=110G
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a40:1
#SBATCH --partition=cip
#SBATCH --output=/home/z/Zekang.Zhang/logs/cg_v22_s%a_%j.out
set -euo pipefail

# Evaluation only: constgold is never read by target construction, emulator training or flow
# training. Each task writes one seed dump; the dependent CPU aggregation forms the 2-seed result.
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot

SEEDS=(${SEEDS:-501 502})
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}; else SEED=${SEED:-501}; fi
TAG=${TAG:-ablate_s2c_lt500_v22}
CACHE=/project/ls-gruen/users/zekang.zhang/sbsi_caches/ablation
CAT=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant/constant_response_catalogue_train.feather
DUMPDIR=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v22_domain_dumps
CK="$CACHE/measurement_flow_g0_ngmix_${TAG}_s${SEED}_swaavg.pt"
LOOKUP=results/blend_lookup_v22_c40-139.feather
CROWD=/home/z/Zekang.Zhang/SBSI/results/crowd_flux_conc_c0-199.feather
mkdir -p "$DUMPDIR"
for f in "$CK" "$LOOKUP" "$CROWD"; do [ -f "$f" ] || { echo "MISSING $f"; exit 1; }; done
OUT="$DUMPDIR/${TAG}_perobj_s${SEED}.feather"
[ ! -e "$OUT" ] || { echo "REFUSING to overwrite $OUT"; exit 1; }

echo "### V2.2 CONSTGOLD seed=$SEED job=$SLURM_JOB_ID ###"; nvidia-smi -L; date
python -u scripts/validate_constant_with_blend.py \
  --measurement-model "$CK" --catalogue "$CAT" --min-case 40 \
  --blend-lookup "$LOOKUP" --crowd-flux-lookup "$CROWD" \
  --global-only --flow-seed 12345 --n-samples 64 --max-rows 45000000 --batch-size 16384 \
  --dump "$OUT"
echo "CG_V22_DONE seed=$SEED"; date
