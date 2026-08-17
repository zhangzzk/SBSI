#!/bin/bash
#SBATCH --job-name=hssmoke
#SBATCH --time=00:40:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=inter
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/hssmoke_%j.out

# End-to-end smoke of the fresh-case dump path on ONE case (40) and TWO checkpoints:
# base-cache build -> cache-backed scoring -> merge. Validates the --min-case window, the cache
# round-trip, the per-seed part layout and every merge assertion before the 160-case run is spent.
# Writes to a throwaway directory; touches nothing the real run uses.
set -o pipefail
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot

D=/project/ls-gruen/users/zekang.zhang/sbsi_caches/ablation
S=/project/ls-gruen/users/zekang.zhang/sbsi_caches/hs_selfresp_smoke
rm -rf $S; mkdir -p $S/parts
echo "### HS SELFRESP SMOKE job=$SLURM_JOB_ID ###"; nvidia-smi -L; date

python -u scripts/dump_halfshear_selfresp.py --min-case 40 --max-case 40 \
  --base-cache $S/base.feather --build-base-only --blind || { echo SMOKE_FAILED_BASE; exit 1; }

for SD in 501 502; do
  python -u scripts/dump_halfshear_selfresp.py \
    --ckpt $D/measurement_flow_g0_ngmix_ablate_s2c_lt500_dom6x6_s${SD}_swaavg.pt \
    --min-case 40 --max-case 40 --base-cache $S/base.feather \
    --n-samples 32 --batch-size 16384 --blind --out $S/parts/part_s${SD}.feather \
    || { echo SMOKE_FAILED_SCORE_$SD; exit 1; }
done

python -u scripts/merge_halfshear_selfresp_parts.py \
  --parts "$S/parts/part_s*.feather" --out $S/merged.feather \
  --min-case 40 --max-case 40 --expect-seeds 501 502 || { echo SMOKE_FAILED_MERGE; exit 1; }

# negative control: the merge MUST refuse a dump that reaches into the burned cases 0-39
python -u scripts/merge_halfshear_selfresp_parts.py \
  --parts "$S/parts/part_s*.feather" --out $S/merged_bad.feather \
  --min-case 0 --max-case 39 --expect-seeds 501 502 \
  && { echo "SMOKE_FAILED: merge accepted an out-of-window case range"; exit 1; }
echo "negative control OK: merge refused the wrong case window"
ls -la $S $S/parts
echo SMOKE_DONE; date
