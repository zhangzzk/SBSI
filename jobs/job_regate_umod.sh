#!/bin/bash
#SBATCH --job-name=regate
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/regate_%j.out

# CORRECTED P0 RE-GATE (results/regate_prereg_h.json, id RA-P0-REGATE-2026-08-01h).
#
# 2026-08-01h is the parent RA-P0-REGATE-2026-08-01e (results/regate_prereg.json) with exactly two
# owner-authorised corrections: the measured-size thresholds restated in the column's native PIXEL
# units (measured_flux_radius_0 is SExtractor FLUX_RADIUS in px at 0.2 arcsec/px, so 0.60" -> 3.0 px,
# 0.70" -> 3.5 px and the units band [0.4, 2.0]" -> [2.0, 10.0] px), and a re-derived C2e permutation
# null that compares F_bar to the permutation DISTRIBUTION (999 permutations, seeds 2001-2999,
# p <= 0.01) instead of to an absolute constant.  Everything else is carried over verbatim.
#
# The parent stays executable for the audit -- it BLOCKS on its own units check, which is correct:
#   python -u scripts/eval_regate_umod.py --prereg results/regate_prereg.json \
#     --base-cache <base> --selfresp <dump> --min-case 40 --max-case 199 \
#     --out-json results/regate_result_parent_repro.json --out-npz /tmp/regate_parent_repro.npz
#
# Executes the pre-registration verbatim on the FRESH half-shear cases 40-199.  CPU only, as
# pre-registered (64G / 8 CPU, no --gres).  Trains nothing, opens no constgold file, does not
# touch results/halfshear_selfresp.feather, results/constgold_neardomain_*.npz, results/nd_seeds*/
# or any checkpoint under sbsi_caches/ablation/.  results/regate_result.json -- the parent's BLOCKED
# record -- is NOT overwritten: this job writes results/regate_result_h.json, and the script refuses
# to clobber an existing --out-json unless --overwrite is passed.
set -o pipefail
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot

echo "### RE-GATE job=$SLURM_JOB_ID ###"; date
python -u scripts/eval_regate_umod.py \
  --base-cache /project/ls-gruen/users/zekang.zhang/sbsi_caches/hs_selfresp_c40-199/base_c40-199.feather \
  --selfresp   /project/ls-gruen/users/zekang.zhang/sbsi_caches/hs_selfresp_c40-199/halfshear_selfresp_c40-199.feather \
  --min-case 40 --max-case 199 \
  --prereg    results/regate_prereg_h.json \
  --out-json  results/regate_result_h.json \
  --out-npz   /project/ls-gruen/users/zekang.zhang/sbsi_caches/ra/regate_umod_h_c40-199.npz \
  2>&1 | grep -v --line-buffered "module command" || { echo REGATE_FAILED; exit 1; }
echo REGATE_DONE; date
