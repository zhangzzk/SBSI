#!/bin/bash
#SBATCH --job-name=anti_target
#SBATCH --time=03:00:00
#SBATCH --mem=140G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cluster
#SBATCH --output=/home/z/Zekang.Zhang/logs/anti_target_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/anti_target_%j.err

# Pre-registered comparison on exact matched rows:
#   forward  = (e(+g)-e(0))/g
#   backward = (e(0)-e(-g))/g
#   central  = (e(+g)-e(-g))/(2g)
# The central target reuses the existing V2.1 5x4x5 grid edges. Constgold is
# never read, and no response is scaled or tuned to its m.
set -euo pipefail
SIMS=/project/ls-gruen/users/zekang.zhang/envs/sims1
export CONDA_PREFIX="$SIMS"
export PATH="$SIMS/bin:$PATH"
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}"
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
D=/project/ls-gruen/users/zekang.zhang/sbsi_catalogues
R=/home/z/Zekang.Zhang/SBSI/results
OUT=$R/response_target_crowd_rblend_antithetic_self_c0-99_5x4x5_v21.npz
if [ -e "$OUT" ]; then echo "REFUSING to overwrite $OUT"; exit 1; fi

echo "ANTITHETIC SELF TARGET job=$SLURM_JOB_ID"; date
python -u scripts/compute_response_target_antithetic_self.py \
  --plus-catalogue "$D/det_meas_crowd_g0.02_test_full.feather" \
  --minus-catalogue "$D/det_meas_ngmix_np7_gm002_antithetic_c0-99.feather" \
  --snc-lookup "$R/g0_lookup_c0-99.feather" \
  --reference-target "$R/response_target_crowd_rblend_snc_c0-99_5x4x5_v21.npz" \
  --output "$OUT"
echo ANTITHETIC_SELF_TARGET_JOB_DONE; date
