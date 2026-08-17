#!/bin/bash
#SBATCH --job-name=anti_build
#SBATCH --time=04:00:00
#SBATCH --mem=100G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cluster
#SBATCH --output=/home/z/Zekang.Zhang/logs/anti_build_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/anti_build_%j.err

# Build a one-row-per-target -g catalogue using the same 7 arcsec nearest-pair
# definition as det_meas_ngmix_np7_g0.02_test.feather. Catalogue construction is
# delegated to blendemu.response; SBSI only supplies the existing I/O wrapper.
set -euo pipefail
SIMS=/project/ls-gruen/users/zekang.zhang/envs/sims1
export CONDA_PREFIX="$SIMS"
export PATH="$SIMS/bin:$PATH"
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}"
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
OUT=/project/ls-gruen/users/zekang.zhang/sbsi_catalogues/det_meas_ngmix_np7_gm002_antithetic_c0-99.feather
if [ -e "$OUT" ]; then echo "REFUSING to overwrite $OUT"; exit 1; fi

echo "ANTITHETIC SELF BUILD job=$SLURM_JOB_ID"; date
python -u scripts/build_detection_measurement_catalogue.py \
  --data-path /project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876 \
  --shear=-0.02 --cases 0-99 --include-shapes --flow-only \
  --r-max 7 --r-min 0 --k 2 --n-jobs 8 --batch-size 8 \
  --output "$OUT"
echo ANTITHETIC_SELF_BUILD_DONE; date
