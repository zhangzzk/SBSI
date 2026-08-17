#!/bin/bash
#SBATCH --job-name=anti_cat
#SBATCH --time=01:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cluster
#SBATCH --output=/home/z/Zekang.Zhang/logs/anti_cat_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/anti_cat_%j.err

# Generate only the missing random-direction -0.02 input catalogues/configs.
# Deterministic case seeds reproduce the existing +0.02 galaxies, positions,
# shear axes and noise seeds; the target-shear sign is the sole change.
set -euo pipefail
SIMS=/project/ls-gruen/users/zekang.zhang/envs/sims1
export CONDA_PREFIX="$SIMS"
export PATH="$SIMS/bin:$PATH"
export PYTHONPATH="/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}"
cd /home/z/Zekang.Zhang/blendemu/scripts
CFG=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot/configs/fs2_lsst_r_antithetic_self_gm002.yaml

echo "ANTITHETIC SELF CATALOGUES job=$SLURM_JOB_ID"; date
python -u run_pipeline.py --config "$CFG" --steps 1
echo ANTITHETIC_SELF_CATALOG_DONE; date
