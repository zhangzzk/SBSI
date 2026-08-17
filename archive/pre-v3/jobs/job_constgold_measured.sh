#!/bin/bash
#SBATCH --job-name=cg_meas
#SBATCH --time=01:00:00
#SBATCH --mem=120G
#SBATCH --cpus-per-task=16
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/cg_meas_%j.out

# Rebuild the MISSING per-leg measured columns (MAG_AUTO, FLUX_RADIUS) for constgold +/-0.02.
# The per-leg catalogues stored only measured_e1/e2 + S/N, so no measured (shear-responding) cut
# could be applied -> no selection test was possible on constgold. ~1s/case/leg, 140 cases x 2 legs.
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
OUT=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/constgold_measured_c0-139.feather
echo "### CONSTGOLD MEASURED job=$SLURM_JOB_ID ###"; date
python -u scripts/build_constgold_measured.py \
  --min-case 0 --max-case 139 --shears 0.02 -0.02 \
  --n-jobs 16 --output $OUT
echo "CG_MEAS_DONE"; date
