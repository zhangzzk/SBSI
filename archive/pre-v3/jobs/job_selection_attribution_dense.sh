#!/bin/bash
#SBATCH --job-name=sel_dense
#SBATCH --time=00:45:00
#SBATCH --mem=150G
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --partition=inter
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/sel_dense_%j.out

# Attribution on the DENSE cut grid, so the remade figure samples the cut axis as finely as the
# 2026-07-25 original it replaces.
#
# WHY. The first remake used only the 7 cuts the attribution job happened to carry (4 size + 3 mag).
# The original figure swept size 0.2-0.8" and mag 26.5-24.5, which is what makes the TURNOVER
# visible -- the size effect is flat below ~0.5" and climbs steeply after. Seven points cannot show
# a turnover; they can only show that one exists.
#
# GRID matches the original. Size thresholds are stored in PIXELS (0.2"/px), so 0.2-0.9" is 1.0-4.5:
#   size px : 1.0 1.5 2.0 2.5 3.0 3.5 4.0 4.5   ->  0.20 0.30 0.40 0.50 0.60 0.70 0.80 0.90 arcsec
#   mag     : 24.5 25.0 25.5 26.0 26.5
# The PSF half-light radius is 0.527", so the flat part below ~0.5" is the PSF floor: a measured-size
# cut there selects essentially nothing shear-dependent. That is a prediction the dense grid tests.
#
# Saves sim_err (analytic SE) and mod_sem (4-seed spread) so the figure gets error bars -- without
# them "is m_flow consistent with zero?" cannot be answered from the saved product.
#
# FIREWALL: half-shear legs, ISOLATED (R_blend ~ 0). No emulator, no constgold. Trains nothing.
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot

D=/project/ls-gruen/users/zekang.zhang/sbsi_caches/ablation
SEEDS="${SEEDS:-501 502 503 505}"   # s504 absent from the dom6x6 set
CK=""; for sd in $SEEDS; do CK="$CK $D/measurement_flow_g0_ngmix_ablate_s2c_lt500_dom6x6_s${sd}_swaavg.pt"; done
echo "### SELECTION ATTRIBUTION -- DENSE GRID (4 seeds, n=32, ISO)  job=$SLURM_JOB_ID ###"
nvidia-smi -L; date

python -u scripts/eval_selection_attribution.py --ckpt $CK \
  --max-case 39 --n-samples 32 --batch-size 16384 \
  --size-cuts 1.0 1.5 2.0 2.5 3.0 3.5 4.0 4.5 \
  --mag-cuts 24.5 25.0 25.5 26.0 26.5 \
  --output "$D/selection_attribution_dense_4seed_n32.npz" || { echo SEL_DENSE_FAILED; exit 1; }
echo SEL_DENSE_DONE; date
