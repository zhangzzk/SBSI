#!/bin/bash
#SBATCH --job-name=rblendgap
#SBATCH --time=01:00:00
#SBATCH --mem=200G
#SBATCH --cpus-per-task=8
#SBATCH --partition=inter
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/rblendgap_%j.out

# Score the BlendEMU per-pair blending response against half-shear truth ON THE DELIVERABLE DOMAIN.
# R_flow is cleared (-0.05%, WORKLOG 2026-07-28h), so the in-domain m of -0.508% must live in R_blend
# or the population transfer. The emulator's training cuts are a SUPERSET of our domain (primary mag
# 18-28 vs <26, Re 0.1-1.5 vs >0.3) and its accuracy INSIDE the domain has never been measured.
#
# Truth = the NEIGHBOUR-ONLY-SHEARED leg of the half-shear 2x2 design. CPU-only (XGBoost on cpu).
# FIREWALL: no constgold is read.
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
D=/project/ls-gruen/users/zekang.zhang/sbsi_caches/ablation/eval; mkdir -p $D
echo "### R_BLEND GAP job=$SLURM_JOB_ID ###"; date
python -u scripts/eval_rblend_gap.py \
  --true-re-min 0.3 --true-mag-max 26.0 \
  --tag ${TAG:-lsst_r_extnbr_ho} \
  --output "$D/rblend_gap_${TAG:-lsst_r_extnbr_ho}.npz"
echo "RBLENDGAP_DONE"; date
