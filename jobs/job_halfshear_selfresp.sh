#!/bin/bash
#SBATCH --job-name=hs_self
#SBATCH --time=03:00:00
#SBATCH --mem=180G
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --partition=inter
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/hs_self_%j.out

# Per-object SELF-response on the half-shear legs (sim vs flow), for figure 5.
# Self-response is isolated by projecting on the PRIMARY's shear direction; neighbours carry
# independent random directions and average away. FORWARD extraction on both sides (0 -> +g),
# matching the sim -- antithetic would reintroduce the known 0.49-vs-0.60 extraction gap.
set -o pipefail
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
D=/project/ls-gruen/users/zekang.zhang/sbsi_caches/ablation
SEEDS="${SEEDS:-501 502 503 505 506 507 508 509 510 511 512 513 514 515 516 517}"
CK=""; for sd in $SEEDS; do CK="$CK $D/measurement_flow_g0_ngmix_ablate_s2c_lt500_dom6x6_s${sd}_swaavg.pt"; done
echo "### HALF-SHEAR SELF-RESPONSE job=$SLURM_JOB_ID ###"; nvidia-smi -L; date
python -u scripts/dump_halfshear_selfresp.py --ckpt $CK --n-samples 32 --batch-size 16384 \
  2>&1 | grep -v "module command" || { echo HS_SELF_FAILED; exit 1; }
echo HS_SELF_ALL_DONE; date
