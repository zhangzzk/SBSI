#!/bin/bash
#SBATCH --job-name=cg_nd
#SBATCH --time=01:30:00
#SBATCH --mem=200G
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --partition=inter
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/cg_nd_%j.out

# constgold selection table at NEAR-DOMAIN cuts using the REAL per-leg measured mag/size added by
# job 15366166. Sim and model cut on the SAME quantity at the SAME absolute threshold -- no proxy
# except the (*) real-S/N rows. Reports ALL and ISOLATED; m_flow is only a clean model bias on the
# ISOLATED block (the ALL block's column (3) carries the neighbour term the flow does not model).
set -o pipefail
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
D=/project/ls-gruen/users/zekang.zhang/sbsi_caches/ablation
SEEDS="${SEEDS:-501 502 503 505}"
CK=""; for sd in $SEEDS; do CK="$CK $D/measurement_flow_g0_ngmix_ablate_s2c_lt500_dom6x6_s${sd}_swaavg.pt"; done
echo "### CONSTGOLD NEAR-DOMAIN job=$SLURM_JOB_ID ###"; nvidia-smi -L; date
python -u scripts/eval_selection_constgold_neardomain.py --ckpt $CK --n-samples 32 --batch-size 16384 \
  2>&1 | grep -v "module command" || { echo CG_ND_FAILED; exit 1; }
echo CG_ND_ALL_DONE; date
