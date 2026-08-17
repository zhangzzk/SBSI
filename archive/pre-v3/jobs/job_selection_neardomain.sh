#!/bin/bash
#SBATCH --job-name=neardom
#SBATCH --time=00:50:00
#SBATCH --mem=150G
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --partition=inter
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/neardom_%j.out

# Selection bias for cuts AROUND THE TRAINING DOMAIN EDGE (mag<26, Re>0.3"), which is the regime a
# real analysis uses -- not the aggressive keep-0.30 S/N tail where the flow is known to be
# under-resolved. Intrinsic shapes on both sides, so m_flow carries no measurement-error term and no
# missing blend term. Detection separated: both-detected pairs only.
# pipefail: python is piped into grep, so WITHOUT this the `||` guard below tests grep's exit status
# and a python traceback still prints NEARDOMAIN_ALL_DONE (it did, on job 15365939).
set -o pipefail
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
D=/project/ls-gruen/users/zekang.zhang/sbsi_caches/ablation
SEEDS="${SEEDS:-501 502 503 505}"
CK=""; for sd in $SEEDS; do CK="$CK $D/measurement_flow_g0_ngmix_ablate_s2c_lt500_dom6x6_s${sd}_swaavg.pt"; done
echo "### NEAR-DOMAIN SELECTION job=$SLURM_JOB_ID ###"; nvidia-smi -L; date
python -u scripts/eval_selection_neardomain.py --ckpt $CK --n-samples 32 --batch-size 16384 \
  2>&1 | grep -v "module command" || { echo NEARDOMAIN_FAILED; exit 1; }
echo NEARDOMAIN_ALL_DONE; date
