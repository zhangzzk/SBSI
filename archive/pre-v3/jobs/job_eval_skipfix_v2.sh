#!/bin/bash
#SBATCH --job-name=eval_skipfix
#SBATCH --time=01:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint=x86-64-v3
#SBATCH --partition=inter
#SBATCH --output=/home/z/Zekang.Zhang/logs/eval_skipfix_%j.out

# Score the shape-skip V2 ensemble on the SAME ISO acceptance ruler. S3a control must re-print
# +4.70% (proves the ruler is the same one that gave V2-count -2.69% / V2-equal -5.24%).
# eval_selfresp_gap_v2.py -> worktree eval_constgold_closure (skip-aware load_model) via worktree-
# first PYTHONPATH, so the skip head is loaded and its response is read out.
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI-ablation:/home/z/Zekang.Zhang/SBSI:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/z/Zekang.Zhang/SBSI-ablation

V2GLOB='/project/ls-gruen/users/zekang.zhang/sbsi_caches/ablation/forward_skipfix_lr250_swa8_seed*_joint.pt'
V1GLOB='/project/ls-gruen/users/zekang.zhang/sbsi_caches/ablation/measurement_flow_g0_ngmix_ablate_s3a_grid6x9_s*_swaavg.pt'
OUTDIR=/project/ls-gruen/users/zekang.zhang/sbsi_caches/ablation/eval
mkdir -p "$OUTDIR"

echo "### EVAL_SKIPFIX job=$SLURM_JOB_ID ###"; nvidia-smi -L; date
python -u scripts/eval_selfresp_gap_v2.py \
  --v2-glob "$V2GLOB" \
  --v1-glob "$V1GLOB" \
  --v1-difference forward \
  --max-case 39 \
  --output "$OUTDIR/selfresp_gap_v2skipfix_vs_s3a.npz" || { echo "EVAL_SKIPFIX FAILED"; exit 1; }
echo "EVAL_SKIPFIX_JOB_DONE job=$SLURM_JOB_ID"; date
