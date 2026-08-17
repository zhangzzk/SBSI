#!/bin/bash
#SBATCH --job-name=selfresp_v2
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=inter
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/selfresp_v2_%j.out

# Apples-to-apples self-response gap: V2 ensemble vs the V1-ladder S3a control on ONE ruler.
# Builds the base + truth R_hs + ISO mask ONCE (scripts.eval_selfresp_gap.load_ruler), then
# scores the V2 SetConditionedForwardModel ensemble (native central-secant readout) and the
# S3a ConditionalMeanFlow control (forward AND central) against the identical objects/truth.
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI-ablation:/home/z/Zekang.Zhang/SBSI:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/z/Zekang.Zhang/SBSI-ablation

V2GLOB='/project/ls-gruen/users/zekang.zhang/sbsi_caches/forward_proto/forward_ens_lr250_swa8_seed*_joint.pt'
V1GLOB='/project/ls-gruen/users/zekang.zhang/sbsi_caches/ablation/measurement_flow_g0_ngmix_ablate_s3a_grid6x9_s*_swaavg.pt'
OUTDIR=/project/ls-gruen/users/zekang.zhang/sbsi_caches/ablation/eval
mkdir -p "$OUTDIR"

echo "### SELFRESP_GAP_V2 job=$SLURM_JOB_ID ###"; nvidia-smi -L; date
python -u scripts/eval_selfresp_gap_v2.py \
  --v2-glob "$V2GLOB" \
  --v1-glob "$V1GLOB" \
  --v1-difference forward --v1-both-stencils \
  --max-case 39 \
  --output "$OUTDIR/selfresp_gap_v2_vs_s3a.npz" || { echo "SELFRESP_GAP_V2 FAILED"; exit 1; }
echo "SELFRESP_GAP_V2_JOB_DONE job=$SLURM_JOB_ID"; date
