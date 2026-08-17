#!/bin/bash
#SBATCH --job-name=isolad8
#SBATCH --time=00:45:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=inter
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/isolad8_%j.out

# Regenerate the dom6x6 self-response dump with ALL 8 seeds. The existing dump was written when only
# 3 seeds had finished, so the isolation-strictness ladder built from it is internally valid (every
# rung shares the same 3 seeds) but its absolute numbers are not the 8-seed ensemble's.
#
# FIREWALL-CLEAN: scores existing checkpoints against the det_meas half-shear truth only. No training,
# no constgold. The npz carries per-object nbr_flux/iso/R_flow_seeds, which is what the ladder needs.
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
D=/project/ls-gruen/users/zekang.zhang/sbsi_caches/ablation
TAG=${TAG:-ablate_s2c_lt500_dom6x6}
OUT=${OUT:-$D/eval/selfresp_${TAG}_8seed.npz}
N=$(ls $D/measurement_flow_g0_ngmix_${TAG}_s*_swaavg.pt 2>/dev/null | wc -l)
echo "### ISO LADDER 8-SEED tag=$TAG ($N ckpts) job=$SLURM_JOB_ID ###"; nvidia-smi -L; date
[ "$N" -eq 0 ] && { echo "no checkpoints for $TAG"; exit 1; }
python -u scripts/eval_selfresp_gap.py \
  --ckpt-glob "$D/measurement_flow_g0_ngmix_${TAG}_s*_swaavg.pt" \
  --true-re-min 0.3 --true-mag-max 26.0 \
  --output "$OUT"
echo "ISOLAD8_DONE tag=$TAG"; date
