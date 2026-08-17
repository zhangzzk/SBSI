#!/bin/bash
#SBATCH --job-name=domsplit
#SBATCH --time=00:45:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=inter
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/domsplit_%j.out

# Is the domain model's residual in-domain m = -0.819% an R_flow error or an R_blend error?
#
# FIREWALL-CLEAN: scores both checkpoint ensembles against the det_meas HALF-SHEAR truth on the
# ISOLATED acceptance set (no brighter true neighbour within 7"), where R_blend ~ 0 by construction.
# constgold is never read. So `flow/R_hs - 1` here is a pure R_flow verdict:
#   ~0  -> R_flow is right in-domain, and the -0.82% overshoot must live in R_blend
#   >0  -> R_flow over-predicts, and the flow itself is the thing to tune
# The acceptance cut (--true-re-min 0.3 --true-mag-max 26.0) is exactly the deliverable domain.
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
D=/project/ls-gruen/users/zekang.zhang/sbsi_caches/ablation
E=$D/eval; mkdir -p $E
echo "### DOMAIN SELF-RESPONSE SPLIT job=$SLURM_JOB_ID ###"; nvidia-smi -L; date

echo; echo "=================== BASELINE (full-range training, 8 seeds) ==================="
python -u scripts/eval_selfresp_gap.py \
  --ckpt-glob "$D/measurement_flow_g0_ngmix_ablate_s2c_coupling_lt500_s50*_swaavg.pt" \
  --true-re-min 0.3 --true-mag-max 26.0 \
  --output "$E/selfresp_domain_baseline.npz"

echo; echo "=================== DOMAIN-TRAINED (dom2, 8 seeds) ==================="
python -u scripts/eval_selfresp_gap.py \
  --ckpt-glob "$D/measurement_flow_g0_ngmix_ablate_s2c_coupling_lt500_dom2_s50*_swaavg.pt" \
  --true-re-min 0.3 --true-mag-max 26.0 \
  --output "$E/selfresp_domain_dom2.npz"

echo "DOMSPLIT_DONE"; date
