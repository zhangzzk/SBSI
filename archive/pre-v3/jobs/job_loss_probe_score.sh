#!/bin/bash
#SBATCH --job-name=lossprobe
#SBATCH --time=00:45:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=inter
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/lossprobe_%j.out

# Score the four response-loss probes against the half-shear ruler, PAIRED: every model below is
# seed 501 trained on the same refined target, so the ONLY difference is the response loss. The
# control is the unmodified absolute lam=450 run at that same seed -- comparing 1-seed probes to
# the 8-seed ensemble would confound the loss change with ~0.8% seed scatter in the target bin.
#
# FIREWALL-CLEAN: half-shear truth only, no constgold. The 2026-07-28e acceptance test stands:
# [0.30,0.38) must shrink substantially AND no other size bin may degrade beyond its noise floor.
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
D=/project/ls-gruen/users/zekang.zhang/sbsi_caches/ablation
E=$D/eval; mkdir -p $E
echo "### LOSS PROBE SCORING job=$SLURM_JOB_ID ###"; nvidia-smi -L; date

for T in ${TAGS:-lt500_dom6x9lows lows_rw1500 lows_rw4500 lows_rel030 lows_rel050}; do
  CK=$D/measurement_flow_g0_ngmix_ablate_s2c_${T}_s501_swaavg.pt
  echo; echo "=================== $T (seed 501) ==================="
  [ -f "$CK" ] || { echo "MISSING $CK"; continue; }
  python -u scripts/eval_selfresp_gap.py --ckpt "$CK" \
    --true-re-min 0.3 --true-mag-max 26.0 \
    --output "$E/lossprobe_${T}_s501.npz" 2>&1 | grep -v "^  measurement_flow"
done
echo "LOSSPROBE_DONE"; date
