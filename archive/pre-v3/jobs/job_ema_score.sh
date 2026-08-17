#!/bin/bash
#SBATCH --job-name=emascore
#SBATCH --time=01:30:00
#SBATCH --mem=200G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=inter
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/emascore_%j.out

# Close out the mini-batch label-noise hypothesis (WORKLOG 2026-07-28g) with the ONE test that has no
# confound: same batch, same step count, same LR, same seed, same target -- only the per-bin estimate
# is accumulated across batches (--response-bin-ema) instead of re-measured per batch.
#
# Scored on the WIDE ruler (cases 0-99, 5.9M galaxies) because 28h showed the cases-0-39 ruler cannot
# resolve differences in the [0.30,0.38) bin (+-1.4% truth error). Control is the SAME seed 501.
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
D=/project/ls-gruen/users/zekang.zhang/sbsi_caches/ablation
echo "### EMA SCORING (wide ruler) job=$SLURM_JOB_ID ###"; nvidia-smi -L; date
for T in lt500_dom6x9lows lows_ema090 lows_ema098; do
  echo; echo "=================== $T (seed 501) ==================="
  python -u scripts/eval_selfresp_gap.py \
    --ckpt "$D/measurement_flow_g0_ngmix_ablate_s2c_${T}_s501_swaavg.pt" \
    --true-re-min 0.3 --true-mag-max 26.0 --max-case 99 2>&1 | grep -v "^  measurement_flow"
done
echo "EMASCORE_DONE"; date
