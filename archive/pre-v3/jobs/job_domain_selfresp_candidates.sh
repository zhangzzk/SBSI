#!/bin/bash
#SBATCH --job-name=domcand
#SBATCH --time=00:45:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=inter
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/domcand_%j.out

# NON-CIRCULAR candidate selection. Scores each grid candidate against the det_meas half-shear
# truth on the deliverable domain -- constgold is NOT read, so choosing a winner here cannot
# contaminate the final constgold readout. (Prior "sub-percent" results in this project were
# retracted for exactly that kind of circularity; see WORKLOG cont.66-67 and the R_blend firewall.)
#
# The criterion is BOTH the OVERALL gap AND the spread across true-size bins: the 3-bin dom2 grid
# already showed that a small OVERALL number can hide large cancelling per-bin errors (baseline
# +0.54% overall = -25.24% against +11.54%).
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
D=/project/ls-gruen/users/zekang.zhang/sbsi_caches/ablation
E=$D/eval; mkdir -p $E
echo "### DOMAIN CANDIDATES job=$SLURM_JOB_ID ###"; nvidia-smi -L; date

for TAG in ablate_s2c_lt500_dom6x6 ablate_s2c_lt500_dom8x8; do
  N=$(ls $D/measurement_flow_g0_ngmix_${TAG}_s*_swaavg.pt 2>/dev/null | wc -l)
  echo; echo "=================== $TAG  ($N seeds) ==================="
  [ "$N" -eq 0 ] && { echo "no checkpoints -- skipped"; continue; }
  python -u scripts/eval_selfresp_gap.py \
    --ckpt-glob "$D/measurement_flow_g0_ngmix_${TAG}_s*_swaavg.pt" \
    --true-re-min 0.3 --true-mag-max 26.0 \
    --output "$E/selfresp_${TAG}.npz"
done
echo "DOMCAND_DONE"; date
