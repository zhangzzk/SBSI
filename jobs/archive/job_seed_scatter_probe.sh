#!/bin/bash
#SBATCH --job-name=seedprobe
#SBATCH --time=01:30:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint=x86-64-v3
#SBATCH --partition=inter
#SBATCH --output=/home/z/Zekang.Zhang/logs/seedprobe_%j.out

# Goal-1 (task #32): quantify the SEED contribution to realistic-cut m, to answer "would more
# seeds pull the ~0.4-1.7% resolution-keep residuals under 0.3%?". Harvest each _prod seed
# (421/422/423) SEPARATELY, eval the realistic family for each against the SAME size-aware
# ensemble R_blend (common-mode -> cancels in the seed-to-seed scatter), then diag_seed_scatter.py
# splits each residual into seed-scatter (shrinks 1/sqrt(N)) vs systematic bias (seed-independent).
# Inference-only; certified m and all certified artifacts untouched.
set -e
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
cd /home/z/Zekang.Zhang/SBSI
FP=/project/ls-gruen/users/zekang.zhang/sbsi_caches/forward_proto
D=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk
RBLEND=$D/rblend_scene_jointrflow_prod_szphi_c40-139.npz   # size-aware ensemble R_blend (common-mode)
echo "### SEEDPROBE job=$SLURM_JOB_ID node=$SLURMD_NODENAME ###"; nvidia-smi -L; date

for S in 421 422 423; do
  echo "===== harvest single seed $S ====="; date
  RFLOW=$D/rflow_joint_prod_seed${S}_c40-139.npz
  if [ ! -s "$RFLOW" ]; then
    stdbuf -oL -eL python -B -u scripts/harvest_joint_rflow.py \
      --checkpoint $FP/forward_proto_c0-99_prod_seed${S}_joint.pt \
      --min-case 40 --chunk-batches 4 --out $RFLOW \
      || { echo "HARVEST FAILED seed=$S"; exit 1; }
  else
    echo "  (exists, reuse) $RFLOW"
  fi
  echo "===== eval realistic, single seed $S (fixed size-aware R_blend) ====="; date
  stdbuf -oL -eL python -B -u scripts/eval_selection_robustness.py \
    --rflow-override $RFLOW --rblend-override $RBLEND \
    --realistic --target 0.003 --tag seedprobe_s${S} \
    || { echo "EVAL FAILED seed=$S"; exit 1; }
done

echo "===== SEED-SCATTER ANALYSIS (all selections) ====="
stdbuf -oL -eL python -B -u scripts/diag_seed_scatter.py \
  --seed-rows $D/selrobust_seedprobe_s421_rows.json \
              $D/selrobust_seedprobe_s422_rows.json \
              $D/selrobust_seedprobe_s423_rows.json \
  --ens-rows  $D/selrobust_jointprod_szphi_rows.json
echo
echo "===== SEED-SCATTER ANALYSIS (acceptance family only) ====="
stdbuf -oL -eL python -B -u scripts/diag_seed_scatter.py \
  --seed-rows $D/selrobust_seedprobe_s421_rows.json \
              $D/selrobust_seedprobe_s422_rows.json \
              $D/selrobust_seedprobe_s423_rows.json \
  --ens-rows  $D/selrobust_jointprod_szphi_rows.json --acceptance-only
echo "### SEEDPROBE_DONE job=$SLURM_JOB_ID ###"; date
