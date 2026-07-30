#!/bin/bash
#SBATCH --job-name=cgmodel
#SBATCH --time=02:00:00
#SBATCH --mem=250G
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --partition=inter
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/cgmodel_%j.out

# constgold selection table + MODEL m column, cutting on the M2 proxy S/N from the flow's outputs.
# 250G because constgold is 29.8M rows and the draws are (N, n_samples) per leg -- chunked, but the
# base frame itself is large. FIREWALL: constgold is EVALUATION only; nothing trains or is selected.
set -o pipefail
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
D=/project/ls-gruen/users/zekang.zhang/sbsi_caches/ablation
SEEDS="${SEEDS:-501 502 503 505}"
CK=""; for sd in $SEEDS; do CK="$CK $D/measurement_flow_g0_ngmix_ablate_s2c_lt500_dom6x6_s${sd}_swaavg.pt"; done
echo "### CONSTGOLD MODEL SELECTION job=$SLURM_JOB_ID ###"; nvidia-smi -L; date
python -u scripts/eval_selection_constgold_model.py --ckpt $CK \
  --n-samples "${NS:-32}" --batch-size 16384 --max-rows "${MAXROWS:-4000000}" \
  2>&1 | grep -v "module command" || { echo CGMODEL_FAILED; exit 1; }
echo CGMODEL_ALL_DONE; date
