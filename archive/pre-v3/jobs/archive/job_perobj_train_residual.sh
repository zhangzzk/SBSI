#!/bin/bash
#SBATCH --job-name=perobj_tr
#SBATCH --time=01:00:00
#SBATCH --mem=34G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a40-16gb:1
#SBATCH --partition=cip
#SBATCH --output=/home/z/Zekang.Zhang/logs/perobj_tr_%j.out
set -o pipefail
# Does the small-size response offset exist on the TRAINING rows, under the TRAINING readout,
# against the TRAINING target? Separates OPTIMISATION failure (offset already there) from
# GENERALISATION gap (offset only on the ruler). See WORKLOG 2026-08-03s.
# FIREWALL: no constgold, no ruler, no `m` -- model vs its own target only.
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
unset PYTORCH_CUDA_ALLOC_CONF
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
D=/project/ls-gruen/users/zekang.zhang/sbsi_caches/ablation
T=/project/ls-gruen/users/zekang.zhang/sbsi_caches/perobj/perobj_target_D6_c0-99_dom.joblib
TAG=${TAG:-perobj_D6_rw450}
CK=""
for sd in ${SEEDS:-501 502 503}; do
  P=$D/measurement_flow_g0_ngmix_${TAG}_s${sd}_swaavg.pt
  [ -f "$P" ] || { echo "MISSING $P"; exit 1; }
  CK="$CK $P"
done
echo "### PEROBJ TRAIN RESIDUAL tag=$TAG job=$SLURM_JOB_ID ###"; nvidia-smi -L; date
python -u scripts/diag_perobj_train_residual.py --ckpt $CK --target "$T" \
  --response-delta ${RDELTA:-0.02} --response-difference ${RDIFF:-central} \
  --rows ${ROWS:-train} --max-rows ${MAXROWS:-1000000} 2>&1 | grep -v --line-buffered "module command" || exit 1
echo; echo PEROBJ_TR_DONE; date
