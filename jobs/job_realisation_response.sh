#!/bin/bash
#SBATCH --job-name=realresp
#SBATCH --time=03:00:00
#SBATCH --mem=180G
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a40:1
#SBATCH --partition=inter
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/realresp_%j.out

# NON-CIRCULAR acceptance harness for realisation-dependent shear response.
# scripts/eval_realisation_response.py -- model-vs-sim response in bins of MEASURED size and MEASURED
# magnitude, on the HALF-SHEAR legs.  NO constgold file is opened; nothing is trained, fit or tuned.
#
# Scopes: ISOLATED (no brighter true nbr within 7") and, with --all-too, every matched both-detected
# in-domain row.  NO R_blend is added on either: each galaxy carries its own random shear direction
# and the estimator projects on the PRIMARY's ghat_p, so the neighbours' term averages away (measured:
# ALL-scope no-cut R_sim +0.7202 vs R_model +0.7181 = -0.30%, job 15422516, no blend term anywhere).
# So no emulator enters, and the run is firewall-clean end to end on both scopes.
#
# SEEDS: 4 by default (501 502 503 505).  That is enough for the model/sim RATIO structure and for
# rho_sim/rho_mod (both cancel the seed level offset).  It is NOT enough for an absolute per-bin
# m -- per AGENTS.md any m needs the full 16.  Override with SEEDS="501 502 ... 517".
#
# TF32 is on: re-gated on an A40 in WORKLOG 2026-07-30 (ADOPTED, 1.29x).  It is a silent no-op on
# a V100, hence the explicit --gres=gpu:a40:1.
set -o pipefail
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot

D=/project/ls-gruen/users/zekang.zhang/sbsi_caches/ablation
SEEDS="${SEEDS:-501 502 503 505}"
CK=""; for sd in $SEEDS; do CK="$CK $D/measurement_flow_g0_ngmix_ablate_s2c_lt500_dom6x6_s${sd}_swaavg.pt"; done
N=$(ls $CK 2>/dev/null | wc -l)
TAG="${TAG:-dom6x6_${N}seed}"
NS="${NS:-32}"
EXTRA="${EXTRA:---all-too}"

echo "### REALISATION-RESPONSE HARNESS  job=$SLURM_JOB_ID  seeds=$N  n_samples=$NS ###"
nvidia-smi -L; date

python -u scripts/eval_realisation_response.py \
  --ckpt $CK --tf32 \
  --max-case 39 --n-samples "$NS" --batch-size 16384 --chunk 200000 \
  $EXTRA \
  --output "results/realisation_response_${TAG}.npz" \
  || { echo "REALRESP_FAILED"; exit 1; }
echo "REALRESP_DONE"; date
