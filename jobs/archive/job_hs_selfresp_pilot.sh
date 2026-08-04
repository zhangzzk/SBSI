#!/bin/bash
#SBATCH --job-name=hs_pilot
#SBATCH --time=03:00:00
#SBATCH --mem=34G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a40-16gb:1
#SBATCH --partition=cip
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/hs_pilot_%j.out
set -o pipefail

# Half-shear SELF-response dump for the 3-SEED EDGES-ONLY PILOT (tag ablate_s2c_lt500_domB6).
# Scores the prediction recorded in WORKLOG 2026-08-03m against the fiducial's -3.42 +- 1.10% on
# Re <= 0.386". Identical to jobs/job_halfshear_selfresp.sh except the checkpoint TAG and --out.
#
# --out IS MANDATORY AND DIFFERENT: the parent job defaults to results/halfshear_selfresp.feather,
# which is the FIDUCIAL fig-5 data. Overwriting it would destroy the baseline this pilot is scored
# against, so the pilot writes its own file and the fiducial is left untouched.
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
if [ "${NO_EXPANDABLE_SEGMENTS:-1}" = "1" ]; then unset PYTORCH_CUDA_ALLOC_CONF
else export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; fi
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
D=/project/ls-gruen/users/zekang.zhang/sbsi_caches/ablation
TAG=${TAG:-ablate_s2c_lt500_domB6}
SEEDS="${SEEDS:-501 502 503}"
OUT=${OUT:-/project/ls-gruen/users/zekang.zhang/sbsi_caches/pin_realloc/halfshear_selfresp_domB6.feather}

[ "$OUT" = "results/halfshear_selfresp.feather" ] && { echo "REFUSING: would overwrite the fiducial fig-5 data"; exit 1; }
CK=""
for sd in $SEEDS; do
  # SWA=0 scores the RAW final-epoch weights instead of the SWA average. The response is a
  # DERIVATIVE of the mean head, and averaging weights across epochs can blunt derivative structure
  # that each individual epoch has -- a level-preserving average is not a slope-preserving one.
  # Default 1 keeps every previous dump byte-identical.
  if [ "${SWA:-1}" = "1" ]; then P=$D/measurement_flow_g0_ngmix_${TAG}_s${sd}_swaavg.pt
  else P=$D/measurement_flow_g0_ngmix_${TAG}_s${sd}.pt; fi
  [ -f "$P" ] || { echo "MISSING checkpoint $P"; exit 1; }
  CK="$CK $P"
done
echo "### HS SELFRESP PILOT tag=$TAG seeds='$SEEDS' job=$SLURM_JOB_ID ###"; nvidia-smi -L; date
echo "out=$OUT"
python -u scripts/dump_halfshear_selfresp.py --ckpt $CK --n-samples 32 --batch-size 16384 \
  ${EXTRACT:+--extraction $EXTRACT} ${EXTDELTA:+--extraction-delta $EXTDELTA} \
  --out "$OUT" 2>&1 | grep -v --line-buffered "module command" || { echo HS_PILOT_FAILED; exit 1; }
echo HS_PILOT_DONE; date
