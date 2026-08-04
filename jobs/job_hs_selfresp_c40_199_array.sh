#!/bin/bash
#SBATCH --job-name=hsa40199
#SBATCH --time=02:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --partition=inter
#SBATCH --constraint=x86-64-v3
#SBATCH --array=0-15
#SBATCH --output=/home/z/Zekang.Zhang/logs/hsa40199_%A_%a.out

# STAGE B of the FRESH half-shear self-response dump (cases 40-199): ONE CHECKPOINT PER TASK.
#
# Requires the stage-A base cache (jobs/job_hs_selfresp_c40_199_base.sh). Each task loads the
# already-built population -- so all 16 tasks score the IDENTICAL rows in the IDENTICAL order --
# and writes only its own seed column alongside the sim columns. Row identity is re-checked at
# merge time by an elementwise (case, input_index) comparison, so a drifted task cannot pass.
#
# Submit with a dependency so it cannot start before the cache exists:
#   sbatch --dependency=afterok:<BASE_JOBID> jobs/job_hs_selfresp_c40_199_array.sh
#
# FRESH-DATA BLIND: --blind suppresses every response print. The fresh cases must not have a
# response number in a log before the gate statistic is fixed in writing.
set -o pipefail
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
python -c "import torch,sys; sys.exit(0 if torch.__version__>='2' else 1)" \
  || { echo "WRONG PYTHON/TORCH -- conda activate sims1 did not take"; exit 1; }

D=/project/ls-gruen/users/zekang.zhang/sbsi_caches/ablation
OUTDIR=/project/ls-gruen/users/zekang.zhang/sbsi_caches/hs_selfresp_c40-199
CACHE=$OUTDIR/base_c40-199.feather
mkdir -p $OUTDIR/parts
[ -f "$CACHE" ] || { echo "MISSING BASE CACHE $CACHE -- run stage A first"; exit 1; }

# 16 e-response seeds; 504 does not exist.
SEEDS=(501 502 503 505 506 507 508 509 510 511 512 513 514 515 516 517)
SD=${SEEDS[$SLURM_ARRAY_TASK_ID]}
if [ -z "$SD" ]; then echo "no seed for array index $SLURM_ARRAY_TASK_ID"; exit 1; fi
CK=$D/measurement_flow_g0_ngmix_ablate_s2c_lt500_dom6x6_s${SD}_swaavg.pt
[ -f "$CK" ] || { echo "MISSING CHECKPOINT $CK"; exit 1; }

echo "### HS SELFRESP c40-199 ARRAY job=$SLURM_ARRAY_JOB_ID task=$SLURM_ARRAY_TASK_ID seed=$SD ###"
nvidia-smi -L; date
python -u scripts/dump_halfshear_selfresp.py \
  --ckpt $CK --min-case 40 --max-case 199 \
  --base-cache $CACHE --n-samples 32 --batch-size 16384 --blind \
  --out $OUTDIR/parts/part_s${SD}.feather \
  2>&1 | grep -v --line-buffered "module command" || { echo HSA_FAILED; exit 1; }
echo HSA_TASK_DONE; date
