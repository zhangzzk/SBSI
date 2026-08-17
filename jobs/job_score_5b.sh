#!/bin/bash
#SBATCH --job-name=score_5b
#SBATCH --time=08:00:00
#SBATCH --mem=180G
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --partition=cip
#SBATCH --output=/home/z/Zekang.Zhang/logs/score_5b_%j.out
# INFERENCE.md §5B: the score/posterior route to the shear response, run on data.
#
#   MODE=closure    synthetic data drawn FROM the flow at a known shear -> ghat must
#                   return it, and Cov(ehat,s) must return the flow's transport R_flow.
#                   This is the unit test of Fisher's identity; it MUST pass before the
#                   constgold number means anything.
#   MODE=constgold  the certified catalogue.  Prediction from Gold-v1: bare ghat/g =
#                   R_sim/R_flow = 1.547, and with the §5C.3 blend injection 1.0025.
#
# Submit e.g.:  MODE=closure ROWS=1000000 sbatch jobs/job_score_5b.sh
#
# `cip` is often backed up.  Override the partition on the sbatch command line -- CLI
# flags beat the #SBATCH directives below:
#   MODE=... sbatch --partition=inter --gpus-per-node=a40:1 jobs/job_score_5b.sh
# Pin the GPU type on `inter`: it also holds rtx2080ti (11G) and p5000 (16G) nodes, which
# cannot hold a slab; a40 / a100 / h200nvl are all fine.
set -o pipefail
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
REPO=${REPO:-/home/z/Zekang.Zhang/SBSI/.claude/worktrees/inference-5b}
cd "$REPO" || exit 1
export PYTHONPATH="$REPO:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"

MODE=${MODE:-closure}
ROWS=${ROWS:-1000000}
GRID=${GRID:-61}
MODEL=${MODEL:-models/measurement_flow_g0_ngmix_meas_szfl_noz_lam450_fixresp_s501.pt}
CHUNK=${CHUNK:-1024}
EXTRA=${EXTRA:-}
CACHE_DIR=${CACHE_DIR:-/project/ls-gruen/users/zekang.zhang/sbsi_caches/score5b}
mkdir -p "$CACHE_DIR"

echo "### MODE=$MODE ROWS=$ROWS GRID=$GRID MODEL=$(basename $MODEL) ###"; date
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null
stdbuf -oL -eL python -u scripts/eval_score_response.py \
  --mode "$MODE" --measurement-model "$MODEL" \
  --max-rows "$ROWS" --grid-n "$GRID" --chunk "$CHUNK" \
  --dump "$CACHE_DIR/score5b_${MODE}_g${GRID}_r${ROWS}.npz" \
  $EXTRA 2>&1 | grep --line-buffered -vE "module command"
date
echo "### DONE ###"
