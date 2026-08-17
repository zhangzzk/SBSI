#!/bin/bash
#SBATCH --job-name=abv2c_fullsm
#SBATCH --time=03:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --exclude=usm-cl-183r01,usm-cl-826bac01,usm-cl-826bac02,usm-cl-826bac03,usm-cl-1116cs01,usm-cl-1116cs02,usm-cl-1116cs03,usm-cl-seitz3
#SBATCH --output=/home/z/Zekang.Zhang/logs/abv2c_fullsm_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/abv2c_fullsm_%j.err
set -euo pipefail

ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
BE=/home/z/Zekang.Zhang/blendemu
ENV=/project/ls-gruen/users/zekang.zhang/envs/sims1
CFG=$ROOT/configs/fs2_lsst_r_anchorblend_g002_c400-499.yaml
BASE=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/anchor_v2_complement_smoke_case400
REFERENCE=$ROOT/results/anchorblend_v2_complement_smoke_case400.feather
SCORES=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/anchor_v2_complement_smoke_scores
MODEL=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v2_reweighted_vector_fixed_v1/weighted_model.json
MODEL_SHA=3cf70b6e74ad382f3ec59c6e8a2d0a5b9b0615d4c4677c7a71dd2344cbf35553

export PATH="$ENV/bin:$PATH"
export PYTHONPATH="$ROOT:$ROOT/scripts:$BE:${PYTHONPATH:-}"
export BLENDEMU_SIM_RUN=/home/z/Zekang.Zhang/MultiBand_ImSim/modules/Run.py
module load sextractor
export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
cd "$BE/scripts"
python -u run_pipeline.py --config "$CFG" --steps 2,3 --n-cases 1 \
  --case-offset 400 --n-mpi 1 --output-path "$BASE"
cd "$ROOT"
[ ! -e "$REFERENCE" ] || { echo "REFUSING existing $REFERENCE"; exit 1; }
python -u scripts/build_anchorblend_response.py \
  --base "$BASE" --cases 400 --g 0.02 --primary-domain v2-complement \
  --tags lsst_r_extnbr_indom_tuned --output "$REFERENCE"
mkdir -p "$SCORES"
python -u scripts/score_anchor_response_model.py \
  --base "$BASE" --reference "$REFERENCE" \
  --reference-baseline-column R_blend_lsst_r_extnbr_indom_tuned \
  --model-tag lsst_r_extnbr_indom_tuned --reg-file "$MODEL" \
  --expected-reg-sha256 "$MODEL_SHA" \
  --candidate-label v2_reweighted_vector_fixed_from_v22_trial15 \
  --output-dir "$SCORES" --case-offset 400 --n-cases 1 --sign 0.02
test -s "$REFERENCE"
test -s "$SCORES/case400.feather"
echo ANCHOR_V2_COMPLEMENT_FULL_SMOKE_DONE
