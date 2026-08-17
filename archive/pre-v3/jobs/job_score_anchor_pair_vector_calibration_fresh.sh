#!/bin/bash
#SBATCH --job-name=abfr_paircal
#SBATCH --time=00:30:00
#SBATCH --mem=12G
#SBATCH --cpus-per-task=2
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --array=0-19%20
#SBATCH --output=/home/z/Zekang.Zhang/logs/abfr_paircal_%A_%a.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/abfr_paircal_%A_%a.err
set -euo pipefail
ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
ENV=/project/ls-gruen/users/zekang.zhang/envs/sims1
BASE=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_anchorblend_g005_c300-399
PARTS=/project/ls-gruen/users/zekang.zhang/sbsi_gap_anchor_paircal_c300-399
REFERENCE=$ROOT/results/anchorblend_g005_response_v22_c300-399.feather
CALIBRATION=$ROOT/results/halfshear_pair_vector_calibration_refitall_v22_c0-39.json
export PATH=$ENV/bin:$PATH
export LD_LIBRARY_PATH=$ENV/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=/home/z/Zekang.Zhang/blendemu:$ROOT:$ROOT/scripts:${PYTHONPATH:-}
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
mkdir -p "$PARTS"
cd "$ROOT"
START=$((300 + 5 * SLURM_ARRAY_TASK_ID))
END=$((START + 4))
for case in $(seq "$START" "$END"); do
  env SLURM_PROCID=0 python -u scripts/score_anchor_pair_vector_calibration.py \
    --base "$BASE" --reference "$REFERENCE" --calibration "$CALIBRATION" \
    --output-dir "$PARTS" --case-offset "$case" --n-cases 1
done
echo "ANCHOR_FRESH_PAIR_VECTOR_CALIBRATION_PART_DONE cases=$START-$END"
date
