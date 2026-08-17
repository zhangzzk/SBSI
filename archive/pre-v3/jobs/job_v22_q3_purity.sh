#!/bin/bash
#SBATCH --job-name=v22q3rho
#SBATCH --array=0-9
#SBATCH --time=06:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=10
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/v22q3rho_%A_%a.%N.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/v22q3rho_%A_%a.%N.err

set -euo pipefail
ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
PY=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
CACHE=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk
TRUTH=$CACHE/v22_constgold_q3_decomp_c40-139_truth.feather
PILOT=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_v22cgq3_pilot_total
MAIN=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_v22cgq3_c48-139_total
TASK=${SLURM_ARRAY_TASK_ID:?array task required}
START=$((40 + 10 * TASK))
END=$((START + 9))
OUT=$CACHE/v22_constgold_q3_purity_adaptive_c${START}-${END}.feather
CASES=()
for ((case=START; case<=END; case++)); do CASES+=("$case"); done
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
cd "$ROOT"
"$PY" -u scripts/compute_v22_q3_purity.py \
  --truth "$TRUTH" --pilot-base "$PILOT" --main-base "$MAIN" \
  --cases "${CASES[@]}" --n-jobs 10 --output "$OUT"
echo "V22_Q3_PURITY_JOB_DONE cases=$START-$END"
